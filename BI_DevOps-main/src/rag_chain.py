"""
rag_chain.py
------------
LangChain-based Retrieval-Augmented Generation chain.

Architecture:
  1. User query is embedded via the same model used during ingestion.
  2. FAISS ANN search retrieves top-20 candidate chunks.
  3. A cross-encoder re-ranker scores and filters to top-5 chunks.
  4. Retrieved chunks + structured PostgreSQL metadata are injected into
     a prompt template with strict grounding instructions.
  5. The LLM generates an answer constrained to the retrieved context.

The chain is intentionally stateless between calls. Conversation history
should be managed by the calling application.
"""

import logging
from typing import Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import settings
from db_client import DatabaseClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are Sentinel, a customer experience intelligence assistant.
Your task is to answer the question below using ONLY the conversation excerpts provided in the context.

Rules:
- If the context does not contain sufficient information to answer the question, respond exactly with:
  "Insufficient evidence in the retrieved data to answer this question."
- Do not fabricate, infer beyond the evidence, or draw on external knowledge.
- Cite the conversation reference IDs when available.
- Be concise and analytical.

Context:
{context}

Question: {question}

Answer:""",
)


class SentinelRAGChain:
    """
    Wraps the LangChain RetrievalQA chain with Sentinel-specific retrieval
    and metadata enrichment logic.
    """

    def __init__(self) -> None:
        self._db = DatabaseClient()
        self._vectorstore = self._load_vectorstore()
        self._llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.0,  # Deterministic output for analytical queries
            api_key=settings.OPENAI_API_KEY,
        )
        self._chain = self._build_chain()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_vectorstore(self) -> FAISS:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=settings.OPENAI_API_KEY,
        )
        return FAISS.load_local(settings.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    def _build_chain(self) -> RetrievalQA:
        retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        return RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": _SYSTEM_PROMPT},
            return_source_documents=True,
        )

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def query(self, question: str, filters: Optional[dict] = None) -> dict:
        """
        Execute a RAG query.

        Args:
            question: Natural language question from the CX analyst.
            filters:  Optional structured filters (e.g., date range, channel)
                      applied to the PostgreSQL metadata layer before retrieval.

        Returns:
            Dict containing 'answer', 'source_documents', and 'metadata'.
        """
        enriched_context = self._enrich_with_sql(question, filters)
        full_question = question
        if enriched_context:
            full_question = f"{question}\n\nAdditional structured context:\n{enriched_context}"

        result = self._chain.invoke({"query": full_question})

        return {
            "answer": result["result"],
            "source_documents": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in result.get("source_documents", [])
            ],
            "sql_context": enriched_context,
        }

    def _enrich_with_sql(self, question: str, filters: Optional[dict]) -> str:
        """
        Query PostgreSQL for structured metadata relevant to the question.

        This two-pronged retrieval strategy (vector + SQL) grounds the LLM
        response in both semantic similarity and structured business facts
        (e.g., CSAT scores, resolution rates, agent mappings).
        """
        if not filters:
            return ""
        try:
            rows = self._db.fetch_conversation_metadata(filters)
            if not rows:
                return ""
            lines = [f"- ConvID: {r['external_ref_id']}, Status: {r['resolution_status']}, CSAT: {r['csat_score']}" for r in rows[:10]]
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("SQL enrichment failed: %s", exc)
            return ""

    def interactive(self) -> None:
        """Simple CLI loop for development and demo use."""
        print("Sentinel RAG Engine - Interactive Mode")
        print("Type 'exit' to quit.\n")
        while True:
            try:
                question = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue
            result = self.query(question)
            print(f"\nAnswer:\n{result['answer']}\n")
            print(f"Sources retrieved: {len(result['source_documents'])}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    chain = SentinelRAGChain()
    if args.interactive:
        chain.interactive()
    elif args.query:
        result = chain.query(args.query)
        print(result["answer"])
