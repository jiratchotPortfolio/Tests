"""
test_sentinel.py  —  Sentinel RAG Pipeline Test Suite
Run:  python3 test_sentinel.py
"""

import sys, time, traceback

GRN="\033[92m"; RED="\033[91m"; CYN="\033[96m"
BLD="\033[1m";  DIM="\033[2m";  RST="\033[0m"
PASS_ICON=f"{GRN}✔{RST}"; FAIL_ICON=f"{RED}✘{RST}"

sys.path.insert(0, ".")
from sentinel_core import (
    strip_html, strip_boilerplate, redact_pii, normalise_unicode,
    estimate_tokens, validate_record, ValidationError, process_record,
    embed_text, embed_with_cache, clear_embed_cache, cosine_similarity,
    top_k, build_metadata_query,
)

_results = []
_suite = [""]

def suite(name): _suite[0] = name

def test(name, fn):
    t0 = time.perf_counter()
    try:
        fn()
        ms = (time.perf_counter() - t0) * 1000
        _results.append((_suite[0], name, True, f"{ms:.1f}ms"))
    except Exception:
        ms = (time.perf_counter() - t0) * 1000
        msg = traceback.format_exc(limit=2).strip().splitlines()[-1]
        _results.append((_suite[0], name, False, msg))

def eq(a, b, m=""): assert a == b, m or f"Expected {b!r}, got {a!r}"
def ok(c, m=""):    assert c, m or "Assertion failed"
def has(s, sub):    assert sub in s, f"{sub!r} not in {s!r}"
def not_has(s, sub):assert sub not in s, f"{sub!r} should NOT be in {s!r}"
def raises(exc, fn):
    try:
        fn(); raise AssertionError(f"Expected {exc.__name__} but nothing raised")
    except exc: pass


# ══════════════════════════════════════════
#  1. HTML Stripping
# ══════════════════════════════════════════
suite("HTML Stripping")

def test_strips_basic():
    r = strip_html("<b>Hello</b> <i>World</i>")
    has(r, "Hello"); has(r, "World"); not_has(r, "<b>")
test("strips basic bold / italic tags", test_strips_basic)

def test_strips_nested():
    eq(strip_html("<div><p>Customer message</p></div>").strip(), "Customer message")
test("strips nested block tags", test_strips_nested)

def test_strips_anchor():
    r = strip_html('<a href="http://x.com">Click here</a>')
    has(r, "Click here"); not_has(r, "<a")
test("strips anchor tags, preserves text", test_strips_anchor)

def test_plain_text():
    eq(strip_html("No tags here."), "No tags here.")
test("plain text passes through unchanged", test_plain_text)

def test_empty_string():
    eq(strip_html(""), "")
test("empty string returns empty string", test_empty_string)

def test_strips_style():
    not_has(strip_html("<style>.c{color:red}</style>Hello"), "<style>")
test("strips style tags", test_strips_style)


# ══════════════════════════════════════════
#  2. PII Redaction
# ══════════════════════════════════════════
suite("PII Redaction")

def test_email():
    r = redact_pii("Contact me at john.doe@example.com please")
    not_has(r, "@"); has(r, "[EMAIL]")
test("redacts email address", test_email)

def test_multi_email():
    eq(redact_pii("From a@b.com to c@d.org").count("[EMAIL]"), 2)
test("redacts multiple emails in one string", test_multi_email)

def test_phone():
    has(redact_pii("Call me at +66 81 234 5678"), "[PHONE]")
test("redacts phone number", test_phone)

def test_card():
    r = redact_pii("Card: 4111 1111 1111 1111")
    has(r, "[CARD]"); not_has(r, "4111 1111")
test("redacts credit card number", test_card)

def test_clean_text():
    eq(redact_pii("I need help with my booking."), "I need help with my booking.")
test("leaves clean text untouched", test_clean_text)


# ══════════════════════════════════════════
#  3. Boilerplate Removal
# ══════════════════════════════════════════
suite("Boilerplate Removal")

def test_confidential():
    not_has(
        strip_boilerplate("Hi. This email is confidential and intended solely for the recipient."),
        "confidential"
    )
test("removes confidentiality disclaimer", test_confidential)

def test_sent_from():
    not_has(strip_boilerplate("Great. Sent from my iPhone"), "iPhone")
test("removes 'Sent from my iPhone'", test_sent_from)

def test_ticket_ref():
    not_has(strip_boilerplate("Re: issue [Ticket #98765]"), "[Ticket #98765]")
test("removes ticket reference tags", test_ticket_ref)

def test_preserves_real():
    has(strip_boilerplate("My flight was delayed by 3 hours."), "delayed by 3 hours")
test("preserves real content", test_preserves_real)


# ══════════════════════════════════════════
#  4. Unicode Normalisation
# ══════════════════════════════════════════
suite("Unicode Normalisation")

def test_ligature():
    eq(normalise_unicode("\uFB02ight"), "flight")   # ﬂ (fl ligature) → fl
test("NFKC resolves ﬂ ligature to fl", test_ligature)

def test_fullwidth():
    eq(normalise_unicode("\uFF48\uFF45\uFF4C\uFF4C\uFF4F"), "hello")
test("NFKC normalises fullwidth chars", test_fullwidth)

def test_ascii_unchanged():
    eq(normalise_unicode("Hello World 123"), "Hello World 123")
test("plain ASCII passes through unchanged", test_ascii_unchanged)


# ══════════════════════════════════════════
#  5. Token Estimation
# ══════════════════════════════════════════
suite("Token Estimation")

def test_short_word():
    ok(1 <= estimate_tokens("Hello") <= 5)
test("short word returns 1-5 tokens", test_short_word)

def test_minimum():
    ok(estimate_tokens("Hi") >= 1)
test("minimum is always 1 token", test_minimum)

def test_scaling():
    eq(estimate_tokens("x" * 400), 100)
test("400-char string → exactly 100 tokens", test_scaling)

def test_empty_tokens():
    eq(estimate_tokens(""), 1)
test("empty string returns 1 token", test_empty_tokens)


# ══════════════════════════════════════════
#  6. Schema Validation
# ══════════════════════════════════════════
suite("Schema Validation")

VALID = {
    "external_ref_id": "REF-001", "agent_id": "agent-abc",
    "channel": "live_chat", "started_at": "2024-01-15T10:00:00Z",
    "resolution_status": "resolved",
    "turns": [{"speaker_role": "customer", "message_text": "Hello",
               "sent_at": "2024-01-15T10:00:01Z"}]
}

def test_valid_record():
    ok(validate_record(VALID) is not None)
test("valid record passes validation", test_valid_record)

def test_missing_turns():
    r = {k: v for k, v in VALID.items() if k != "turns"}
    raises(ValidationError, lambda: validate_record(r))
test("raises on missing 'turns' field", test_missing_turns)

def test_bad_channel():
    raises(ValidationError, lambda: validate_record({**VALID, "channel": "telegram"}))
test("raises on invalid channel value", test_bad_channel)

def test_bad_status():
    raises(ValidationError, lambda: validate_record({**VALID, "resolution_status": "unknown"}))
test("raises on invalid resolution_status", test_bad_status)

def test_bad_role():
    bad = {**VALID, "turns": [{"speaker_role": "robot", "message_text": "Hi",
                                "sent_at": "2024-01-15T10:00:01Z"}]}
    raises(ValidationError, lambda: validate_record(bad))
test("raises on invalid speaker_role", test_bad_role)

def test_all_channels():
    for ch in ("live_chat", "email", "phone", "in_app"):
        validate_record({**VALID, "channel": ch})
test("all four channel values accepted", test_all_channels)

def test_all_statuses():
    for s in ("resolved", "escalated", "abandoned", "pending"):
        validate_record({**VALID, "resolution_status": s})
test("all four resolution_status values accepted", test_all_statuses)


# ══════════════════════════════════════════
#  7. Full Pipeline
# ══════════════════════════════════════════
suite("Full Pipeline")

def make_record(**kw):
    base = {
        "external_ref_id": "REF-100", "agent_id": "agt-1",
        "channel": "email", "started_at": "2024-03-01T09:00:00Z",
        "resolution_status": "resolved",
        "turns": [
            {"speaker_role": "customer",
             "message_text": "<p>My booking <b>AB123</b> is wrong.</p>",
             "sent_at": "2024-03-01T09:00:01Z"},
            {"speaker_role": "agent",
             "message_text": "Let me check. Thank you for contacting Agoda support.",
             "sent_at": "2024-03-01T09:00:30Z"},
        ]
    }
    base.update(kw)
    return base

def test_pipeline_ok():
    ok(process_record(make_record()) is not None)
test("valid record returns cleaned dict", test_pipeline_ok)

def test_html_stripped():
    r = process_record(make_record())
    not_has(r["turns"][0]["message_text"], "<p>")
    not_has(r["turns"][0]["message_text"], "<b>")
test("HTML tags stripped from turn text", test_html_stripped)

def test_boilerplate_removed():
    r = process_record(make_record())
    not_has(r["turns"][1]["message_text"], "Thank you for contacting")
test("boilerplate removed from agent turn", test_boilerplate_removed)

def test_full_text_populated():
    ok(len(process_record(make_record())["full_text"]) > 0)
test("full_text field is populated", test_full_text_populated)

def test_checksum_format():
    r = process_record(make_record())
    eq(len(r["checksum"]), 64)
    ok(all(c in "0123456789abcdef" for c in r["checksum"]))
test("checksum is valid 64-char hex", test_checksum_format)

def test_invalid_returns_none():
    eq(process_record({"agent_id": "x", "channel": "email"}), None)
test("invalid record → None (dead-letter route)", test_invalid_returns_none)

def test_empty_turns_none():
    r = make_record(turns=[{"speaker_role": "agent", "message_text": "   ",
                             "sent_at": "2024-03-01T09:00:01Z"}])
    eq(process_record(r), None)
test("all-whitespace turns → None", test_empty_turns_none)

def test_pii_redacted_pipeline():
    r = process_record(make_record(turns=[{
        "speaker_role": "customer", "message_text": "Email me at user@test.com",
        "sent_at": "2024-03-01T09:00:01Z"
    }]))
    not_has(r["full_text"], "@"); has(r["full_text"], "[EMAIL]")
test("PII redacted in pipeline output", test_pii_redacted_pipeline)

def test_checksum_deterministic():
    eq(process_record(make_record())["checksum"], process_record(make_record())["checksum"])
test("checksum is deterministic across runs", test_checksum_deterministic)


# ══════════════════════════════════════════
#  8. Embedding & Cache
# ══════════════════════════════════════════
suite("Embedding & Cache")

def test_embed_dim():
    eq(len(embed_text("hello")), 8)
test("embed_text returns 8-dim vector", test_embed_dim)

def test_embed_range():
    ok(all(0.0 <= v <= 1.0 for v in embed_text("test sentence")))
test("all embedding values in [0.0, 1.0]", test_embed_range)

def test_embed_deterministic():
    eq(embed_text("booking issue"), embed_text("booking issue"))
test("same text → same vector (deterministic)", test_embed_deterministic)

def test_embed_different():
    ok(embed_text("flight delay") != embed_text("hotel complaint"))
test("different texts → different vectors", test_embed_different)

def test_cache_miss():
    clear_embed_cache()
    _, hit = embed_with_cache("unique text xyz 123")
    ok(not hit)
test("cache miss on first call", test_cache_miss)

def test_cache_hit():
    clear_embed_cache()
    embed_with_cache("repeat text abc")
    _, hit = embed_with_cache("repeat text abc")
    ok(hit)
test("cache hit on second call", test_cache_hit)

def test_cache_correct_value():
    clear_embed_cache()
    v1, _ = embed_with_cache("check me now")
    v2, _ = embed_with_cache("check me now")
    eq(v1, v2)
test("cache returns correct vector", test_cache_correct_value)


# ══════════════════════════════════════════
#  9. Vector Retrieval
# ══════════════════════════════════════════
suite("Vector Retrieval")

def test_self_similarity():
    v = embed_text("hello")
    ok(abs(cosine_similarity(v, v) - 1.0) < 1e-6)
test("identical vectors → similarity 1.0", test_self_similarity)

def test_zero_vector():
    eq(cosine_similarity([0.0] * 8, embed_text("anything")), 0.0)
test("zero vector → similarity 0.0", test_zero_vector)

def test_symmetry():
    a, b = embed_text("flight"), embed_text("delay")
    ok(abs(cosine_similarity(a, b) - cosine_similarity(b, a)) < 1e-9)
test("cosine similarity is symmetric", test_symmetry)

def test_topk_count():
    corpus = [(f"doc{i}", embed_text(f"document number {i}")) for i in range(10)]
    ok(len(top_k(embed_text("document number 3"), corpus, k=3)) == 3)
test("top_k returns exactly k results", test_topk_count)

def test_topk_exact_match():
    corpus = [
        ("target", embed_text("exact query match unique")),
        ("other1", embed_text("unrelated content about cats")),
        ("other2", embed_text("something entirely different here")),
    ]
    ok(top_k(embed_text("exact query match unique"), corpus, k=1)[0] == "target")
test("top_k returns exact match at rank 1", test_topk_exact_match)

def test_topk_overflow():
    corpus = [("a", embed_text("one")), ("b", embed_text("two"))]
    ok(len(top_k(embed_text("query"), corpus, k=10)) == 2)
test("top_k handles k > corpus size gracefully", test_topk_overflow)


# ══════════════════════════════════════════
#  10. SQL Query Builder
# ══════════════════════════════════════════
suite("SQL Query Builder")

def test_no_filter():
    sql, params = build_metadata_query({})
    not_has(sql, "WHERE"); eq(params, [])
test("no filters → no WHERE clause", test_no_filter)

def test_date_from():
    sql, params = build_metadata_query({"date_from": "2024-01-01"})
    has(sql, "WHERE"); has(sql, "started_at >="); eq(params, ["2024-01-01"])
test("date_from generates correct clause", test_date_from)

def test_channel_filter():
    sql, params = build_metadata_query({"channel": "email"})
    has(sql, "channel = %s"); eq(params, ["email"])
test("channel filter generated correctly", test_channel_filter)

def test_multi_filter():
    sql, params = build_metadata_query({"channel": "phone", "resolution_status": "escalated"})
    has(sql, "AND"); eq(len(params), 2)
test("multiple filters combined with AND", test_multi_filter)

def test_unknown_key():
    raises(ValueError, lambda: build_metadata_query({"unknown_key": "bad"}))
test("unknown filter key raises ValueError", test_unknown_key)

def test_all_valid_keys():
    _, params = build_metadata_query({
        "date_from": "2024-01-01", "date_to": "2024-12-31",
        "channel": "live_chat", "topic_id": 5, "resolution_status": "resolved"
    })
    eq(len(params), 5)
test("all five valid filter keys accepted", test_all_valid_keys)

def test_limit_present():
    sql, _ = build_metadata_query({})
    has(sql, "LIMIT 100")
test("query always includes LIMIT 100", test_limit_present)


# ══════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════
SUITE_ORDER = [
    "HTML Stripping", "PII Redaction", "Boilerplate Removal",
    "Unicode Normalisation", "Token Estimation", "Schema Validation",
    "Full Pipeline", "Embedding & Cache", "Vector Retrieval", "SQL Query Builder"
]

def print_report():
    print()
    print(f"{BLD}{'━' * 70}{RST}")
    print(f"{BLD}   SENTINEL  ·  RAG Pipeline Test Suite{RST}")
    print(f"{BLD}{'━' * 70}{RST}")

    current = ""; passed = failed = 0

    for suite_name, test_name, is_ok, meta in _results:
        if suite_name != current:
            current = suite_name
            idx = SUITE_ORDER.index(suite_name) + 1 if suite_name in SUITE_ORDER else "?"
            print(f"\n  {CYN}{BLD}  {idx:02d}.  {suite_name}{RST}")

        icon = PASS_ICON if is_ok else FAIL_ICON
        name_col = f"{test_name:<54}"
        meta_col = f"{DIM}{meta}{RST}" if is_ok else f"{RED}{meta}{RST}"
        print(f"    {icon}  {name_col} {meta_col}")
        passed += is_ok; failed += (not is_ok)

    total = passed + failed
    W = 62
    print()
    print(f"  {'━' * W}")
    if failed == 0:
        bar    = f"{GRN}{'█' * W}{RST}"
        status = f"{GRN}{BLD}  ✔  ALL {total} TESTS PASSED{RST}"
    else:
        g = int(W * passed / total) if total else 0
        bar    = f"{GRN}{'█' * g}{RST}{RED}{'█' * (W - g)}{RST}"
        status = f"{RED}{BLD}  ✘  {failed} FAILED  /  {passed} PASSED  /  {total} TOTAL{RST}"
    print(f"  {bar}")
    print(status)
    print(f"  {'━' * W}")
    print()
    return failed

if __name__ == "__main__":
    sys.exit(1 if print_report() else 0)
