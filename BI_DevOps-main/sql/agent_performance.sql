-- agent_performance.sql
-- Detailed agent performance report with resolution rates and CSAT breakdown.
-- Uses the mv_agent_performance materialized view for performance.

SELECT
    ap.full_name,
    ap.team_name,
    ap.total_conversations,
    ap.avg_csat,
    ap.avg_handle_time_secs,
    ap.resolution_rate_pct,
    ap.escalated_count,
    -- Rank agents by CSAT within their team
    RANK() OVER (
        PARTITION BY ap.team_name
        ORDER BY ap.avg_csat DESC NULLS LAST
    ) AS team_csat_rank,
    -- Week-over-week conversation volume change
    ap.total_conversations - LAG(ap.total_conversations, 1) OVER (
        PARTITION BY ap.full_name
        ORDER BY ap.first_conversation_month
    ) AS wow_volume_delta
FROM mv_agent_performance ap
WHERE ap.total_conversations >= 10  -- Exclude agents with insufficient sample
ORDER BY ap.team_name, team_csat_rank;
