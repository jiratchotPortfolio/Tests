-- topic_trends.sql
-- Identifies emerging issue topics by comparing current week volume
-- to a 4-week rolling average. Flags topics with >50% spike.

WITH weekly AS (
    SELECT
        topic_id,
        topic_name,
        category_name,
        week_start,
        conversation_count,
        avg_csat,
        AVG(conversation_count) OVER (
            PARTITION BY topic_id
            ORDER BY week_start
            ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
        ) AS rolling_4wk_avg
    FROM mv_topic_weekly_trends
),
current_week AS (
    SELECT *
    FROM weekly
    WHERE week_start = DATE_TRUNC('week', CURRENT_DATE)::DATE
)
SELECT
    cw.topic_name,
    cw.category_name,
    cw.conversation_count                                   AS this_week,
    ROUND(cw.rolling_4wk_avg, 1)                           AS avg_4wk,
    cw.avg_csat,
    ROUND(
        (cw.conversation_count - cw.rolling_4wk_avg)
        / NULLIF(cw.rolling_4wk_avg, 0) * 100, 1
    )                                                       AS pct_change_vs_avg,
    CASE
        WHEN cw.conversation_count > cw.rolling_4wk_avg * 1.5 THEN 'SPIKE'
        WHEN cw.conversation_count < cw.rolling_4wk_avg * 0.5 THEN 'DROP'
        ELSE 'NORMAL'
    END                                                     AS volume_signal
FROM current_week cw
ORDER BY pct_change_vs_avg DESC NULLS LAST;
