-- Business question: "can I trust the risk_segment column, or did
-- something go wrong between Python and Postgres?" Independently
-- recomputes the Low/Medium/High tertile cutoffs IN SQL (PERCENTILE_CONT,
-- not just CASE WHEN on hardcoded numbers) and cross-tabs the result
-- against the Python-written risk_segment column. A clean diagonal
-- (every row's python_segment equals its sql_segment) is the actual
-- proof that score_and_write.py's tertile logic did what it claims -
-- a data-quality check, not just a segmentation exercise.
WITH thresholds AS (
    SELECT
        percentile_cont(0.3333) WITHIN GROUP (ORDER BY calibrated_pd) AS low_cut,
        percentile_cont(0.6667) WITHIN GROUP (ORDER BY calibrated_pd) AS high_cut
    FROM scored_applications
),
sql_recomputed AS (
    SELECT
        s.sk_id_curr,
        s.calibrated_pd,
        s.risk_segment AS python_segment,
        CASE
            WHEN s.calibrated_pd <= t.low_cut THEN 'Low'
            WHEN s.calibrated_pd <= t.high_cut THEN 'Medium'
            ELSE 'High'
        END AS sql_segment
    FROM scored_applications s, thresholds t
)
SELECT python_segment, sql_segment, COUNT(*) AS applicant_count
FROM sql_recomputed
GROUP BY python_segment, sql_segment
ORDER BY python_segment, sql_segment;
