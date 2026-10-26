-- Phase 8: approval-rate vs. expected-loss tradeoff swept across PD cutoffs.
-- Sweeps cutoff from 0.05 to 0.50 in 0.05 steps via generate_series, then
-- for each cutoff: approval_rate = share of applicants BELOW it (approved),
-- expected_loss = SUM(calibrated_pd * amt_credit) over the approved group
-- (a simplified expected-loss proxy - full loss-given-default weighting
-- belongs in Phase 9's cost-matrix DAX measure, not duplicated here).
WITH cutoffs AS (
    SELECT generate_series(0.05, 0.50, 0.05) AS cutoff
)
SELECT
    c.cutoff,
    COUNT(s.sk_id_curr)                                     AS approved_count,
    ROUND(COUNT(s.sk_id_curr)::numeric / (SELECT COUNT(*) FROM scored_applications), 4) AS approval_rate,
    ROUND(SUM(s.calibrated_pd * s.amt_credit)::numeric, 0)  AS expected_loss_proxy
FROM cutoffs c
LEFT JOIN scored_applications s ON s.calibrated_pd < c.cutoff
GROUP BY c.cutoff
ORDER BY c.cutoff;
