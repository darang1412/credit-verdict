-- Phase 8: segment-level summary stats on the scored validation population.
-- avg_calibrated_pd is the model's own expected-default-rate estimate per
-- segment (a forward-looking portfolio view) - the backward-looking
-- accuracy check against the real TARGET label already happened in Phase
-- 5/6's MLflow-logged metrics, which is why TARGET isn't in this table.
SELECT
    risk_segment,
    COUNT(*)                              AS applicant_count,
    ROUND(AVG(calibrated_pd)::numeric, 4) AS avg_calibrated_pd,
    ROUND(AVG(predicted_pd)::numeric, 4)  AS avg_raw_predicted_pd,
    ROUND(AVG(amt_credit)::numeric, 0)    AS avg_credit_amount,
    ROUND(MIN(calibrated_pd)::numeric, 4) AS min_calibrated_pd,
    ROUND(MAX(calibrated_pd)::numeric, 4) AS max_calibrated_pd
FROM scored_applications
GROUP BY risk_segment
ORDER BY avg_calibrated_pd;
