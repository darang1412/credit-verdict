-- Business question: "within each risk segment, who are the riskiest
-- applicants, and what's actually driving their risk?" - the SQL analogue
-- of the tutorial's "top 3 products per category" window-function pattern.
-- ROW_NUMBER (not RANK/DENSE_RANK) because ties on calibrated_pd shouldn't
-- collapse the top-3 down to fewer than 3 distinct rows per segment.
WITH ranked AS (
    SELECT
        sk_id_curr,
        risk_segment,
        calibrated_pd,
        amt_credit,
        top_shap_factor_1,
        ROW_NUMBER() OVER (
            PARTITION BY risk_segment
            ORDER BY calibrated_pd DESC
        ) AS risk_rank
    FROM scored_applications
)
SELECT risk_segment, risk_rank, sk_id_curr, calibrated_pd, amt_credit, top_shap_factor_1
FROM ranked
WHERE risk_rank <= 3
ORDER BY risk_segment, risk_rank;
