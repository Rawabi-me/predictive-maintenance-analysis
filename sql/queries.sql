-- Query 1: Which product types fail most often?
-- Operational question: should some product types get more frequent inspections?
SELECT
    type,
    COUNT(*) AS records,
    SUM(machine_failure) AS failures,
    ROUND(100.0 * AVG(machine_failure), 2) AS failure_rate_pct
FROM machine_data
GROUP BY type
ORDER BY failure_rate_pct DESC;

-- Query 2: Do failed records show more tool wear than normal ones?
-- Operational question: is accumulated tool wear higher when a failure occurs?
SELECT
    machine_failure,
    COUNT(*) AS records,
    ROUND(AVG(tool_wear_min), 2) AS avg_tool_wear_min
FROM machine_data
GROUP BY machine_failure
ORDER BY machine_failure;

-- Query 3: How does the failure rate change with tool wear?
-- Operational question: at what wear level should a tool be replaced?
SELECT
    CASE
        WHEN tool_wear_min < 50  THEN '0-49'
        WHEN tool_wear_min < 100 THEN '50-99'
        WHEN tool_wear_min < 150 THEN '100-149'
        WHEN tool_wear_min < 200 THEN '150-199'
        ELSE '200+'
    END AS wear_range,
    COUNT(*) AS records,
    SUM(machine_failure) AS failures,
    ROUND(100.0 * AVG(machine_failure), 2) AS failure_rate_pct
FROM machine_data
GROUP BY wear_range
ORDER BY MIN(tool_wear_min);

-- Query 4: Which combinations of product type and tool wear are riskiest?
-- Operational question: where should inspections be concentrated first?
-- HAVING keeps only groups with at least 30 records and a failure rate above the overall rate.
WITH banded AS (
    SELECT
        type,
        CASE
            WHEN tool_wear_min < 100 THEN '0-99'
            WHEN tool_wear_min < 200 THEN '100-199'
            ELSE '200+'
        END AS wear_range,
        machine_failure
    FROM machine_data
)
SELECT
    type,
    wear_range,
    COUNT(*) AS records,
    SUM(machine_failure) AS failures,
    ROUND(100.0 * AVG(machine_failure), 2) AS failure_rate_pct
FROM banded
GROUP BY type, wear_range
HAVING COUNT(*) >= 30
   AND AVG(machine_failure) > (SELECT AVG(machine_failure) FROM machine_data)
ORDER BY failure_rate_pct DESC;

-- Query 5: How risky is it to combine above-average torque with high tool wear?
-- Operational question: do high load and a worn tool together deserve a priority inspection?
SELECT
    COUNT(*) AS records,
    SUM(machine_failure) AS failures,
    ROUND(100.0 * AVG(machine_failure), 2) AS failure_rate_pct
FROM machine_data
WHERE torque_nm > (SELECT AVG(torque_nm) FROM machine_data)
  AND tool_wear_min >= 200;