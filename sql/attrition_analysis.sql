-- ==============================================================================
-- HR ANALYTICS & EMPLOYEE ATTRITION - SQL BUSINESS ANALYSIS
-- Dataset: IBM HR Analytics Employee Attrition (1,470 records, 35 attributes)
-- Database Compatibility: SQLite, PostgreSQL, Snowflake, BigQuery, MySQL 8.0+
-- Table: employees
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- TIER 1: EXECUTIVE WORKFORCE KPIS (Core Headcount & Turnover Metrics)
-- ------------------------------------------------------------------------------

-- Query 1: Executive KPI Summary (Overall Headcount, Turnover Rate, Averages)
-- Purpose: Provides high-level snapshot for the Chief Human Resources Officer (CHRO)
SELECT 
    COUNT(*) AS total_headcount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS total_leavers,
    SUM(CASE WHEN Attrition = 'No' THEN 1 ELSE 0 END) AS active_headcount,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS overall_attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0) AS company_avg_monthly_income,
    ROUND(AVG(Age), 1) AS company_avg_age,
    ROUND(AVG(TotalWorkingYears), 1) AS company_avg_tenure_years
FROM employees;


-- Query 2: Attrition & Headcount Breakdown by Department
-- Purpose: Identifies departments with disproportionate turnover rates
SELECT 
    Department,
    COUNT(*) AS department_headcount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income,
    ROUND(AVG(PercentSalaryHike), 1) AS avg_salary_hike_pct
FROM employees
GROUP BY Department
ORDER BY attrition_rate_pct DESC;


-- Query 3: Attrition & Compensation Gap by Job Role
-- Purpose: Highlights specific job titles facing retention challenges and pay disparities
SELECT 
    Department,
    JobRole,
    COUNT(*) AS total_staff,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers_count,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN MonthlyIncome END), 0) AS avg_leaver_income,
    ROUND(AVG(CASE WHEN Attrition = 'No' THEN MonthlyIncome END), 0) AS avg_stayer_income,
    ROUND(AVG(CASE WHEN Attrition = 'No' THEN MonthlyIncome END) - AVG(CASE WHEN Attrition = 'Yes' THEN MonthlyIncome END), 0) AS income_gap
FROM employees
GROUP BY Department, JobRole
HAVING COUNT(*) >= 10
ORDER BY attrition_rate_pct DESC;


-- ------------------------------------------------------------------------------
-- TIER 2: ROOT-CAUSE DIAGNOSTICS (Burnout, Commute, and Satisfaction)
-- ------------------------------------------------------------------------------

-- Query 4: Overtime & Work-Life Balance Burnout Matrix
-- Purpose: Demonstrates the compounding effect of frequent overtime and low work-life balance
SELECT 
    OverTime,
    WorkLifeBalance,
    CASE WorkLifeBalance
        WHEN 1 THEN '1 - Bad'
        WHEN 2 THEN '2 - Good'
        WHEN 3 THEN '3 - Better'
        WHEN 4 THEN '4 - Best'
    END AS work_life_balance_label,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
FROM employees
GROUP BY OverTime, WorkLifeBalance
ORDER BY OverTime DESC, WorkLifeBalance ASC;


-- Query 5: Business Travel & Distance From Home Commute Impact
-- Purpose: Evaluates physical fatigue factors on turnover
SELECT 
    BusinessTravel,
    CASE 
        WHEN DistanceFromHome <= 5 THEN '01. Short (0-5 km)'
        WHEN DistanceFromHome <= 15 THEN '02. Medium (6-15 km)'
        ELSE '03. Long (16+ km)'
    END AS commute_distance_bucket,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
FROM employees
GROUP BY BusinessTravel, commute_distance_bucket
ORDER BY BusinessTravel, commute_distance_bucket;


-- Query 6: Composite Workplace Satisfaction Matrix
-- Purpose: Cross-analyzes Environment Satisfaction vs Job Satisfaction
SELECT 
    EnvironmentSatisfaction,
    JobSatisfaction,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
FROM employees
GROUP BY EnvironmentSatisfaction, JobSatisfaction
ORDER BY attrition_rate_pct DESC;


-- ------------------------------------------------------------------------------
-- TIER 3: ADVANCED ANALYTICS (Window Functions & Peer Benchmarking)
-- ------------------------------------------------------------------------------

-- Query 7: Peer Salary Equity & Dense Ranking within Job Roles
-- Purpose: Flags underpaid employees relative to their role peers using Window Functions
WITH RoleSalaryRanked AS (
    SELECT 
        EmployeeNumber,
        Department,
        JobRole,
        MonthlyIncome,
        YearsAtCompany,
        Attrition,
        ROUND(AVG(MonthlyIncome) OVER (PARTITION BY JobRole), 0) AS role_avg_income,
        DENSE_RANK() OVER (PARTITION BY JobRole ORDER BY MonthlyIncome ASC) AS salary_rank_in_role,
        COUNT(*) OVER (PARTITION BY JobRole) AS total_peers_in_role
    FROM employees
)
SELECT 
    EmployeeNumber,
    Department,
    JobRole,
    MonthlyIncome,
    role_avg_income,
    ROUND(MonthlyIncome - role_avg_income, 0) AS variance_from_role_avg,
    salary_rank_in_role,
    total_peers_in_role,
    Attrition
FROM RoleSalaryRanked
WHERE MonthlyIncome < role_avg_income
ORDER BY JobRole, variance_from_role_avg ASC;


-- Query 8: Promotion Stagnation vs Company Tenure
-- Purpose: Computes rolling ratio of years without promotion relative to total company tenure
SELECT 
    Department,
    JobRole,
    ROUND(AVG(YearsAtCompany), 1) AS avg_tenure_years,
    ROUND(AVG(YearsSinceLastPromotion), 1) AS avg_years_since_promotion,
    ROUND(AVG(CAST(YearsSinceLastPromotion AS FLOAT) / NULLIF(YearsAtCompany, 0)), 2) AS promotion_stagnation_ratio,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
FROM employees
WHERE YearsAtCompany > 0
GROUP BY Department, JobRole
ORDER BY promotion_stagnation_ratio DESC;


-- ------------------------------------------------------------------------------
-- TIER 4: ENTERPRISE ACTION (CTEs & Flight Risk Cohort Identification)
-- ------------------------------------------------------------------------------

-- Query 9: High-Risk Active Employee Early-Warning Cohort
-- Purpose: Proactively identifies currently active employees with >= 3 high-risk turnover signals
WITH RiskSignals AS (
    SELECT 
        EmployeeNumber,
        Age,
        Department,
        JobRole,
        MonthlyIncome,
        OverTime,
        WorkLifeBalance,
        JobSatisfaction,
        YearsSinceLastPromotion,
        YearsWithCurrManager,
        Attrition,
        -- Signal 1: Required overtime
        (CASE WHEN OverTime = 'Yes' THEN 1 ELSE 0 END) +
        -- Signal 2: Poor work-life balance
        (CASE WHEN WorkLifeBalance <= 2 THEN 1 ELSE 0 END) +
        -- Signal 3: Dissatisfied with job
        (CASE WHEN JobSatisfaction <= 2 THEN 1 ELSE 0 END) +
        -- Signal 4: Stagnant promotion (>= 3 years)
        (CASE WHEN YearsSinceLastPromotion >= 3 THEN 1 ELSE 0 END) +
        -- Signal 5: Low tenure with manager
        (CASE WHEN YearsWithCurrManager <= 1 THEN 1 ELSE 0 END) AS total_risk_signals
    FROM employees
)
SELECT 
    EmployeeNumber,
    Age,
    Department,
    JobRole,
    MonthlyIncome,
    OverTime,
    WorkLifeBalance,
    JobSatisfaction,
    YearsSinceLastPromotion,
    total_risk_signals,
    CASE 
        WHEN total_risk_signals >= 4 THEN 'CRITICAL RISK'
        WHEN total_risk_signals = 3 THEN 'HIGH RISK'
        ELSE 'MODERATE RISK'
    END AS flight_risk_category
FROM RiskSignals
WHERE Attrition = 'No' AND total_risk_signals >= 3
ORDER BY total_risk_signals DESC, MonthlyIncome ASC;


-- Query 10: Retention Power of Equity (Stock Option Levels)
-- Purpose: Evaluates whether equity grants serve as an effective golden handcuff
SELECT 
    StockOptionLevel,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income,
    ROUND(AVG(TotalWorkingYears), 1) AS avg_experience_years
FROM employees
GROUP BY StockOptionLevel
ORDER BY StockOptionLevel ASC;
