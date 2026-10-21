# HR Analytics & Attrition — SQL Business Insights Report
**Source Dataset:** `C:\Users\Arpan\Downloads\ML Final Project\ML-DE Final Project\employee-attrition-prediction\data\raw\WA_Fn-UseC_-HR-Employee-Attrition.csv` (1,470 records)
**Generated via:** SQLite 3 in-memory engine with Window Functions & CTEs

---
### Q1: Executive KPI Summary

*Overall workforce headcount, total leavers, company-wide turnover rate, and average tenure.*

```sql
SELECT 
                COUNT(*) AS total_headcount,
                SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS total_leavers,
                SUM(CASE WHEN Attrition = 'No' THEN 1 ELSE 0 END) AS active_headcount,
                ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct,
                ROUND(AVG(MonthlyIncome), 0) AS company_avg_monthly_income,
                ROUND(AVG(Age), 1) AS company_avg_age,
                ROUND(AVG(TotalWorkingYears), 1) AS company_avg_tenure_years
            FROM employees;
```

| total_headcount | total_leavers | active_headcount | attrition_rate_pct | company_avg_monthly_income | company_avg_age | company_avg_tenure_years |
| --- | --- | --- | --- | --- | --- | --- |
| 1470.0 | 237.0 | 1233.0 | 16.12 | 6503.0 | 36.9 | 11.3 |

---

### Q2: Attrition Rate by Department

*Departmental breakdown of headcount, turnover rate, and average monthly income.*

```sql
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
```

| Department | department_headcount | leavers | attrition_rate_pct | avg_monthly_income | avg_salary_hike_pct |
| --- | --- | --- | --- | --- | --- |
| Sales | 446 | 92 | 20.63 | 6959.0 | 15.1 |
| Human Resources | 63 | 12 | 19.05 | 6655.0 | 14.8 |
| Research & Development | 961 | 133 | 13.84 | 6281.0 | 15.3 |

---

### Q3: Job Role Turnover & Compensation Disparity

*Comparing salary of stayers vs leavers across roles with >= 10 employees.*

```sql
SELECT 
                JobRole,
                COUNT(*) AS total_staff,
                SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers_count,
                ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct,
                ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN MonthlyIncome END), 0) AS avg_leaver_income,
                ROUND(AVG(CASE WHEN Attrition = 'No' THEN MonthlyIncome END), 0) AS avg_stayer_income,
                ROUND(AVG(CASE WHEN Attrition = 'No' THEN MonthlyIncome END) - AVG(CASE WHEN Attrition = 'Yes' THEN MonthlyIncome END), 0) AS income_gap
            FROM employees
            GROUP BY JobRole
            HAVING COUNT(*) >= 10
            ORDER BY attrition_rate_pct DESC;
```

| JobRole | total_staff | leavers_count | attrition_rate_pct | avg_leaver_income | avg_stayer_income | income_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Sales Representative | 83 | 33 | 39.76 | 2365.0 | 2798.0 | 434.0 |
| Laboratory Technician | 259 | 62 | 23.94 | 2919.0 | 3337.0 | 418.0 |
| Human Resources | 52 | 12 | 23.08 | 3716.0 | 4392.0 | 676.0 |
| Sales Executive | 326 | 57 | 17.48 | 7489.0 | 6805.0 | -684.0 |
| Research Scientist | 292 | 47 | 16.1 | 2780.0 | 3328.0 | 548.0 |
| Manufacturing Director | 145 | 10 | 6.9 | 7366.0 | 7290.0 | -76.0 |
| Healthcare Representative | 131 | 9 | 6.87 | 8548.0 | 7454.0 | -1095.0 |
| Manager | 102 | 5 | 4.9 | 16797.0 | 17201.0 | 404.0 |
| Research Director | 80 | 2 | 2.5 | 19396.0 | 15947.0 | -3448.0 |

---

### Q4: Overtime & Work-Life Balance Burnout Matrix

*Evaluating turnover rates at the intersection of overtime and work-life balance satisfaction.*

```sql
SELECT 
                OverTime,
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
```

| OverTime | work_life_balance_label | total_employees | leavers | attrition_rate_pct |
| --- | --- | --- | --- | --- |
| Yes | 1 - Bad | 22 | 10 | 45.45 |
| Yes | 2 - Good | 104 | 32 | 30.77 |
| Yes | 3 - Better | 254 | 73 | 28.74 |
| Yes | 4 - Best | 36 | 12 | 33.33 |
| No | 1 - Bad | 58 | 15 | 25.86 |
| No | 2 - Good | 240 | 26 | 10.83 |
| No | 3 - Better | 639 | 54 | 8.45 |
| No | 4 - Best | 117 | 15 | 12.82 |

---

### Q5: Business Travel & Commute Distance Impact

*Impact of travel frequency combined with distance from home buckets.*

```sql
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
```

| BusinessTravel | commute_distance_bucket | total_employees | leavers | attrition_rate_pct |
| --- | --- | --- | --- | --- |
| Non-Travel | 01. Short (0-5 km) | 63 | 2 | 3.17 |
| Non-Travel | 02. Medium (6-15 km) | 48 | 4 | 8.33 |
| Non-Travel | 03. Long (16+ km) | 39 | 6 | 15.38 |
| Travel_Frequently | 01. Short (0-5 km) | 125 | 28 | 22.4 |
| Travel_Frequently | 02. Medium (6-15 km) | 86 | 21 | 24.42 |
| Travel_Frequently | 03. Long (16+ km) | 66 | 20 | 30.3 |
| Travel_Rarely | 01. Short (0-5 km) | 444 | 57 | 12.84 |
| Travel_Rarely | 02. Medium (6-15 km) | 375 | 57 | 15.2 |
| Travel_Rarely | 03. Long (16+ km) | 224 | 42 | 18.75 |

---

### Q6: Composite Workplace Satisfaction Matrix

*Environment satisfaction vs Job satisfaction cross-tabulation.*

```sql
SELECT 
                EnvironmentSatisfaction,
                JobSatisfaction,
                COUNT(*) AS total_employees,
                SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers,
                ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
            FROM employees
            GROUP BY EnvironmentSatisfaction, JobSatisfaction
            ORDER BY attrition_rate_pct DESC
            LIMIT 6;
```

| EnvironmentSatisfaction | JobSatisfaction | total_employees | leavers | attrition_rate_pct |
| --- | --- | --- | --- | --- |
| 1.0 | 1.0 | 53.0 | 20.0 | 37.74 |
| 1.0 | 2.0 | 47.0 | 16.0 | 34.04 |
| 2.0 | 1.0 | 55.0 | 14.0 | 25.45 |
| 1.0 | 3.0 | 100.0 | 22.0 | 22.0 |
| 4.0 | 1.0 | 87.0 | 16.0 | 18.39 |
| 3.0 | 1.0 | 94.0 | 16.0 | 17.02 |

---

### Q7: Window Function: Salary Equity & Internal Role Ranking

*Flags employees earning below their job role average using DENSE_RANK() and AVG() OVER (PARTITION BY).*

```sql
WITH RoleSalaryRanked AS (
                SELECT 
                    EmployeeNumber,
                    Department,
                    JobRole,
                    MonthlyIncome,
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
            ORDER BY JobRole, variance_from_role_avg ASC
            LIMIT 10;
```

| EmployeeNumber | Department | JobRole | MonthlyIncome | role_avg_income | variance_from_role_avg | salary_rank_in_role | total_peers_in_role | Attrition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 369 | Research & Development | Healthcare Representative | 4000 | 7529.0 | -3529.0 | 1 | 131 | No |
| 942 | Research & Development | Healthcare Representative | 4014 | 7529.0 | -3515.0 | 2 | 131 | No |
| 1501 | Research & Development | Healthcare Representative | 4035 | 7529.0 | -3494.0 | 3 | 131 | No |
| 1552 | Research & Development | Healthcare Representative | 4069 | 7529.0 | -3460.0 | 4 | 131 | No |
| 659 | Research & Development | Healthcare Representative | 4089 | 7529.0 | -3440.0 | 5 | 131 | No |
| 1062 | Research & Development | Healthcare Representative | 4107 | 7529.0 | -3422.0 | 6 | 131 | No |
| 1630 | Research & Development | Healthcare Representative | 4148 | 7529.0 | -3381.0 | 7 | 131 | No |
| 117 | Research & Development | Healthcare Representative | 4152 | 7529.0 | -3377.0 | 8 | 131 | No |
| 1022 | Research & Development | Healthcare Representative | 4240 | 7529.0 | -3289.0 | 9 | 131 | No |
| 807 | Research & Development | Healthcare Representative | 4244 | 7529.0 | -3285.0 | 10 | 131 | No |

---

### Q8: Promotion Stagnation vs Company Tenure

*Ratio of years since last promotion relative to total tenure at company.*

```sql
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
```

| Department | JobRole | avg_tenure_years | avg_years_since_promotion | promotion_stagnation_ratio | attrition_rate_pct |
| --- | --- | --- | --- | --- | --- |
| Research & Development | Manager | 13.8 | 5.6 | 0.43 | 5.66 |
| Human Resources | Manager | 16.3 | 4.2 | 0.37 | 0.0 |
| Sales | Sales Representative | 3.2 | 1.2 | 0.34 | 38.16 |
| Sales | Sales Executive | 7.7 | 2.6 | 0.32 | 17.03 |
| Research & Development | Healthcare Representative | 8.5 | 3.0 | 0.3 | 6.98 |
| Research & Development | Research Scientist | 5.3 | 1.6 | 0.3 | 15.96 |
| Research & Development | Research Director | 11.1 | 3.2 | 0.28 | 2.53 |
| Research & Development | Laboratory Technician | 5.2 | 1.5 | 0.27 | 22.18 |
| Sales | Manager | 15.6 | 4.2 | 0.27 | 5.56 |
| Human Resources | Human Resources | 5.3 | 1.3 | 0.26 | 23.08 |
| Research & Development | Manufacturing Director | 7.7 | 2.1 | 0.26 | 6.99 |

---

### Q9: CTE: High-Risk Active Employee Early Warning Cohort

*Filters currently active employees matching >= 3 burnout and turnover risk indicators.*

```sql
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
                    Attrition,
                    (CASE WHEN OverTime = 'Yes' THEN 1 ELSE 0 END) +
                    (CASE WHEN WorkLifeBalance <= 2 THEN 1 ELSE 0 END) +
                    (CASE WHEN JobSatisfaction <= 2 THEN 1 ELSE 0 END) +
                    (CASE WHEN YearsSinceLastPromotion >= 3 THEN 1 ELSE 0 END) +
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
                total_risk_signals,
                CASE 
                    WHEN total_risk_signals >= 4 THEN 'CRITICAL RISK'
                    WHEN total_risk_signals = 3 THEN 'HIGH RISK'
                    ELSE 'MODERATE RISK'
                END AS flight_risk_category
            FROM RiskSignals
            WHERE Attrition = 'No' AND total_risk_signals >= 3
            ORDER BY total_risk_signals DESC, MonthlyIncome ASC
            LIMIT 10;
```

| EmployeeNumber | Age | Department | JobRole | MonthlyIncome | OverTime | total_risk_signals | flight_risk_category |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1244 | 27 | Research & Development | Research Scientist | 2235 | Yes | 4 | CRITICAL RISK |
| 913 | 44 | Research & Development | Research Scientist | 2559 | Yes | 4 | CRITICAL RISK |
| 10 | 59 | Research & Development | Laboratory Technician | 2670 | Yes | 4 | CRITICAL RISK |
| 802 | 37 | Research & Development | Research Scientist | 2782 | Yes | 4 | CRITICAL RISK |
| 233 | 27 | Research & Development | Research Scientist | 3058 | Yes | 4 | CRITICAL RISK |
| 214 | 58 | Research & Development | Research Scientist | 3346 | Yes | 4 | CRITICAL RISK |
| 518 | 37 | Research & Development | Manufacturing Director | 4197 | Yes | 4 | CRITICAL RISK |
| 104 | 30 | Research & Development | Laboratory Technician | 5126 | Yes | 4 | CRITICAL RISK |
| 1251 | 31 | Research & Development | Healthcare Representative | 6833 | Yes | 4 | CRITICAL RISK |
| 605 | 42 | Research & Development | Manufacturing Director | 7406 | Yes | 4 | CRITICAL RISK |

---

### Q10: Retention Power of Equity (Stock Option Levels)

*Analyzing attrition rate across stock option levels (0 to 3).*

```sql
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
```

| StockOptionLevel | total_employees | leavers | attrition_rate_pct | avg_monthly_income | avg_experience_years |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 631.0 | 154.0 | 24.41 | 6188.0 | 10.7 |
| 1.0 | 596.0 | 56.0 | 9.4 | 7020.0 | 12.2 |
| 2.0 | 158.0 | 12.0 | 7.59 | 6173.0 | 10.3 |
| 3.0 | 85.0 | 15.0 | 17.65 | 5829.0 | 10.8 |

---

