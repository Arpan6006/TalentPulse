# ==============================================================================
# Employee Attrition Prediction - Automated SQL Analytics Runner
# ==============================================================================
# Loads the raw employee dataset into an in-memory SQLite database,
# executes the 10 business analysis queries from sql/attrition_analysis.sql,
# prints formatted tables and business takeaways, and exports a report.
# ==============================================================================

import os
import sqlite3
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset Resolution & SQLite In-Memory Database Setup
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

POSSIBLE_CSV_PATHS = [
    os.path.join(PROJECT_ROOT, "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv"),
    os.path.join(os.path.dirname(PROJECT_ROOT), "WA_Fn-UseC_-HR-Employee-Attrition.csv"),
    os.path.join(PROJECT_ROOT, "WA_Fn-UseC_-HR-Employee-Attrition.csv"),
    os.path.join("data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv"),
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
]

def get_dataset_path() -> str:
    for path in POSSIBLE_CSV_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not locate WA_Fn-UseC_-HR-Employee-Attrition.csv. "
        "Please ensure it is located in data/raw/."
    )


def build_sqlite_database(csv_path: str) -> sqlite3.Connection:
    """Load CSV into an in-memory SQLite database as table 'employees'."""
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(":memory:")
    df.to_sql("employees", conn, index=False, if_exists="replace")
    return conn, len(df)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Query Definitions & Business Context
# ──────────────────────────────────────────────────────────────────────────────

QUERIES = [
    {
        "id": "Q1",
        "title": "Executive KPI Summary",
        "description": "Overall workforce headcount, total leavers, company-wide turnover rate, and average tenure.",
        "sql": """
            SELECT 
                COUNT(*) AS total_headcount,
                SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS total_leavers,
                SUM(CASE WHEN Attrition = 'No' THEN 1 ELSE 0 END) AS active_headcount,
                ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct,
                ROUND(AVG(MonthlyIncome), 0) AS company_avg_monthly_income,
                ROUND(AVG(Age), 1) AS company_avg_age,
                ROUND(AVG(TotalWorkingYears), 1) AS company_avg_tenure_years
            FROM employees;
        """
    },
    {
        "id": "Q2",
        "title": "Attrition Rate by Department",
        "description": "Departmental breakdown of headcount, turnover rate, and average monthly income.",
        "sql": """
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
        """
    },
    {
        "id": "Q3",
        "title": "Job Role Turnover & Compensation Disparity",
        "description": "Comparing salary of stayers vs leavers across roles with >= 10 employees.",
        "sql": """
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
        """
    },
    {
        "id": "Q4",
        "title": "Overtime & Work-Life Balance Burnout Matrix",
        "description": "Evaluating turnover rates at the intersection of overtime and work-life balance satisfaction.",
        "sql": """
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
        """
    },
    {
        "id": "Q5",
        "title": "Business Travel & Commute Distance Impact",
        "description": "Impact of travel frequency combined with distance from home buckets.",
        "sql": """
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
        """
    },
    {
        "id": "Q6",
        "title": "Composite Workplace Satisfaction Matrix",
        "description": "Environment satisfaction vs Job satisfaction cross-tabulation.",
        "sql": """
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
        """
    },
    {
        "id": "Q7",
        "title": "Window Function: Salary Equity & Internal Role Ranking",
        "description": "Flags employees earning below their job role average using DENSE_RANK() and AVG() OVER (PARTITION BY).",
        "sql": """
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
        """
    },
    {
        "id": "Q8",
        "title": "Promotion Stagnation vs Company Tenure",
        "description": "Ratio of years since last promotion relative to total tenure at company.",
        "sql": """
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
        """
    },
    {
        "id": "Q9",
        "title": "CTE: High-Risk Active Employee Early Warning Cohort",
        "description": "Filters currently active employees matching >= 3 burnout and turnover risk indicators.",
        "sql": """
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
        """
    },
    {
        "id": "Q10",
        "title": "Retention Power of Equity (Stock Option Levels)",
        "description": "Analyzing attrition rate across stock option levels (0 to 3).",
        "sql": """
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
        """
    }
]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Execution & Reporting
# ──────────────────────────────────────────────────────────────────────────────

def run_all_queries():
    csv_path = get_dataset_path()
    conn, total_rows = build_sqlite_database(csv_path)

    print("=" * 80)
    print("EMPLOYEE ATTRITION PREDICTION — SQL BUSINESS ANALYTICS RUNNER")
    print(f"Loaded {total_rows} employee records into SQLite in-memory database.")
    print("=" * 80)

    report_lines = [
        "# HR Analytics & Attrition — SQL Business Insights Report\n",
        f"**Source Dataset:** `{csv_path}` ({total_rows:,} records)\n",
        "**Generated via:** SQLite 3 in-memory engine with Window Functions & CTEs\n\n",
        "---\n"
    ]

    for q in QUERIES:
        print(f"\n[{q['id']}] {q['title']}")
        print(f"Goal: {q['description']}")
        print("-" * 80)
        
        df_result = pd.read_sql_query(q["sql"], conn)
        print(df_result.to_string(index=False))

        # Format dataframe as markdown table without external dependencies
        headers = df_result.columns.tolist()
        md_table = "| " + " | ".join(str(h) for h in headers) + " |\n"
        md_table += "| " + " | ".join("---" for _ in headers) + " |\n"
        for _, row in df_result.iterrows():
            md_table += "| " + " | ".join(str(val) for val in row.values) + " |\n"

        # Append to markdown report
        report_lines.append(f"### {q['id']}: {q['title']}\n\n")
        report_lines.append(f"*{q['description']}*\n\n")
        report_lines.append("```sql\n" + q["sql"].strip() + "\n```\n\n")
        report_lines.append(md_table)
        report_lines.append("\n---\n\n")

    conn.close()

    # Save output report
    report_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "sql_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Executed all {len(QUERIES)} SQL queries successfully!")
    print(f"[REPORT] Full Markdown business report written to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_queries()
