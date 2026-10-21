import os
import sqlite3
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def sqlite_db():
    candidates = [
        os.path.join("data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv"),
        os.path.join(os.getcwd(), "employee-attrition-prediction", "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv"),
    ]
    raw_path = None
    for c in candidates:
        if os.path.exists(c):
            raw_path = c
            break
    if not raw_path:
        pytest.skip("Dataset file not found for SQL testing")
    df = pd.read_csv(raw_path)
    conn = sqlite3.connect(":memory:")
    df.to_sql("employees", conn, index=False, if_exists="replace")
    yield conn
    conn.close()


def test_executive_kpi_query(sqlite_db):
    query = """
    SELECT 
        COUNT(*) AS total_headcount,
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS total_leavers,
        ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS overall_attrition_rate_pct
    FROM employees;
    """
    df_res = pd.read_sql_query(query, sqlite_db)
    assert len(df_res) == 1
    assert df_res["total_headcount"].iloc[0] == 1470
    assert df_res["total_leavers"].iloc[0] == 237
    assert round(df_res["overall_attrition_rate_pct"].iloc[0], 1) == 16.1


def test_department_attrition_query(sqlite_db):
    query = """
    SELECT 
        Department,
        COUNT(*) AS department_headcount,
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers
    FROM employees
    GROUP BY Department;
    """
    df_res = pd.read_sql_query(query, sqlite_db)
    assert len(df_res) == 3
    dept_names = set(df_res["Department"].tolist())
    assert "Sales" in dept_names
    assert "Research & Development" in dept_names
    assert "Human Resources" in dept_names


def test_overtime_burnout_query(sqlite_db):
    query = """
    SELECT 
        OverTime,
        COUNT(*) AS total_count,
        ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
    FROM employees
    GROUP BY OverTime;
    """
    df_res = pd.read_sql_query(query, sqlite_db)
    ot_yes = df_res[df_res["OverTime"] == "Yes"]["attrition_rate_pct"].iloc[0]
    ot_no = df_res[df_res["OverTime"] == "No"]["attrition_rate_pct"].iloc[0]
    # Overtime attrition rate is significantly higher than non-overtime (~30.5% vs ~10.4%)
    assert ot_yes > ot_no * 2
