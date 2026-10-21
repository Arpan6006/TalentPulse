# ============================================================
# Employee Attrition Prediction - API Schemas (Pydantic Models)
# ============================================================

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):
    """Schema for a single employee record."""
    Age: int = Field(32, ge=18, le=65, description="Employee age in years")
    BusinessTravel: str = Field("Travel_Rarely", description="Travel frequency (Travel_Rarely, Travel_Frequently, Non-Travel)")
    DailyRate: float = Field(800.0, ge=100, le=1500, description="Daily compensation rate")
    Department: str = Field("Research & Development", description="Organizational unit")
    DistanceFromHome: float = Field(8.0, ge=1, le=50, description="Commute distance in kilometers")
    Education: int = Field(3, ge=1, le=5, description="Education level (1=Below College to 5=Doctor)")
    EducationField: str = Field("Life Sciences", description="Discipline of education")
    EnvironmentSatisfaction: int = Field(3, ge=1, le=4, description="Workplace environment satisfaction (1=Low, 4=Very High)")
    Gender: str = Field("Male", description="Gender (Male, Female)")
    HourlyRate: float = Field(65.0, ge=30, le=120, description="Hourly compensation rate")
    JobInvolvement: int = Field(3, ge=1, le=4, description="Level of job involvement (1=Low, 4=Very High)")
    JobLevel: int = Field(2, ge=1, le=5, description="Organizational job hierarchy level (1-5)")
    JobRole: str = Field("Research Scientist", description="Job title / role specification")
    JobSatisfaction: int = Field(3, ge=1, le=4, description="Job satisfaction rating (1=Low, 4=Very High)")
    MaritalStatus: str = Field("Single", description="Marital status (Single, Married, Divorced)")
    MonthlyIncome: float = Field(5000.0, ge=1000, le=25000, description="Gross monthly income in USD")
    MonthlyRate: float = Field(14000.0, ge=1000, le=30000, description="Monthly billing rate")
    NumCompaniesWorked: int = Field(2, ge=0, le=12, description="Number of prior companies worked at")
    OverTime: str = Field("No", description="Overtime work requirement (Yes, No)")
    PercentSalaryHike: float = Field(15.0, ge=10, le=30, description="Percentage salary increase in latest review")
    PerformanceRating: int = Field(3, ge=1, le=4, description="Performance review rating (1=Low to 4=Outstanding)")
    RelationshipSatisfaction: int = Field(3, ge=1, le=4, description="Workplace interpersonal relationship rating (1-4)")
    StockOptionLevel: int = Field(1, ge=0, le=3, description="Equity grant level (0=No equity to 3=High equity)")
    TotalWorkingYears: float = Field(8.0, ge=0, le=45, description="Total professional career experience in years")
    TrainingTimesLastYear: int = Field(2, ge=0, le=10, description="Training sessions attended in the preceding year")
    WorkLifeBalance: int = Field(3, ge=1, le=4, description="Work-life balance satisfaction rating (1-4)")
    YearsAtCompany: float = Field(4.0, ge=0, le=45, description="Total tenure at current company in years")
    YearsInCurrentRole: float = Field(2.0, ge=0, le=25, description="Tenure in present job role in years")
    YearsSinceLastPromotion: float = Field(1.0, ge=0, le=20, description="Years elapsed since previous promotion")
    YearsWithCurrManager: float = Field(2.0, ge=0, le=20, description="Years under reporting structure of current manager")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 32,
                "BusinessTravel": "Travel_Rarely",
                "DailyRate": 800.0,
                "Department": "Research & Development",
                "DistanceFromHome": 8.0,
                "Education": 3,
                "EducationField": "Life Sciences",
                "EnvironmentSatisfaction": 3,
                "Gender": "Male",
                "HourlyRate": 65.0,
                "JobInvolvement": 3,
                "JobLevel": 2,
                "JobRole": "Research Scientist",
                "JobSatisfaction": 3,
                "MaritalStatus": "Single",
                "MonthlyIncome": 5000.0,
                "MonthlyRate": 14000.0,
                "NumCompaniesWorked": 2,
                "OverTime": "No",
                "PercentSalaryHike": 15.0,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 3,
                "StockOptionLevel": 1,
                "TotalWorkingYears": 8.0,
                "TrainingTimesLastYear": 2,
                "WorkLifeBalance": 3,
                "YearsAtCompany": 4.0,
                "YearsInCurrentRole": 2.0,
                "YearsSinceLastPromotion": 1.0,
                "YearsWithCurrManager": 2.0
            }
        }


class PredictionResponse(BaseModel):
    probability: float
    retention_confidence: float
    prediction: int
    verdict: str
    risk_tier: str
    financial_exposure: Dict[str, float]
    retention_roi: Dict[str, Any]
    retention_directive: str


class BatchPredictionRequest(BaseModel):
    records: List[EmployeeInput]


class BatchPredictionResponse(BaseModel):
    total_records: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    cohort_financial_summary: Dict[str, Any]
    predictions: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    feature_count: int
    api_version: str


class ValidationResponse(BaseModel):
    is_valid: bool
    errors: List[str]
    sanitized_record: Dict[str, Any]
