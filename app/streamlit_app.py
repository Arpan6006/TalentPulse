# ============================================================
# Employee Attrition Prediction - Executive Intelligence Portal
# ============================================================
# Modern SaaS UI & Explanatory Analytics Platform
#   - Professional dark aesthetics & glassmorphic containers
#   - Tree pathway & compounding SHAP waterfall walk
#   - 4-Pillar strategic retention matrix
#   - Relational SQL analytics engine (10 business queries)
#   - Bulk CSV scoring engine with live telemetry
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
import sqlite3
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# Ensure project root is on Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.roi_calculator import (
    calculate_flight_risk_exposure,
    calculate_retention_roi,
    compute_cohort_financial_impact
)
from src.data_validation import (
    validate_single_record,
    validate_batch_dataframe,
    SCHEMA_CONTRACT
)
from src.drift_detection import compute_feature_drift

# ──────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TalentPulse · Workforce Retention Platform",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────
# SVG Icon Tokens (Minimalist Line Art)
# ──────────────────────────────────────────────────────────
SVG_ACCURACY = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>'
SVG_CHART = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>'
SVG_USERS = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>'
SVG_LAYERS = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>'

SVG_USER_BOX = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#a5b4fc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>'
SVG_BRIEF_BOX = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#a5b4fc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>'
SVG_CLOCK_BOX = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#a5b4fc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>'
SVG_HEART_BOX = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#a5b4fc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>'
SVG_CHEVRON_DOWN = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>'
SVG_ALERT_CIRCLE = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'
SVG_SHIELD_CHECK = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg>'

# ──────────────────────────────────────────────────────────
# Comprehensive Modern Design System (CSS)
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Base Reset */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Ambient Dark Background with Radial Mesh Gradients */
    .stApp {
        background-color: #07090e !important;
        background-image: 
            radial-gradient(circle at 10% 8%, rgba(99, 102, 241, 0.10) 0%, transparent 40%),
            radial-gradient(circle at 90% 12%, rgba(168, 85, 247, 0.08) 0%, transparent 45%),
            radial-gradient(circle at 50% 90%, rgba(14, 165, 233, 0.06) 0%, transparent 55%) !important;
        background-attachment: fixed !important;
        color: #e2e8f0 !important;
    }

    /* Completely Remove Left Sidebar & Controls */
    [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    .stApp > header {
        display: none !important;
    }
    .anchor-link, [data-testid="stHeaderActionElements"], a[href^="#"] {
        display: none !important;
    }

    /* Typography Overrides */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em !important;
        font-weight: 700 !important;
    }

    /* Top Brand Navigation Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 24px;
    }
    .top-nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-logo-mark {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 0.95rem;
        color: #ffffff;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
    }
    .brand-name {
        font-size: 1.15rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
        line-height: 1.1;
    }
    .brand-tagline {
        font-size: 0.72rem;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .top-nav-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(18, 24, 38, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 9999px;
        font-size: 0.76rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .status-indicator-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
    }

    /* Hero Section */
    .hero-container {
        padding: 4px 0 16px 0;
        margin-bottom: 8px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 5px 12px;
        background: rgba(99, 102, 241, 0.10);
        border: 1px solid rgba(99, 102, 241, 0.28);
        border-radius: 9999px;
        color: #a5b4fc;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffffff 30%, #cbd5e1 70%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0 !important;
        line-height: 1.15 !important;
        letter-spacing: -0.03em !important;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        max-width: 860px;
        line-height: 1.55;
        margin-bottom: 22px;
    }

    /* Top Stats Ribbon */
    .stats-ribbon {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 24px;
    }
    @media (max-width: 900px) {
        .stats-ribbon { grid-template-columns: repeat(2, 1fr); }
    }
    .stat-pill {
        background: rgba(18, 24, 38, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        gap: 12px;
        backdrop-filter: blur(14px);
        transition: all 0.2s ease;
    }
    .stat-pill:hover {
        border-color: rgba(99, 102, 241, 0.35);
        background: rgba(25, 33, 52, 0.75);
    }
    .stat-pill-icon {
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: rgba(99, 102, 241, 0.10);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .stat-pill-val {
        font-size: 1.15rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
        line-height: 1.1;
    }
    .stat-pill-lbl {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(15, 20, 32, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 6px !important;
        backdrop-filter: blur(20px) !important;
        margin-bottom: 22px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 10px 22px !important;
        border: 1px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(139, 92, 246, 0.20)) !important;
        border: 1px solid rgba(99, 102, 241, 0.45) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.2) !important;
    }

    /* Bento Cards */
    .bento-card {
        background: rgba(15, 20, 32, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.35);
    }
    .bento-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 18px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .bento-icon {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(99, 102, 241, 0.10);
        border: 1px solid rgba(99, 102, 241, 0.22);
    }
    .bento-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
        margin: 0;
    }
    .bento-desc {
        font-size: 0.76rem;
        color: #94a3b8;
        margin: 0;
    }

    /* Streamlit Input Styling */
    .stNumberInput input, .stTextInput input {
        background: rgba(22, 29, 46, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        font-size: 0.92rem !important;
        transition: all 0.2s ease !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }
    div[data-baseweb="select"] > div {
        background: rgba(22, 29, 46, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"] > div:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }
    label[data-testid="stWidgetLabel"] {
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 4px !important;
    }

    /* Primary CTA Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 60%, #8b5cf6 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-size: 1.0rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        padding: 14px 28px !important;
        box-shadow: 0 4px 18px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(79, 70, 229, 0.5) !important;
    }

    /* Result Flight-Risk Cockpit */
    .cockpit-card-wrap {
        background: rgba(15, 20, 32, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        padding: 24px 28px;
        margin: 24px 0;
        backdrop-filter: blur(20px);
        box-shadow: 0 15px 40px -10px rgba(0, 0, 0, 0.6);
        position: relative;
        overflow: hidden;
    }
    .cockpit-top-bar {
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #a855f7);
    }
    .cockpit-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 20px;
    }
    .cockpit-title-group h2 {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        color: #ffffff !important;
    }
    .cockpit-title-group p {
        font-size: 0.82rem;
        color: #94a3b8;
        margin: 4px 0 0 0;
    }

    /* Dynamic Flight Risk Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.86rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .status-badge-high {
        background: rgba(244, 63, 94, 0.12);
        border: 1px solid rgba(244, 63, 94, 0.4);
        color: #fb7185;
    }
    .status-badge-medium {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fbbf24;
    }
    .status-badge-low {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
    }

    /* Metric Display Grid inside Result Cockpit */
    .cockpit-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin: 18px 0;
    }
    .cockpit-stat-box {
        background: rgba(22, 29, 46, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 16px 18px;
    }
    .cockpit-stat-lbl {
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .cockpit-stat-val {
        font-size: 2.1rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        line-height: 1;
    }

    /* Progress Track */
    .glow-track {
        background: rgba(255, 255, 255, 0.06);
        height: 12px;
        border-radius: 999px;
        overflow: hidden;
        position: relative;
        margin: 16px 0 8px 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .glow-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Prescriptive Advice Box */
    .cockpit-advice {
        background: rgba(99, 102, 241, 0.07);
        border-left: 3px solid #6366f1;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        color: #cbd5e1;
        font-size: 0.90rem;
        line-height: 1.5;
        margin-top: 18px;
    }

    /* Decision Tree / Primary Reasons Styling */
    .reasons-banner {
        background: rgba(18, 24, 38, 0.85);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 22px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .reasons-banner-risk {
        border-left: 4px solid #f43f5e !important;
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.08), rgba(18, 24, 38, 0.85)) !important;
    }
    .reasons-banner-protect {
        border-left: 4px solid #10b981 !important;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(18, 24, 38, 0.85)) !important;
    }
    .reason-item-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 10px;
        transition: all 0.2s ease;
    }
    .reason-item-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.12);
    }
    .tree-card {
        background: rgba(18, 24, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 4px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        backdrop-filter: blur(14px);
    }
    .tree-card-split { border-left: 3px solid #6366f1 !important; }
    .tree-card-risk { border-left: 3px solid #f43f5e !important; }
    .tree-card-protect { border-left: 3px solid #10b981 !important; }
    .tree-card-leaf-leave {
        border-left: 4px solid #f43f5e !important;
        background: rgba(244, 63, 94, 0.08) !important;
        border-color: rgba(244, 63, 94, 0.25) !important;
    }
    .tree-card-leaf-stay {
        border-left: 4px solid #10b981 !important;
        background: rgba(16, 185, 129, 0.08) !important;
        border-color: rgba(16, 185, 129, 0.25) !important;
    }
    .tree-arrow {
        text-align: center;
        margin: 4px 0;
        opacity: 0.8;
    }

    /* 4-Pillar Scorecard */
    .pillar-card {
        background: rgba(18, 24, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        backdrop-filter: blur(14px);
    }
    .pillar-icon-wrap {
        width: 36px;
        height: 36px;
        margin: 0 auto 8px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: rgba(99, 102, 241, 0.10);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .pillar-title {
        font-weight: 700;
        margin: 6px 0;
        color: #ffffff;
        font-size: 0.92rem;
        font-family: 'Outfit', sans-serif;
    }

    /* Download button */
    .stDownloadButton > button {
        background: rgba(99, 102, 241, 0.12) !important;
        color: #a5b4fc !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(99, 102, 241, 0.25) !important;
        border-color: #6366f1 !important;
        color: #ffffff !important;
    }

    /* Dataframe Overrides */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Load Artifacts
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("models/best_model.pkl")
    preprocessor  = joblib.load("models/preprocessor.pkl")
    feature_names = joblib.load("data/processed/feature_names.pkl")
    return model, preprocessor, feature_names


@st.cache_resource
def build_shap_explainer(_model):
    """Build SHAP explainer on-the-fly from the model."""
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return None


@st.cache_data
def load_results():
    path = "models/results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_resource
def get_sql_database():
    possible_paths = [
        os.path.join("data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv"),
        os.path.join("..", "WA_Fn-UseC_-HR-Employee-Attrition.csv"),
        "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                conn = sqlite3.connect(":memory:", check_same_thread=False)
                df.to_sql("employees", conn, index=False, if_exists="replace")
                return conn, len(df), df
            except Exception:
                pass
    return None, 0, None


model_loaded = False
try:
    model, preprocessor, feature_names = load_artifacts()
    explainer = build_shap_explainer(model)
    model_loaded = True
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.info("Make sure these files exist:\n- models/best_model.pkl\n- models/preprocessor.pkl\n- data/processed/feature_names.pkl")


# ──────────────────────────────────────────────────────────
# Feature Engineering (mirrors preprocessing pipeline)
# ──────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["YearsAtOtherCompanies"] = df["TotalWorkingYears"] - df["YearsAtCompany"]
    df["IncomePerYearExp"]       = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
    df["RoleStability"]          = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
    df["PromotionGap"]           = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]
    df["AgeWhenStarted"]         = df["Age"] - df["TotalWorkingYears"]
    return df


def classify_risk(probability: float):
    if probability >= 0.65:
        return "Critical Flight Risk", "status-badge-high", "Immediate intervention recommended: schedule a priority retention review, examine compensation alignment, and evaluate workload reallocation.", "#f43f5e"
    elif probability >= 0.40:
        return "Moderate Risk", "status-badge-medium", "Active monitoring recommended: schedule a career growth discussion, evaluate role trajectory, and review overtime demands.", "#f59e0b"
    else:
        return "Low Risk", "status-badge-low", "Employee exhibits strong retention stability: continue regular leadership mentorship and standard recognition programs.", "#10b981"


def get_factor_details(feat_name: str, went_left: bool, val: float, delta_prob: float = 0.0) -> dict:
    """
    Translates raw decision tree split conditions into human-friendly business reasons,
    explaining exactly why the factor causes employees to stay or leave.
    """
    clean_name = (
        feat_name.replace("Department_", "")
        .replace("JobRole_", "")
        .replace("BusinessTravel_", "")
        .replace("EducationField_", "")
        .replace("MaritalStatus_", "")
        .replace("_", " ")
    )
    title = clean_name.title()
    condition = f"{clean_name}: Below average" if went_left else f"{clean_name}: Normal"
    reason = "This organizational factor directly influences employee engagement and career stability."

    if "OverTime" in feat_name:
        title = "Overtime & Workload Hours"
        is_ot = ("Yes" in feat_name and not went_left) or ("No" in feat_name and went_left)
        if is_ot:
            condition = "Works Frequent Overtime: YES"
            reason = "Working routine overtime without adequate rest is the single strongest driver of chronic burnout across the company, tripling departure rates."
        else:
            condition = "Works Overtime: NO"
            reason = "Standard 40-hour work weeks protect employee stamina and support long-term retention."

    elif "MonthlyIncome" in feat_name or "IncomePerYearExp" in feat_name:
        title = "Compensation & Pay Equity"
        if went_left:
            condition = "Monthly Salary: Below Role & Market Benchmark"
            reason = "Compensation that trails peer averages creates perceived under-valuation, making the employee significantly more receptive to external recruiters."
        else:
            condition = "Monthly Salary: Competitive / Above Peer Average"
            reason = "Competitive compensation acts as a strong financial anchor that discourages external job hunting."

    elif "PercentSalaryHike" in feat_name:
        title = "Salary Merit Increase"
        if went_left:
            condition = "Salary Growth: Modest Annual Merit Hike"
            reason = "Below-average merit raises fail to outpace inflation and prompt employees to test external market valuation."
        else:
            condition = "Salary Growth: Generous Merit Hike (>= 15%)"
            reason = "Substantial merit increases strongly validate personal contribution and reinforce financial loyalty."

    elif "YearsSinceLastPromotion" in feat_name or "PromotionGap" in feat_name:
        title = "Promotion Velocity & Advancement"
        if not went_left:
            condition = "Promotion Status: Stagnant (3+ Years Without Upward Advancement)"
            reason = "Going several years without title progression creates career frustration and signals limited internal growth potential."
        else:
            condition = "Promotion Status: Active Progression (< 3 Years Ago)"
            reason = "Recent recognition and title progression reinforce career growth within the company."

    elif "StockOptionLevel" in feat_name:
        title = "Equity Grants & Long-Term Incentives"
        if went_left:
            condition = "Equity Ownership: Level 0 (Zero Stock Options Granted)"
            reason = "Zero equity ownership leaves the employee without any financial vesting schedule tying them to the company's long-term enterprise growth."
        else:
            condition = "Equity Ownership: Granted (Level 1+ Equity)"
            reason = "Unvested company equity acts as an effective financial retention golden handcuff."

    elif "JobLevel" in feat_name:
        title = "Job Seniority & Responsibility Level"
        if went_left:
            condition = "Job Level: Entry / Junior Tier (Level 1-2)"
            reason = "Lower hierarchy levels carry higher baseline turnover due to rapid career switching and lower sunk costs."
        else:
            condition = "Job Level: Senior Leadership Tier (Level 3+)"
            reason = "Higher corporate hierarchy levels provide executive influence and stronger retention stability."

    elif "TotalWorkingYears" in feat_name or "Age" in feat_name:
        title = "Career Stage & Experience Profile"
        if went_left:
            condition = "Experience Profile: Early-Career / Junior Staff"
            reason = "Early-career employees have higher market curiosity and lower switching friction before settling into a long-term company."
        else:
            condition = "Experience Profile: Established Senior Professional"
            reason = "Experienced professionals typically prioritize established team roots and organizational stability."

    elif "PerformanceRating" in feat_name:
        title = "Performance Appraisal Score"
        if not went_left:
            condition = "Appraisal: Top Performer (Rating 4)"
            reason = "Top-tier performers face higher external headhunting and aggressive recruitment from competitors."
        else:
            condition = "Appraisal: Solid Contributor (Rating 3)"
            reason = "Steady contributors provide stable operational output."

    elif "Education" in feat_name:
        title = "Educational Attainment"
        if not went_left:
            condition = "Education: Advanced Degree (Master's / Doctorate)"
            reason = "Employees with higher degrees possess broader market mobility and specialized outside opportunities."
        else:
            condition = "Education: Bachelor's / Associate"
            reason = "Educational qualifications align well with current role scope."

    elif "WorkLifeBalance" in feat_name:
        title = "Work-Life Balance"
        if went_left:
            condition = "Work-Life Balance: Poor / Strained (Rating 1 or 2)"
            reason = "Chronic interference between job demands and personal life leads to mental exhaustion, burnout, and voluntary exit."
        else:
            condition = "Work-Life Balance: Healthy & Sustainable (Rating 3 or 4)"
            reason = "Healthy work-life balance is a cornerstone of employee psychological well-being and loyalty."

    elif "JobSatisfaction" in feat_name:
        title = "Daily Role Satisfaction"
        if went_left:
            condition = "Role Satisfaction: Low (Rating 1 or 2)"
            reason = "Day-to-day dissatisfaction with daily responsibilities leads to disengagement and early exit signals."
        else:
            condition = "Role Satisfaction: High (Rating 3 or 4)"
            reason = "Enjoyment of day-to-day work tasks provides strong intrinsic motivation to remain with the team."

    elif "EnvironmentSatisfaction" in feat_name:
        title = "Workplace Environment & Culture"
        if went_left:
            condition = "Workplace Environment: Dissatisfied (Rating 1 or 2)"
            reason = "Frictions with the physical workspace, team climate, or internal tools create constant daily frustration."
        else:
            condition = "Workplace Environment: Supportive Culture (Rating 3 or 4)"
            reason = "A constructive work environment and collaborative culture foster deep team attachment."

    elif "JobInvolvement" in feat_name:
        title = "Role Ownership & Involvement"
        if went_left:
            condition = "Job Involvement: Disengaged (Rating 1 or 2)"
            reason = "Low emotional investment in projects often serves as an early behavioral symptom preceding resignation."
        else:
            condition = "Job Involvement: High Ownership (Rating 3 or 4)"
            reason = "Deep ownership and pride in organizational outcomes creates natural retention stickiness."

    elif "RelationshipSatisfaction" in feat_name:
        title = "Peer & Team Relationships"
        if went_left:
            condition = "Team Camaraderie: Strained (Rating 1 or 2)"
            reason = "Interpersonal friction with coworkers or stakeholders degrades daily job satisfaction and isolates the employee."
        else:
            condition = "Team Camaraderie: Strong & Collaborative (Rating 3 or 4)"
            reason = "Close peer friendships and psychological safety are among the strongest non-monetary retention drivers."

    elif "YearsWithCurrManager" in feat_name:
        title = "Manager-Employee Relationship"
        if went_left:
            condition = "Manager Relationship: New (< 2 Years Together)"
            reason = "New manager-report pairings lack established psychological safety and trust, representing the most vulnerable transition window."
        else:
            condition = "Manager Relationship: Proven Rapport (>= 2 Years Together)"
            reason = "A trusted direct manager relationship is one of the strongest protective anchors against departure."

    elif "YearsInCurrentRole" in feat_name:
        title = "Role Tenure & Monotony"
        if not went_left:
            condition = "Role Duration: Extended (>= 5 Years in Exact Same Position)"
            reason = "Remaining in the identical position for many years without scope expansion breeds monotony and restlessness."
        else:
            condition = "Role Duration: Dynamic (< 5 Years in Role)"
            reason = "Recent role placement provides fresh challenges and learning curves."

    elif "YearsAtCompany" in feat_name:
        title = "Company Tenure & Loyalty"
        if went_left:
            condition = "Tenure: New Employee (< 3 Years at Firm)"
            reason = "New joiners have lower switching friction and less organizational capital invested in the company."
        else:
            condition = "Tenure: Established Veteran (>= 3 Years at Firm)"
            reason = "Longer institutional tenure signals strong social integration and alignment with corporate culture."

    elif "DistanceFromHome" in feat_name:
        title = "Commute Distance & Fatigue"
        if not went_left:
            condition = "Daily Commute: Long Distance (>= 15 km)"
            reason = "Extended commute times cause cumulative daily stress and make local job alternatives significantly more appealing."
        else:
            condition = "Daily Commute: Short & Convenient (< 15 km)"
            reason = "A short commute preserves personal time and prevents transit fatigue."

    elif "NumCompaniesWorked" in feat_name:
        title = "Historical Career Mobility"
        if not went_left:
            condition = "Prior Track Record: High Job Mobility (Worked at 4+ Companies)"
            reason = "Employees with a history of frequent company moves have lower hesitation to resign for new outside opportunities."
        else:
            condition = "Prior Track Record: High Loyalty (< 4 Companies)"
            reason = "A track record of sustained tenures demonstrates a preference for long-term organizational continuity."

    elif "BusinessTravel" in feat_name:
        title = "Business Travel Demands"
        if "Frequently" in feat_name and not went_left:
            condition = "Travel Demands: Heavy (Travels Frequently)"
            reason = "Exhausting travel itineraries disrupt family life and personal routines, prompting searches for domestic roles."
        else:
            condition = "Travel Demands: Light or None"
            reason = "Low travel frequency provides predictable scheduling and daily routine stability."

    elif "TrainingTimesLastYear" in feat_name:
        title = "Professional Upskilling & Training"
        if went_left:
            condition = "Skill Training: Limited (< 2 Sessions Last Year)"
            reason = "A lack of structured learning opportunities makes ambitious talent feel their career growth has plateaued."
        else:
            condition = "Skill Training: Continuous Investment (>= 2 Sessions)"
            reason = "Employer-sponsored skill development signals long-term commitment to the employee's career capital."

    elif "MaritalStatus" in feat_name:
        status = feat_name.replace("MaritalStatus_", "")
        title = "Life Stage & Mobility Profile"
        if not went_left:
            condition = f"Marital Status: {status}"
            reason = f"Historical data reflects distinct career mobility and relocation patterns for {status.lower()} employees."
        else:
            condition = f"Marital Status: Not {status}"
            reason = "Personal life stage interacts with financial commitments and career transition flexibility."

    elif "JobRole" in feat_name:
        role = feat_name.replace("JobRole_", "")
        title = "Job Function Market Dynamics"
        if not went_left:
            condition = f"Job Title: {role}"
            reason = f"The {role} talent market experiences intense competitor recruitment and higher baseline industry turnover."
        else:
            condition = f"Job Title: Other than {role}"
            reason = "Role-specific retention varies based on external industry hiring demand."

    elif "Department" in feat_name:
        dept = feat_name.replace("Department_", "")
        title = "Departmental Dynamics"
        if not went_left:
            condition = f"Department: {dept}"
            reason = f"Departmental turnover benchmarks reflect industry attrition patterns in {dept}."
        else:
            condition = f"Department: Other than {dept}"
            reason = "Organizational structure and department headcount impact retention stability."

    is_risk = delta_prob > 0.005
    is_protect = delta_prob < -0.005

    return {
        "title": title,
        "condition": condition,
        "reason": reason,
        "is_risk": is_risk,
        "is_protect": is_protect
    }


def format_rule_condition(feat_name: str, went_left: bool, val: float) -> str:
    """Backward compatibility wrapper."""
    return get_factor_details(feat_name, went_left, val)["condition"]


def clean_html(raw: str) -> str:
    """Strips leading whitespace from every line so CommonMark never treats HTML as an indented code block."""
    return "\n".join(line.strip() for line in raw.strip().splitlines())


# ──────────────────────────────────────────────────────────
# Top Navigation Bar & Hero Section
# ──────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-nav">
    <div class="top-nav-brand">
        <div class="brand-logo-mark">TP</div>
        <div>
            <div class="brand-name">TalentPulse</div>
            <div class="brand-tagline">Workforce Retention Intelligence</div>
        </div>
    </div>
    <div class="top-nav-status">
        <span class="status-indicator-dot"></span>
        <span>Ensemble Model Operational · IBM HR Benchmark Calibration</span>
    </div>
</div>

<div class="hero-container">
    <div class="hero-badge">
        Workforce Retention Engine
    </div>
    <h1 class="hero-title">Predictive Workforce Retention Platform</h1>
    <p class="hero-subtitle">
        Evaluate employee attrition risk with high-precision machine learning. Uncover the compounding factors driving turnover through transparent decision pathways, SHAP risk walks, and relational SQL cohort intelligence.
    </p>
    <div class="stats-ribbon">
        <div class="stat-pill">
            <div class="stat-pill-icon">{SVG_CHART}</div>
            <div>
                <div class="stat-pill-val">0.814</div>
                <div class="stat-pill-lbl">ROC-AUC Power</div>
            </div>
        </div>
        <div class="stat-pill">
            <div class="stat-pill-icon">{SVG_ACCURACY}</div>
            <div>
                <div class="stat-pill-val">78.7%</div>
                <div class="stat-pill-lbl">Leaver Recall</div>
            </div>
        </div>
        <div class="stat-pill">
            <div class="stat-pill-icon">{SVG_LAYERS}</div>
            <div>
                <div class="stat-pill-val">τ* = 0.38</div>
                <div class="stat-pill-lbl">Cost-Optimal Cutoff</div>
            </div>
        </div>
        <div class="stat-pill">
            <div class="stat-pill-icon">{SVG_USERS}</div>
            <div>
                <div class="stat-pill-val">1,470</div>
                <div class="stat-pill-lbl">Calibrated Baseline</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Collapsible Model Performance Verification
results = load_results()
if results:
    with st.expander("Model Validation Benchmarks & Performance Metrics", expanded=False):
        df_results = pd.DataFrame(results)[["Model", "ROC-AUC", "Recall", "F1-Score"]]
        df_results = df_results.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        st.caption("Benchmark evaluations conducted on stratified holdout validation test cohort.")


# ──────────────────────────────────────────────────────────
# Primary Navigation Tabs
# ──────────────────────────────────────────────────────────
tab_single, tab_batch, tab_sql, tab_data_quality = st.tabs([
    "Individual Flight Risk Assessment", 
    "Batch Cohort Scoring (CSV)", 
    "Relational SQL Analytics",
    "Data Quality & Drift Monitor"
])


# ══════════════════════════════════════════════════════════
# TAB 1 — Individual Flight Risk Assessment
# ══════════════════════════════════════════════════════════
with tab_single:
    st.markdown("<p style='font-size:0.92rem;color:#94a3b8;margin-bottom:18px;'>Configure employee attributes across the 4 foundational workforce dimensions. All parameters feed directly into the real-time inference pipeline.</p>", unsafe_allow_html=True)

    # ── 2x2 Bento Cards Grid ──────────────────────────────
    bento_row1_c1, bento_row1_c2 = st.columns(2)

    with bento_row1_c1:
        st.markdown(f"""
        <div class="bento-header">
            <div class="bento-icon">{SVG_USER_BOX}</div>
            <div>
                <h3 class="bento-title">Demographics & Profile</h3>
                <p class="bento-desc">Biographical profile and educational foundation</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_a, c_b = st.columns(2)
        with c_a:
            age            = st.number_input("Age", min_value=18, max_value=60, value=32, key="age")
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], key="ms")
        with c_b:
            gender         = st.selectbox("Gender", ["Male", "Female"], key="gender")
            education      = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                               format_func=lambda x: {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x], key="edu")
        
        education_field = st.selectbox("Education Field",
                            ["Life Sciences","Medical","Marketing","Technical Degree","Human Resources","Other"], key="ef")

    with bento_row1_c2:
        st.markdown(f"""
        <div class="bento-header">
            <div class="bento-icon">{SVG_BRIEF_BOX}</div>
            <div>
                <h3 class="bento-title">Role, Compensation & Equity</h3>
                <p class="bento-desc">Departmental placement, remuneration and stock incentives</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_c, c_d = st.columns(2)
        with c_c:
            department     = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"], key="dept")
            job_level      = st.selectbox("Job Level", [1, 2, 3, 4, 5], key="jl")
            percent_hike   = st.number_input("% Salary Hike", min_value=11, max_value=25, value=15, key="ph")
        with c_d:
            job_role       = st.selectbox("Job Role", ["Sales Executive","Research Scientist","Laboratory Technician",
                                                      "Manufacturing Director","Healthcare Representative","Manager",
                                                      "Sales Representative","Research Director","Human Resources"], key="jr")
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100, key="mi")
            stock_option   = st.selectbox("Stock Option Level", [0, 1, 2, 3], key="sol")
        
        with st.expander("Advanced Compensation Calibration (Daily / Hourly / Monthly Rates)"):
            c_r1, c_r2, c_r3 = st.columns(3)
            with c_r1: daily_rate   = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800, step=10, key="dr")
            with c_r2: hourly_rate  = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65, key="hr")
            with c_r3: monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=27000, value=14000, step=500, key="mr")

    bento_row2_c1, bento_row2_c2 = st.columns(2)

    with bento_row2_c1:
        st.markdown(f"""
        <div class="bento-header">
            <div class="bento-icon">{SVG_CLOCK_BOX}</div>
            <div>
                <h3 class="bento-title">Workload, Mobility & Tenure</h3>
                <p class="bento-desc">Commute distance, travel frequency and career velocity</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_e, c_f = st.columns(2)
        with c_e:
            overtime               = st.selectbox("OverTime", ["Yes", "No"], key="ot")
            business_travel        = st.selectbox("Business Travel", ["Travel_Rarely","Travel_Frequently","Non-Travel"], key="bt")
            years_at_company       = st.number_input("Years at Company", min_value=0, max_value=40, value=4, key="yac")
            years_since_last_promotion = st.number_input("Years Since Promotion", min_value=0, max_value=15, value=1, key="yslp")
            num_companies_worked   = st.number_input("Companies Worked At", min_value=0, max_value=10, value=2, key="ncw")
        with c_f:
            distance_from_home     = st.number_input("Distance from Home (km)", min_value=1, max_value=30, value=8, key="dfh")
            total_working_years    = st.number_input("Total Working Years", min_value=0, max_value=40, value=8, key="twy")
            years_in_current_role  = st.number_input("Years in Current Role", min_value=0, max_value=20, value=2, key="yicr")
            years_with_curr_manager = st.number_input("Years with Manager", min_value=0, max_value=15, value=2, key="ywcm")
            training_times         = st.number_input("Trainings Last Year", min_value=0, max_value=6, value=2, key="ttly")

    with bento_row2_c2:
        st.markdown(f"""
        <div class="bento-header">
            <div class="bento-icon">{SVG_HEART_BOX}</div>
            <div>
                <h3 class="bento-title">Sentiment & Dynamics</h3>
                <p class="bento-desc">Satisfaction scores, relationship health and performance rating</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_g, c_h = st.columns(2)
        fmt = lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x]
        with c_g:
            job_satisfaction  = st.selectbox("Job Satisfaction", [1, 2, 3, 4], format_func=fmt, key="js")
            work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4],
                                  format_func=lambda x:{1:"1 - Bad",2:"2 - Good",3:"3 - Better",4:"4 - Best"}[x], key="wlb")
            relationship_sat  = st.selectbox("Relationship Sat.", [1, 2, 3, 4], format_func=fmt, key="rs")
        with c_h:
            env_satisfaction  = st.selectbox("Environment Sat.", [1, 2, 3, 4], format_func=fmt, key="es")
            job_involvement   = st.selectbox("Job Involvement", [1, 2, 3, 4], format_func=fmt, key="ji")
            perf_rating       = st.selectbox("Performance Rating", [1, 2, 3, 4],
                                  format_func=lambda x:{1:"1 - Low",2:"2 - Good",3:"3 - Excellent",4:"4 - Outstanding"}[x], key="pr")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Run Flight Risk Analysis", type="primary", use_container_width=True, key="predict")

    # ── Inference Execution & Results ─────────────────────────
    if predict_btn and model_loaded:
        input_data = {
            "Age": age,
            "BusinessTravel": business_travel,
            "DailyRate": daily_rate,
            "Department": department,
            "DistanceFromHome": distance_from_home,
            "Education": education,
            "EducationField": education_field,
            "EnvironmentSatisfaction": env_satisfaction,
            "Gender": gender,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies_worked,
            "OverTime": overtime,
            "PercentSalaryHike": percent_hike,
            "PerformanceRating": perf_rating,
            "RelationshipSatisfaction": relationship_sat,
            "StockOptionLevel": stock_option,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "YearsWithCurrManager": years_with_curr_manager,
        }

        try:
            input_df        = pd.DataFrame([input_data])
            input_df        = engineer_features(input_df)
            input_processed = preprocessor.transform(input_df)

            probability = float(model.predict_proba(input_processed)[0][1])
            prediction  = int(model.predict(input_processed)[0])
            risk_label, status_class, advice, status_color = classify_risk(probability)
            retention_prob = (1.0 - probability) * 100.0

            # ── Deluxe Result Cockpit ──────────────────────────
            verdict_text = "Flight Risk Flagged" if prediction == 1 else "Retention Anchor"
            st.markdown(f"""
            <div class="cockpit-card-wrap">
                <div class="cockpit-top-bar"></div>
                <div class="cockpit-header">
                    <div class="cockpit-title-group">
                        <h2>Predictive Assessment Console</h2>
                        <p>Calibrated risk scoring derived from machine learning ensemble</p>
                    </div>
                    <div class="status-badge {status_class}">
                        <span>●</span> {risk_label}
                    </div>
                </div>
                <div class="cockpit-metrics">
                    <div class="cockpit-stat-box">
                        <div class="cockpit-stat-lbl">Departure Probability</div>
                        <div class="cockpit-stat-val" style="color:{status_color};">{probability*100:.1f}%</div>
                    </div>
                    <div class="cockpit-stat-box">
                        <div class="cockpit-stat-lbl">Retention Confidence</div>
                        <div class="cockpit-stat-val" style="color:#10b981;">{retention_prob:.1f}%</div>
                    </div>
                    <div class="cockpit-stat-box">
                        <div class="cockpit-stat-lbl">Classifier Verdict</div>
                        <div class="cockpit-stat-val" style="color:#ffffff;font-size:1.5rem;padding-top:6px;">
                            {verdict_text}
                        </div>
                    </div>
                </div>
                <div class="glow-track">
                    <div class="glow-fill" style="width:{probability*100:.1f}%;background:linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #f43f5e 100%);"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.74rem;color:#94a3b8;font-weight:600;">
                    <span>0% Stable Baseline</span>
                    <span>40% Threshold</span>
                    <span>65% Critical Flight</span>
                    <span>100% Imminent Exit</span>
                </div>
                <div class="cockpit-advice">
                    <strong>Retention Guidance:</strong> {advice}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Financial Capital Flight Risk & ROI Card ───────────────────
            exposure_data = calculate_flight_risk_exposure(probability, monthly_income)
            roi_data = calculate_retention_roi(probability, monthly_income)

            st.markdown(f"""
            <div style="background:rgba(18,24,38,0.75);border:1px solid rgba(99,102,241,0.22);border-radius:16px;padding:22px;margin:20px 0;backdrop-filter:blur(14px);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
                    <div>
                        <span style="font-size:0.72rem;font-weight:700;color:#818cf8;letter-spacing:0.06em;text-transform:uppercase;">FINANCIAL RISK MODELING · SHRM BENCHMARK</span>
                        <h3 style="margin:4px 0 0 0;font-size:1.25rem;color:#ffffff;">Capital Flight Exposure & Retention ROI</h3>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:0.76rem;color:#94a3b8;">Estimated Replacement Cost</span>
                        <div style="font-size:1.3rem;font-weight:800;color:#f43f5e;">${exposure_data.get('replacement_cost', 0.0):,.0f}</div>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:14px;margin-bottom:14px;">
                    <div style="background:rgba(10,14,24,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:14px;">
                        <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Expected Loss Exposure</div>
                        <div style="font-size:1.35rem;font-weight:800;color:#fbbf24;margin-top:4px;">${exposure_data.get('expected_loss', 0.0):,.0f}</div>
                        <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">P(exit) × 1.5× Annual Salary</div>
                    </div>
                    <div style="background:rgba(10,14,24,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:14px;">
                        <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Net Intervention Value</div>
                        <div style="font-size:1.35rem;font-weight:800;color:#10b981;margin-top:4px;">+${roi_data.get('net_cost_avoidance', roi_data.get('net_savings', 0.0)):,.0f}</div>
                        <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">Savings after $2.5k retention budget</div>
                    </div>
                    <div style="background:rgba(10,14,24,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:14px;">
                        <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Retention Program ROI</div>
                        <div style="font-size:1.35rem;font-weight:800;color:#818cf8;margin-top:4px;">{roi_data.get('roi_multiple', 1.0)}x</div>
                        <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">Break-even flight prob: {roi_data.get('break_even_probability', 0.05)*100:.1f}%</div>
                    </div>
                </div>
                <div style="font-size:0.80rem;color:#cbd5e1;background:rgba(99,102,241,0.08);border-left:3px solid #818cf8;padding:8px 12px;border-radius:4px;">
                    <strong>Executive Financial Insight:</strong> With an exit probability of <strong>{probability*100:.1f}%</strong> and an estimated turnover replacement cost of <strong>${exposure_data.get('replacement_cost', 0.0):,.0f}</strong>, targeted retention yields an expected net benefit of <strong>${roi_data.get('net_cost_avoidance', roi_data.get('net_savings', 0.0)):,.0f}</strong>, delivering quantifiable cost avoidance.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Multi-Factor Explanation Suite ────────────────────────
            st.markdown("<h3 style='margin:28px 0 6px 0;font-size:1.35rem;'>Multi-Factor Diagnostic Suite</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color:#94a3b8;font-size:0.86rem;margin-bottom:18px;'>Turnover is fundamentally caused by compounding organizational tensions. Explore the exact decision pathways and synergistic risk factors below.</p>", unsafe_allow_html=True)

            exp_tab_tree, exp_tab_waterfall, exp_tab_pillars = st.tabs([
                "Primary Reasons (Decision Pathway)", 
                "Factor Attribution Walk (SHAP)", 
                "Four-Pillar Risk Matrix"
            ])

            try:
                # ── 1. Decision Pathway & Primary Reasons ─────────────────
                with exp_tab_tree:
                    st.markdown("<h4 style='font-size:1.15rem;margin-bottom:4px;color:#ffffff;'>Primary Reasons Driving Retention vs. Departure</h4>", unsafe_allow_html=True)
                    st.caption("How this employee's specific working conditions, compensation, and career velocity compound to determine their flight risk:")

                    tree = model.estimators_[0]
                    node_indicator = tree.decision_path(input_processed)
                    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

                    # Compute baseline and step-by-step impact
                    root_samples = int(tree.tree_.n_node_samples[node_index[0]])
                    root_val = tree.tree_.value[node_index[0]][0]
                    base_prob = float(root_val[1] / np.sum(root_val))

                    pathway_steps = []
                    prev_prob = base_prob

                    for i, node_id in enumerate(node_index[:-1]):
                        next_node_id = node_index[i + 1]
                        feat_idx = tree.tree_.feature[node_id]
                        feat_name = feature_names[feat_idx]
                        thresh = float(tree.tree_.threshold[node_id])
                        val = float(input_processed[0, feat_idx])
                        went_left = val <= thresh
                        
                        next_val = tree.tree_.value[next_node_id][0]
                        curr_prob = float(next_val[1] / np.sum(next_val))
                        delta = curr_prob - prev_prob
                        
                        details = get_factor_details(feat_name, went_left, val, delta)
                        pathway_steps.append({
                            "step_num": i + 1,
                            "feat_name": feat_name,
                            "title": details["title"],
                            "condition": details["condition"],
                            "reason": details["reason"],
                            "delta": delta,
                            "cum_prob": curr_prob,
                            "is_risk": delta > 0.005,
                            "is_protect": delta < -0.005,
                        })
                        prev_prob = curr_prob

                    leaf_id = node_index[-1]
                    leaf_samples = int(tree.tree_.n_node_samples[leaf_id])
                    leaf_val = tree.tree_.value[leaf_id][0]
                    leaf_prob = float(leaf_val[1] / np.sum(leaf_val))

                    # Identify Primary Reasons
                    risk_drivers = [s for s in pathway_steps if s["is_risk"]]
                    risk_drivers.sort(key=lambda s: s["delta"], reverse=True)

                    protective_anchors = [s for s in pathway_steps if s["is_protect"]]
                    protective_anchors.sort(key=lambda s: s["delta"])

                    # 1. Executive Summary Box: Primary Reasons
                    if leaf_prob >= 0.40 or len(risk_drivers) > 0:
                        banner_class = "reasons-banner reasons-banner-risk"
                        icon = SVG_ALERT_CIRCLE
                        header_title = "Key Reasons Identified for Potential Departure"
                        header_subtitle = "Based on this employee's unique working environment and career situation, the model identified the following primary factors that make this employee likely to leave:"
                        
                        cards_html = ""
                        for driver in risk_drivers[:3]:
                            pct_push = driver['delta'] * 100
                            cards_html += clean_html(f"""
                            <div class="reason-item-card">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                                    <span style="font-weight:700;color:#ffffff;font-size:0.95rem;">{driver['title']}</span>
                                    <span style="background:rgba(244,63,94,0.18);color:#fb7185;padding:3px 10px;border-radius:6px;font-size:0.76rem;font-weight:700;">+{pct_push:.1f}% Risk Push</span>
                                </div>
                                <div style="color:#a5b4fc;font-size:0.84rem;font-weight:600;margin-bottom:4px;">
                                    Observed Factor: <span style="color:#ffffff;">{driver['condition']}</span>
                                </div>
                                <div style="color:#94a3b8;font-size:0.82rem;line-height:1.45;">
                                    <strong style="color:#cbd5e1;">Why this causes departure:</strong> {driver['reason']}
                                </div>
                            </div>
                            """)
                        if not cards_html:
                            cards_html = "<p style='color:#94a3b8;font-size:0.84rem;'>No single catastrophic factor dominates; cumulative mild pressures are present.</p>"
                    else:
                        banner_class = "reasons-banner reasons-banner-protect"
                        icon = SVG_SHIELD_CHECK
                        header_title = "Key Reasons for Strong Retention Stability"
                        header_subtitle = "The analysis indicates this employee is well-anchored with low voluntary turnover risk. The following protective factors explain why they are likely to stay:"
                        
                        cards_html = ""
                        for anchor in protective_anchors[:3]:
                            pct_drop = abs(anchor['delta'] * 100)
                            cards_html += clean_html(f"""
                            <div class="reason-item-card">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                                    <span style="font-weight:700;color:#ffffff;font-size:0.95rem;">{anchor['title']}</span>
                                    <span style="background:rgba(16,185,129,0.18);color:#34d399;padding:3px 10px;border-radius:6px;font-size:0.76rem;font-weight:700;">-{pct_drop:.1f}% Retention Anchor</span>
                                </div>
                                <div style="color:#6ee7b7;font-size:0.84rem;font-weight:600;margin-bottom:4px;">
                                    Observed Factor: <span style="color:#ffffff;">{anchor['condition']}</span>
                                </div>
                                <div style="color:#94a3b8;font-size:0.82rem;line-height:1.45;">
                                    <strong style="color:#cbd5e1;">Why this keeps employees:</strong> {anchor['reason']}
                                </div>
                            </div>
                            """)
                        if not cards_html:
                            cards_html = "<p style='color:#94a3b8;font-size:0.84rem;'>Employee exhibits balanced baseline conditions across all retention pillars.</p>"

                    st.markdown(clean_html(f"""
                    <div class="{banner_class}">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                            {icon}
                            <span style="font-weight:800;font-size:1.06rem;color:#ffffff;font-family:'Outfit',sans-serif;">{header_title}</span>
                        </div>
                        <p style="color:#cbd5e1;font-size:0.84rem;margin-bottom:12px;line-height:1.5;">{header_subtitle}</p>
                        {cards_html}
                    </div>
                    """), unsafe_allow_html=True)

                    # 2. Step-by-Step Risk Decision Pathway
                    st.markdown("<h4 style='font-size:1.05rem;margin:24px 0 4px 0;color:#ffffff;'>Step-by-Step Risk Progression Pathway</h4>", unsafe_allow_html=True)
                    st.caption("Walk through each factor in sequence to see how individual conditions shifted the employee's departure risk from the company baseline:")

                    # Baseline Root Step
                    st.markdown(clean_html(f"""
                    <div class="tree-card tree-card-split">
                        <div>
                            <span style="background:rgba(99,102,241,0.15);color:#a5b4fc;padding:2px 8px;border-radius:4px;font-size:0.70rem;font-weight:700;letter-spacing:0.04em;">STEP 0 · BASELINE</span>
                            <div style="color:#ffffff;font-weight:700;font-size:0.96rem;margin-top:4px;">Company Workforce Baseline</div>
                            <span style="color:#94a3b8;font-size:0.80rem;">Starting baseline turnover rate across historical workforce ({root_samples:,} employees)</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="color:#818cf8;font-weight:800;font-size:1.2rem;">{base_prob*100:.1f}%</span><br>
                            <span style="color:#94a3b8;font-size:0.74rem;">Base Exit Rate</span>
                        </div>
                    </div>
                    """), unsafe_allow_html=True)

                    # Sequential steps
                    for step in pathway_steps:
                        st.markdown(f'<div class="tree-arrow">{SVG_CHEVRON_DOWN}</div>', unsafe_allow_html=True)
                        
                        if step["is_risk"]:
                            card_border_class = "tree-card-risk"
                            delta_badge = f'<span style="background:rgba(244,63,94,0.18);color:#fb7185;padding:2px 8px;border-radius:5px;font-size:0.74rem;font-weight:700;">+{step["delta"]*100:.1f}% Risk Push</span>'
                        elif step["is_protect"]:
                            card_border_class = "tree-card-protect"
                            delta_badge = f'<span style="background:rgba(16,185,129,0.18);color:#34d399;padding:2px 8px;border-radius:5px;font-size:0.74rem;font-weight:700;">{step["delta"]*100:.1f}% Risk Drop</span>'
                        else:
                            card_border_class = "tree-card-split"
                            delta_badge = '<span style="background:rgba(148,163,184,0.15);color:#cbd5e1;padding:2px 8px;border-radius:5px;font-size:0.74rem;font-weight:600;">Neutral Impact</span>'

                        prob_color = "#f43f5e" if step["cum_prob"] >= 0.50 else ("#f59e0b" if step["cum_prob"] >= 0.25 else "#10b981")

                        st.markdown(clean_html(f"""
                        <div class="tree-card {card_border_class}">
                            <div style="max-width:72%;">
                                <div style="display:flex;align-items:center;gap:8px;">
                                    <span style="background:rgba(99,102,241,0.15);color:#a5b4fc;padding:2px 8px;border-radius:4px;font-size:0.70rem;font-weight:700;letter-spacing:0.04em;">STEP {step['step_num']} · {step['title'].upper()}</span>
                                    {delta_badge}
                                </div>
                                <div style="color:#ffffff;font-weight:700;font-size:0.95rem;margin-top:5px;">{step['condition']}</div>
                                <div style="color:#94a3b8;font-size:0.79rem;margin-top:2px;line-height:1.4;">{step['reason']}</div>
                            </div>
                            <div style="text-align:right;">
                                <span style="color:{prob_color};font-weight:800;font-size:1.22rem;">{step['cum_prob']*100:.1f}%</span><br>
                                <span style="color:#94a3b8;font-size:0.74rem;">Resulting Exit Risk</span>
                            </div>
                        </div>
                        """), unsafe_allow_html=True)

                    # Terminal Verdict
                    leaf_pred = "PREDICTED TO LEAVE (HIGH FLIGHT RISK)" if leaf_prob >= 0.50 else "PREDICTED TO STAY (RETENTION SECURED)"
                    leaf_class = "tree-card-leaf-leave" if leaf_prob >= 0.50 else "tree-card-leaf-stay"
                    leaf_color = "#f43f5e" if leaf_prob >= 0.50 else "#10b981"

                    st.markdown(f'<div class="tree-arrow">{SVG_CHEVRON_DOWN}</div>', unsafe_allow_html=True)
                    st.markdown(clean_html(f"""
                    <div class="tree-card {leaf_class}">
                        <div>
                            <span style="background:rgba(255,255,255,0.12);color:#ffffff;padding:2px 8px;border-radius:4px;font-size:0.70rem;font-weight:700;letter-spacing:0.04em;">FINAL DECISION VERDICT</span>
                            <div style="color:#ffffff;font-weight:800;font-size:1.02rem;margin-top:4px;">Outcome: {leaf_pred}</div>
                            <span style="color:#94a3b8;font-size:0.80rem;">Cumulative profile match across {leaf_samples:,} identical historical employee cases</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="color:{leaf_color};font-weight:900;font-size:1.32rem;">{leaf_prob*100:.1f}%</span><br>
                            <span style="color:#94a3b8;font-size:0.74rem;">Terminal Departure Likelihood</span>
                        </div>
                    </div>
                    """), unsafe_allow_html=True)

                    if leaf_prob >= 0.50:
                        synergy_text = f"This employee's departure is not the result of a single isolated issue. As shown above, specific friction points (such as heavy overtime, compensation discrepancy, or lack of promotion) compounded at each step, escalating flight risk from the 16.1% company baseline up to <strong>{leaf_prob*100:.1f}%</strong>. Targeted retention interventions on any of the primary drivers listed above can disrupt this cycle."
                    else:
                        synergy_text = f"This employee's retention is reinforced by multiple protective anchors working together—such as healthy work-life balance, fair compensation, and role engagement—keeping exit risk safely suppressed at <strong>{leaf_prob*100:.1f}%</strong>."

                    st.markdown(clean_html(f"""
                    <div class="cockpit-advice" style="margin-top:16px;">
                        <strong>Business Takeaway:</strong> {synergy_text}
                    </div>
                    """), unsafe_allow_html=True)

                # ── 2. Compounding Waterfall Walk ─────────────────────────
                with exp_tab_waterfall:
                    st.markdown("<h4 style='font-size:1.1rem;margin-bottom:4px;'>Factor Attribution Walk (SHAP)</h4>", unsafe_allow_html=True)
                    st.caption("How each employee factor shifts probability from the company baseline (16.1%) to the final inference score:")

                    shap_values = explainer(input_processed)
                    if hasattr(shap_values, "values"):
                        vals = shap_values.values
                        if vals.ndim == 3:
                            vals = vals[0, :, 1]
                        elif vals.ndim == 2:
                            vals = vals[0]
                        else:
                            vals = vals.flatten()
                    elif isinstance(shap_values, list):
                        vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                    else:
                        vals = np.array(shap_values)[0]

                    shap_raw = pd.Series(vals, index=feature_names).sort_values(key=abs, ascending=False)
                    
                    # Deduplicate one-hot dummy pairs
                    seen_p = set()
                    clean_dict = {}
                    for f_name, f_val in shap_raw.items():
                        p = f_name.split("_")[0] if "_" in f_name else f_name
                        if p not in seen_p:
                            seen_p.add(p)
                            clean_label = f_name.replace("_", ": ") if "_" in f_name else f_name
                            clean_dict[clean_label] = f_val
                    clean_series = pd.Series(clean_dict).head(7)

                    # Build waterfall steps
                    base_rate = 0.1612
                    step_labels = ["Baseline (16.1%)"]
                    step_deltas = [base_rate]

                    for feat, delta in clean_series.items():
                        step_labels.append(feat)
                        step_deltas.append(delta)

                    bottoms = [0.0]
                    curr_sum = base_rate
                    for d in step_deltas[1:]:
                        if d >= 0:
                            bottoms.append(curr_sum)
                            curr_sum += d
                        else:
                            curr_sum += d
                            bottoms.append(curr_sum)

                    step_labels.append("Final Model Score")
                    bottoms.append(0.0)
                    heights = [base_rate] + [abs(d) for d in step_deltas[1:]] + [probability]
                    bar_colors = ["#6366f1"] + ["#f43f5e" if d > 0 else "#10b981" for d in step_deltas[1:]] + ["#f43f5e" if probability >= 0.4 else "#10b981"]

                    fig, ax = plt.subplots(figsize=(10, 4.6))
                    fig.patch.set_facecolor("#0b0f19")
                    ax.set_facecolor("#0f1422")

                    bars = ax.bar(range(len(step_labels)), heights, bottom=bottoms, color=bar_colors, width=0.52, edgecolor="#334155", linewidth=1)

                    # Draw connecting connector lines
                    for b_idx in range(len(step_labels) - 2):
                        y_pos = bottoms[b_idx+1] + (heights[b_idx+1] if step_deltas[b_idx+1] < 0 else 0)
                        ax.hlines(y=y_pos, xmin=b_idx, xmax=b_idx+1, colors="#475569", linestyles="--", linewidth=0.9)

                    ax.tick_params(colors="#94a3b8", labelsize=8.5)
                    ax.set_xticks(range(len(step_labels)))
                    ax.set_xticklabels(step_labels, rotation=20, ha="right", color="#f1f5f9", fontsize=8.5, fontweight="500")
                    ax.set_ylabel("Attrition Probability", color="#94a3b8", fontsize=9.0, fontweight="600")
                    ax.grid(axis="y", color="#1e293b", linestyle=":")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#334155")
                    ax.set_title("Factor Attribution Staircase: Baseline to Predicted Score", color="#ffffff", fontsize=10.5, fontweight="700", pad=12)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    pos_features = clean_series[clean_series > 0].head(3).index.tolist()
                    neg_features = clean_series[clean_series < 0].head(3).index.tolist()
                    if pos_features:
                        st.markdown(f"**Leading Risk Contributors:** `{', '.join(pos_features)}`")
                    if neg_features:
                        st.markdown(f"**Protective Retention Factors:** `{', '.join(neg_features)}`")

                # ── 3. Four-Pillar Risk Matrix ────────────────────────────
                with exp_tab_pillars:
                    st.markdown("<h4 style='font-size:1.1rem;margin-bottom:4px;'>Four-Pillar Strategic Retention Matrix</h4>", unsafe_allow_html=True)
                    st.caption("Categorical evaluation across the four foundational retention dimensions:")

                    p1_flags = (1 if monthly_income < 4000 else 0) + (1 if stock_option == 0 else 0) + (1 if percent_hike <= 12 else 0)
                    p2_flags = (1 if overtime == "Yes" else 0) + (1 if work_life_balance <= 2 else 0) + (1 if distance_from_home >= 15 else 0)
                    p3_flags = (1 if years_since_last_promotion >= 3 else 0) + (1 if years_in_current_role >= 4 else 0) + (1 if num_companies_worked >= 4 else 0)
                    p4_flags = (1 if job_satisfaction <= 2 else 0) + (1 if env_satisfaction <= 2 else 0) + (1 if years_with_curr_manager <= 1 else 0)

                    def get_pillar_badge(flags):
                        if flags >= 2:
                            return "HIGH RISK", "#f43f5e"
                        elif flags == 1:
                            return "MODERATE", "#f59e0b"
                        return "HEALTHY", "#10b981"

                    b1_lbl, b1_col = get_pillar_badge(p1_flags)
                    b2_lbl, b2_col = get_pillar_badge(p2_flags)
                    b3_lbl, b3_col = get_pillar_badge(p3_flags)
                    b4_lbl, b4_col = get_pillar_badge(p4_flags)

                    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

                    with col_p1:
                        st.markdown(f"""
                        <div class="pillar-card">
                            <div class="pillar-icon-wrap">{SVG_BRIEF_BOX}</div>
                            <div class="pillar-title">Compensation & Equity</div>
                            <p style="color:{b1_col};font-weight:800;font-size:0.95rem;margin:4px 0;">{b1_lbl}</p>
                            <span style="color:#94a3b8;font-size:0.74rem;">Income, Stock, % Hike</span>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_p2:
                        st.markdown(f"""
                        <div class="pillar-card">
                            <div class="pillar-icon-wrap">{SVG_CLOCK_BOX}</div>
                            <div class="pillar-title">Workload & Burnout</div>
                            <p style="color:{b2_col};font-weight:800;font-size:0.95rem;margin:4px 0;">{b2_lbl}</p>
                            <span style="color:#94a3b8;font-size:0.74rem;">Overtime, WLB, Commute</span>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_p3:
                        st.markdown(f"""
                        <div class="pillar-card">
                            <div class="pillar-icon-wrap">{SVG_CHART}</div>
                            <div class="pillar-title">Career Velocity</div>
                            <p style="color:{b3_col};font-weight:800;font-size:0.95rem;margin:4px 0;">{b3_lbl}</p>
                            <span style="color:#94a3b8;font-size:0.74rem;">Promotion Gap, Role Tenure</span>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_p4:
                        st.markdown(f"""
                        <div class="pillar-card">
                            <div class="pillar-icon-wrap">{SVG_HEART_BOX}</div>
                            <div class="pillar-title">Workplace Dynamics</div>
                            <p style="color:{b4_col};font-weight:800;font-size:0.95rem;margin:4px 0;">{b4_lbl}</p>
                            <span style="color:#94a3b8;font-size:0.74rem;">Job Sat, Manager, Env</span>
                        </div>
                        """, unsafe_allow_html=True)

                    high_pillar_count = (1 if p1_flags >= 2 else 0) + (1 if p2_flags >= 2 else 0) + (1 if p3_flags >= 2 else 0) + (1 if p4_flags >= 2 else 0)
                    if high_pillar_count >= 2:
                        st.markdown("""
                        <div class="cockpit-advice" style="border-left:3px solid #f43f5e;margin-top:16px;">
                            <strong>Compounding Risk Alert:</strong> This employee exhibits <strong>2 or more operational pillars in the High Risk tier</strong>. 
                            In empirical workforce studies, when <em>Workload Burnout</em> coincides with <em>Compensation Dissatisfaction</em>, retention probability plummets by over 60%.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="cockpit-advice" style="margin-top:16px;">
                            <strong>Pillar Health Assessment:</strong> Risk signals are isolated to a single dimension. Targeted retention interventions in that specific domain will effectively stabilize retention.
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as exp_err:
                st.warning(f"Could not render multi-factor explanation: {exp_err}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    elif predict_btn and not model_loaded:
        st.warning("Model not loaded. Please verify model files.")


# ══════════════════════════════════════════════════════════
# TAB 2 — Batch Cohort Scoring (CSV)
# ══════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("""
    <div style="margin-bottom:16px;">
        <h3 style="font-size:1.35rem;margin:0 0 4px 0;">Batch Cohort Scoring</h3>
        <p style="color:#94a3b8;font-size:0.88rem;">Upload enterprise staff rosters (CSV) for instant mass attrition risk scoring and exportable analytics.</p>
    </div>
    """, unsafe_allow_html=True)

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        uploaded_file = st.file_uploader("Upload Employee CSV Dataset", type=["csv"], key="batch_upload")
    with col_b2:
        sample_cols = [
            "Age","BusinessTravel","DailyRate","Department","DistanceFromHome",
            "Education","EducationField","EnvironmentSatisfaction","Gender",
            "HourlyRate","JobInvolvement","JobLevel","JobRole","JobSatisfaction",
            "MaritalStatus","MonthlyIncome","MonthlyRate","NumCompaniesWorked",
            "OverTime","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction",
            "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear",
            "WorkLifeBalance","YearsAtCompany","YearsInCurrentRole",
            "YearsSinceLastPromotion","YearsWithCurrManager"
        ]
        template_df = pd.DataFrame(columns=sample_cols)
        template_csv = template_df.to_csv(index=False).encode("utf-8")
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        st.download_button("Download Template CSV", data=template_csv,
                           file_name="employee_template.csv", mime="text/csv", use_container_width=True)

    if uploaded_file and model_loaded:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(df_input):,} employee records.** Sample Preview:")
            st.dataframe(df_input.head(5), use_container_width=True)

            if st.button("Run Batch Scoring", type="primary", key="batch_predict"):
                with st.spinner("Scoring workforce cohort …"):
                    df_feat = engineer_features(df_input.copy())
                    X_proc  = preprocessor.transform(df_feat)
                    probs   = model.predict_proba(X_proc)[:, 1]
                    preds   = model.predict(X_proc)

                    df_input["Attrition_Probability"] = probs.round(4)
                    df_input["Attrition_Prediction"]  = ["Will Leave" if p == 1 else "Will Stay" for p in preds]

                    def get_risk_label(p):
                        if p >= 0.65: return "High Risk"
                        if p >= 0.40: return "Medium Risk"
                        return "Low Risk"

                    df_input["Risk_Level"] = [get_risk_label(p) for p in probs]

                high   = (df_input["Risk_Level"] == "High Risk").sum()
                medium = (df_input["Risk_Level"] == "Medium Risk").sum()
                low    = (df_input["Risk_Level"] == "Low Risk").sum()

                st.markdown("<h4 style='margin-top:20px;'>Cohort Distribution Summary</h4>", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Cohort", f"{len(df_input):,}")
                m2.metric("Critical / High Risk", f"{high:,}")
                m3.metric("Moderate Risk", f"{medium:,}")
                m4.metric("Low Risk (Stable)", f"{low:,}")

                # Cohort Financial Risk & Capital Flight Exposure
                cohort_fin = compute_cohort_financial_impact(df_input)
                st.markdown(f"""
                <div style="background:rgba(18,24,38,0.75);border:1px solid rgba(99,102,241,0.22);border-radius:14px;padding:18px;margin:16px 0;">
                    <span style="font-size:0.72rem;font-weight:700;color:#818cf8;letter-spacing:0.06em;text-transform:uppercase;">COHORT CAPITAL AT RISK · SHRM BENCHMARK (1.5× SALARY)</span>
                    <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:14px;margin-top:12px;">
                        <div style="background:rgba(10,14,24,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px;">
                            <span style="color:#94a3b8;font-size:0.74rem;text-transform:uppercase;font-weight:600;">Expected Loss Exposure</span>
                            <div style="color:#fbbf24;font-size:1.35rem;font-weight:800;margin-top:2px;">${cohort_fin.get('total_loss_exposure', cohort_fin.get('total_financial_exposure', 0.0)):,.0f}</div>
                            <span style="color:#94a3b8;font-size:0.72rem;">Expected flight cost</span>
                        </div>
                        <div style="background:rgba(10,14,24,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px;">
                            <span style="color:#94a3b8;font-size:0.74rem;text-transform:uppercase;font-weight:600;">At-Risk Replacement Cost</span>
                            <div style="color:#f43f5e;font-size:1.35rem;font-weight:800;margin-top:2px;">${cohort_fin.get('total_replacement_cost_at_risk', 0.0):,.0f}</div>
                            <span style="color:#94a3b8;font-size:0.72rem;">High risk staff replacement</span>
                        </div>
                        <div style="background:rgba(10,14,24,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px;">
                            <span style="color:#94a3b8;font-size:0.74rem;text-transform:uppercase;font-weight:600;">Estimated Retention ROI</span>
                            <div style="color:#10b981;font-size:1.35rem;font-weight:800;margin-top:2px;">{cohort_fin.get('estimated_cohort_roi_multiple', 1.0)}x</div>
                            <span style="color:#94a3b8;font-size:0.72rem;">Return on targeted retention</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Scored Roster Preview")
                st.dataframe(
                    df_input[["Attrition_Probability", "Attrition_Prediction", "Risk_Level"] +
                              [c for c in df_input.columns if c not in ["Attrition_Probability","Attrition_Prediction","Risk_Level"]]].head(25),
                    use_container_width=True
                )

                output_csv = df_input.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Export Scored Roster (CSV)",
                    data=output_csv,
                    file_name="attrition_predictions_scored.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

    elif uploaded_file and not model_loaded:
        st.warning("Model not loaded. Please verify model files.")


# ══════════════════════════════════════════════════════════
# TAB 3 — Relational SQL Analytics
# ══════════════════════════════════════════════════════════
with tab_sql:
    st.markdown("""
    <div style="margin-bottom:16px;">
        <h3 style="font-size:1.35rem;margin:0 0 4px 0;">Relational SQL Workforce Analytics</h3>
        <p style="color:#94a3b8;font-size:0.88rem;">
            Investigate deep relational dynamics across the <strong>IBM HR database (1,470 records)</strong>. 
            Run pre-compiled business intelligence queries or execute bespoke SQLite commands in the interactive sandbox.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sql_conn, sql_rows, raw_df = get_sql_database()

    if sql_conn is None:
        st.error("Could not initialize SQL database. Ensure data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv exists.")
    else:
        # Top KPI Ribbon
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Database Headcount", f"{sql_rows:,} records")
        k2.metric("Overall Attrition", "16.12%", help="237 total leavers across historical cohorts")
        k3.metric("Highest Turnover Dept", "Sales (20.63%)", help="Sales Representatives exhibit 38.2% exit rate")
        k4.metric("Burnout Catalyst", "Overtime (30.53%)", help="3x higher attrition compared to non-overtime staff")

        st.markdown("<div style='height:1px;background:rgba(255,255,255,0.06);margin:20px 0;'></div>", unsafe_allow_html=True)

        sql_queries = {
            "Q1: Executive KPI Summary": {
                "desc": "High-level summary of headcount, turnover rate, average income, and tenure for executive briefings.",
                "sql": """SELECT 
    COUNT(*) AS total_headcount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS total_leavers,
    SUM(CASE WHEN Attrition = 'No' THEN 1 ELSE 0 END) AS active_headcount,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0) AS company_avg_monthly_income,
    ROUND(AVG(Age), 1) AS company_avg_age,
    ROUND(AVG(TotalWorkingYears), 1) AS company_avg_tenure_years
FROM employees;"""
            },
            "Q2: Department Headcount & Turnover Rate": {
                "desc": "Identifies which organizational units have disproportionate turnover rates and compensation levels.",
                "sql": """SELECT 
    Department,
    COUNT(*) AS department_headcount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income,
    ROUND(AVG(PercentSalaryHike), 1) AS avg_salary_hike_pct
FROM employees
GROUP BY Department
ORDER BY attrition_rate_pct DESC;"""
            },
            "Q3: Job Role Pay Disparity & Attrition": {
                "desc": "Examines whether leavers were underpaid compared to stayers within the same job role.",
                "sql": """SELECT 
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
ORDER BY attrition_rate_pct DESC;"""
            },
            "Q4: Overtime & Work-Life Balance Burnout Matrix": {
                "desc": "Cross-tabulates overtime requirements against employee work-life balance satisfaction ratings.",
                "sql": """SELECT 
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
ORDER BY OverTime DESC, WorkLifeBalance ASC;"""
            },
            "Q5: Business Travel & Commute Distance Impact": {
                "desc": "Analyzes the combined impact of frequent travel and commute distance on departure rates.",
                "sql": """SELECT 
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
ORDER BY BusinessTravel, commute_distance_bucket;"""
            },
            "Q6: Window Function - Salary Equity & Role Ranking": {
                "desc": "Uses AVG() OVER (PARTITION BY JobRole) and DENSE_RANK() to identify staff significantly underpaid relative to peers.",
                "sql": """WITH RoleSalaryRanked AS (
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
LIMIT 15;"""
            },
            "Q7: Promotion Stagnation vs Company Tenure": {
                "desc": "Computes ratio of years without promotion relative to total company tenure.",
                "sql": """SELECT 
    Department,
    JobRole,
    ROUND(AVG(YearsAtCompany), 1) AS avg_tenure_years,
    ROUND(AVG(YearsSinceLastPromotion), 1) AS avg_years_since_promotion,
    ROUND(AVG(CAST(YearsSinceLastPromotion AS FLOAT) / NULLIF(YearsAtCompany, 0)), 2) AS promotion_stagnation_ratio,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
FROM employees
WHERE YearsAtCompany > 0
GROUP BY Department, JobRole
ORDER BY promotion_stagnation_ratio DESC;"""
            },
            "Q8: CTE - Active Employees Early Warning Cohort": {
                "desc": "Filters currently active employees exhibiting >= 3 simultaneous burnout and turnover signals.",
                "sql": """WITH RiskSignals AS (
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
LIMIT 15;"""
            },
            "Q9: Retention Power of Equity (Stock Options)": {
                "desc": "Evaluates turnover rates across stock option levels (Level 0 = No Equity, Level 1-3 = Granted Equity).",
                "sql": """SELECT 
    StockOptionLevel,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income,
    ROUND(AVG(TotalWorkingYears), 1) AS avg_experience_years
FROM employees
GROUP BY StockOptionLevel
ORDER BY StockOptionLevel ASC;"""
            },
            "Custom SQL Query Sandbox": {
                "desc": "Run any valid SQLite SQL query directly against the employees table.",
                "sql": """SELECT Department, JobRole, COUNT(*) AS count, 
       ROUND(AVG(MonthlyIncome), 0) AS avg_income
FROM employees 
GROUP BY Department, JobRole 
ORDER BY count DESC;"""
            }
        }

        selected_query_key = st.selectbox("Select Business Intelligence Query", list(sql_queries.keys()))
        selected_info = sql_queries[selected_query_key]

        st.caption(f"**Analytical Objective:** {selected_info['desc']}")

        if selected_query_key == "Custom SQL Query Sandbox":
            query_to_run = st.text_area("Write SQL Query:", value=selected_info["sql"], height=140)
        else:
            with st.expander("View Raw SQL Query", expanded=False):
                st.code(selected_info["sql"], language="sql")
            query_to_run = selected_info["sql"]

        try:
            df_sql = pd.read_sql_query(query_to_run, sql_conn)
            st.markdown(f"**Query Results:** `{len(df_sql)} records returned`")
            st.dataframe(df_sql, use_container_width=True)

            # Contextual Visual Charts for key questions
            if "Q2" in selected_query_key and "attrition_rate_pct" in df_sql.columns:
                st.markdown("#### Department Attrition Rate (%)")
                chart_data = df_sql.set_index("Department")[["attrition_rate_pct"]]
                st.bar_chart(chart_data)

            elif "Q4" in selected_query_key and "attrition_rate_pct" in df_sql.columns:
                st.markdown("#### Overtime & Work-Life Balance Attrition Rate (%)")
                df_sql["Category"] = df_sql["OverTime"].astype(str) + " OT | " + df_sql["work_life_balance_label"].astype(str)
                chart_data = df_sql.set_index("Category")[["attrition_rate_pct"]]
                st.bar_chart(chart_data)

            elif "Q9" in selected_query_key and "attrition_rate_pct" in df_sql.columns:
                st.markdown("#### Turnover by Stock Option Level (0 = No Equity)")
                chart_data = df_sql.set_index("StockOptionLevel")[["attrition_rate_pct"]]
                st.bar_chart(chart_data)

            # Export query results
            res_csv = df_sql.to_csv(index=False).encode("utf-8")
            st.download_button("Export Query Results (CSV)", data=res_csv, file_name="sql_query_result.csv", mime="text/csv")

        except Exception as sql_err:
            st.error(f"SQL Execution Error: {sql_err}")


# ══════════════════════════════════════════════════════════
# TAB 4 — Data Quality Contracts & Population Drift Monitor
# ══════════════════════════════════════════════════════════
with tab_data_quality:
    st.markdown("""
    <div style="margin-bottom:18px;">
        <h3 style="font-size:1.35rem;margin:0 0 4px 0;">Production Data Quality & Population Drift Monitor</h3>
        <p style="color:#94a3b8;font-size:0.88rem;">
            Enterprise ML systems degrade when input schemas drift or covariate distributions shift. 
            This portal enforces strict data quality contracts and continuously monitors <strong>Population Stability Index (PSI)</strong> 
            and <strong>Kolmogorov-Smirnov two-sample tests</strong> against calibrated training baselines.
        </p>
    </div>
    """, unsafe_allow_html=True)

    dq_subtab1, dq_subtab2 = st.tabs([
        "Data Quality Contract & Schema Audit",
        "Population Stability Index (PSI) & Drift Monitor"
    ])

    with dq_subtab1:
        st.markdown("<h4 style='font-size:1.1rem;margin-bottom:6px;'>Formal Data Contract Specification</h4>", unsafe_allow_html=True)
        st.caption("All incoming production inferences and batch rosters must conform to these strict boundary invariants:")

        contract_summary = []
        for feat, spec in SCHEMA_CONTRACT.items():
            if spec["type"] in ("numeric", "int"):
                rule = f"Numeric range [{spec['min']}, {spec['max']}], default: {spec['default']}"
            elif "allowed" in spec:
                rule = f"Categorical enum ({len(spec['allowed'])} classes): {', '.join(spec['allowed'][:4])}..."
            else:
                rule = f"Default: {spec.get('default', 'N/A')}"
            contract_summary.append({
                "Feature Name": feat,
                "Type": spec["type"].upper(),
                "Validation Rule / Permitted Domain": rule,
                "Imputation Strategy": "Median" if spec["type"] in ("numeric", "int") else "Mode"
            })

        st.dataframe(pd.DataFrame(contract_summary).head(10), use_container_width=True, hide_index=True)
        st.caption("Showing 10 of 30 enforced schema contract rules. Expand below to inspect complete specification.")

        with st.expander("View Complete 30-Feature Schema Contract"):
            st.dataframe(pd.DataFrame(contract_summary), use_container_width=True, hide_index=True)

        st.markdown("<div style='height:1px;background:rgba(255,255,255,0.06);margin:24px 0;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size:1.1rem;margin-bottom:6px;'>Live Ingestion Quality Audit</h4>", unsafe_allow_html=True)

        audit_source = st.radio(
            "Select Dataset for Quality Contract Audit:",
            ["Benchmark Historical Workforce (1,470 records)", "Upload New Production Batch CSV"],
            horizontal=True
        )

        df_to_audit = None
        if audit_source == "Benchmark Historical Workforce (1,470 records)":
            raw_path = os.path.join("data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
            if os.path.exists(raw_path):
                df_to_audit = pd.read_csv(raw_path)
        else:
            up_audit = st.file_uploader("Upload Batch CSV for Audit", type=["csv"], key="audit_uploader")
            if up_audit is not None:
                try:
                    df_to_audit = pd.read_csv(up_audit)
                except Exception as e:
                    st.error(f"Error reading uploaded CSV: {e}")

        if df_to_audit is not None:
            if st.button("Run Data Contract Audit", type="primary", key="btn_run_audit"):
                with st.spinner("Auditing records against schema contracts …"):
                    sanitized_df, report = validate_batch_dataframe(df_to_audit)
                    is_valid = report.get("is_valid", report.get("is_production_ready", True))

                st.markdown("<h4 style='margin-top:16px;'>Quality Scorecard</h4>", unsafe_allow_html=True)
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Ingested Records", f"{report.get('total_records', len(df_to_audit)):,}")
                score = report.get("data_quality_score", report.get("quality_score_pct", 100.0))
                q2.metric("Schema Conformance", f"{score:.1f}%")
                flagged = report.get("flagged_records", 0)
                q3.metric("Records Flagged", f"{flagged:,}")
                clippings = report.get("total_boundary_clippings", sum(report.get("out_of_bounds_clipped", {}).values()))
                q4.metric("Boundary Corrections", f"{clippings:,}")

                if score == 100.0:
                    st.success("All records fully adhere to the enterprise schema contract. Zero anomalies detected.")
                else:
                    st.warning(f"Detected anomalies across {flagged} data points. Automatic bounds clipping and categorical normalization applied safely.")

                with st.expander("Inspect Sanitized Dataset & Quality Audit Details"):
                    st.dataframe(sanitized_df.head(25), use_container_width=True)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if report.get("out_of_bounds_clipped"):
                            st.write("**Boundary Outliers Clipped:**", report["out_of_bounds_clipped"])
                        if report.get("null_imputations"):
                            st.write("**Null Values Imputed:**", report["null_imputations"])
                    with col_b:
                        if report.get("invalid_categories_normalized"):
                            st.write("**Invalid Categories Normalized:**", report["invalid_categories_normalized"])
                        if report.get("missing_columns_synthesized"):
                            st.write("**Missing Columns Synthesized:**", report["missing_columns_synthesized"])

    with dq_subtab2:
        st.markdown("<h4 style='font-size:1.1rem;margin-bottom:6px;'>Continuous Covariate Shift & Distribution Drift</h4>", unsafe_allow_html=True)
        st.caption("Quantifies distribution change using Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov tests.")

        raw_path = os.path.join("data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
        if not os.path.exists(raw_path):
            st.error("Baseline dataset not found at data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv.")
        else:
            baseline_data = pd.read_csv(raw_path)

            drift_source = st.radio(
                "Select Comparison Cohort:",
                ["Benchmark Baseline (Self-Test)", "Simulated Stressed Cohort (Shifted Comp & Overtime)"],
                horizontal=True
            )

            if drift_source == "Benchmark Baseline (Self-Test)":
                target_cohort = baseline_data.sample(n=min(500, len(baseline_data)), random_state=42)
            else:
                # Create a shifted cohort for demo
                target_cohort = baseline_data.copy()
                target_cohort["MonthlyIncome"] = target_cohort["MonthlyIncome"] * 1.35
                target_cohort["OverTime"] = np.random.choice(["Yes", "No"], size=len(target_cohort), p=[0.7, 0.3])

            if st.button("Evaluate Population Drift (PSI)", type="primary", key="btn_run_drift"):
                with st.spinner("Calculating Population Stability Index and statistical tests …"):
                    drift_report = compute_feature_drift(baseline_data, target_cohort)

                psi_score = drift_report["dataset_psi"]
                drift_lvl = drift_report["drift_level"]
                alert_badge = drift_report["alert_badge"]
                rec = drift_report["recommendation"]

                badge_color = "#10b981" if psi_score < 0.10 else ("#f59e0b" if psi_score < 0.20 else "#f43f5e")

                st.markdown(f"""
                <div style="background:rgba(18,24,38,0.75);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:20px;margin:18px 0;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="font-size:0.72rem;font-weight:700;color:#818cf8;letter-spacing:0.06em;text-transform:uppercase;">MLOPS TELEMETRY</span>
                            <h3 style="margin:4px 0 0 0;font-size:1.3rem;color:#ffffff;">Dataset Population Stability Index (PSI): <span style="color:{badge_color};">{psi_score:.4f}</span></h3>
                        </div>
                        <div style="padding:6px 14px;border-radius:9999px;font-size:0.76rem;font-weight:700;letter-spacing:0.05em;background:rgba(255,255,255,0.05);border:1px solid {badge_color};color:{badge_color};">
                            {alert_badge}
                        </div>
                    </div>
                    <div style="margin-top:12px;font-size:0.86rem;color:#cbd5e1;">
                        <strong>Production Directive:</strong> {rec}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Feature-Level Drift Telemetry")
                feat_rows = []
                for feat, met in drift_report["feature_metrics"].items():
                    ks_val = met.get("ks_statistic")
                    ks_str = f"{ks_val:.4f}" if ks_val is not None and not pd.isna(ks_val) else "—"
                    p_val = met.get("ks_pvalue")
                    p_str = f"{p_val:.4f}" if p_val is not None and not pd.isna(p_val) else "—"
                    psi_val = met.get("psi", 0.0)
                    psi_str = f"{psi_val:.4f}"

                    feat_rows.append({
                        "Feature": feat,
                        "Type": met["type"].capitalize(),
                        "PSI Score": psi_str,
                        "KS Statistic": ks_str,
                        "p-Value": p_str,
                        "Stability Status": met["status"]
                    })

                st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

