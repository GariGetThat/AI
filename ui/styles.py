# ui/styles.py

import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');

        :root {
            --bg: #FFFFFF;
            --surface: #F5F5F7;
            --surface-2: #EBEBF0;
            --text: #26262B;
            --muted: #808085;
            --border: #D1D1D6;
            --primary: #26262B;
            --blue: #4D80E6;
            --blue-soft: #EBF2FF;
            --red: #8C1F2E;
            --red-soft: rgba(140, 31, 46, 0.17);
            --green: #2E994D;
            --green-soft: #E0F5E6;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--text);
            background: var(--bg);
        }

        .stApp {
            background: var(--bg);
        }

        header[data-testid="stHeader"] {
            display: none;
        }

        div[data-testid="stToolbar"] {
            display: none;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 0rem;
            padding-bottom: 3rem;
        }

        /* Header */
        .app-header {
            height: 64px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
        }

        .brand {
            font-size: 16px;
            font-weight: 700;
            color: var(--text);
        }

        .brand-subtitle {
            margin-left: 28px;
            font-size: 11px;
            font-weight: 400;
            color: var(--muted);
        }

        /* Common */
        .step-badge {
            background: var(--surface);
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 12px;
            font-weight: 500;
            color: var(--muted);
            margin-bottom: 18px;
        }

        .page-title {
            font-size: 24px;
            font-weight: 700;
            line-height: 1.25;
            color: var(--text);
            margin-bottom: 8px;
        }

        .page-desc {
            font-size: 13px;
            font-weight: 400;
            color: var(--muted);
            margin-bottom: 24px;
        }

        .section-label {
            font-size: 12px;
            font-weight: 500;
            color: var(--muted);
            margin-bottom: 8px;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }

        .white-card {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }

        .dark-panel {
            background: #1F2129;
            border-radius: 12px;
            padding: 16px;
            color: #CCEBCC;
            font-family: monospace;
            font-size: 12px;
            line-height: 1.7;
            min-height: 320px;
        }

        .muted {
            color: var(--muted);
            font-size: 12px;
        }

        .small {
            font-size: 11px;
            color: var(--muted);
        }

        /* Buttons */
        .stButton > button {
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            height: 44px;
            padding: 0 24px;
            font-size: 13px;
            font-weight: 500;
        }

        .stButton > button:hover {
            background: #111114;
            color: white;
            border: none;
        }

        .secondary-button .stButton > button {
            background: var(--surface);
            color: var(--text);
            border: 1px solid var(--border);
        }

        /* Upload */
        div[data-testid="stFileUploader"] section {
            background: var(--surface);
            border: 1.5px solid var(--border);
            border-radius: 12px;
            padding: 24px;
        }

        div[data-testid="stFileUploader"] button {
            background: var(--primary);
            color: white;
            border-radius: 8px;
        }

        /* Inputs */
        textarea, input {
            border-radius: 12px !important;
            border: 1.5px solid var(--border) !important;
        }

        textarea:focus, input:focus {
            border-color: var(--blue) !important;
            box-shadow: none !important;
        }

        /* Chips */
        .chip {
            display: inline-block;
            background: #FFFFFF;
            border: 1.5px solid var(--border);
            border-radius: 17px;
            padding: 9px 14px;
            font-size: 12px;
            color: var(--text);
            margin-right: 8px;
            margin-bottom: 8px;
        }

        /* Pipeline */
        .pipeline-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }

        .pipeline-step {
            background: var(--surface);
            border-radius: 8px;
            padding: 12px;
            height: 76px;
        }

        .pipeline-icon {
            font-size: 22px;
            margin-bottom: 8px;
        }

        .pipeline-name {
            font-size: 12px;
            font-weight: 500;
            color: var(--text);
        }

        /* Feature cards */
        .feature-card {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            min-height: 170px;
            position: relative;
            overflow: hidden;
        }

        .feature-card::after {
            content: "";
            position: absolute;
            left: 0;
            bottom: 0;
            width: 100%;
            height: 3px;
            background: #B3D1FF;
        }

        .icon-badge {
            width: 44px;
            height: 44px;
            background: var(--surface);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            margin-bottom: 14px;
        }

        .feature-title {
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .feature-desc {
            font-size: 12px;
            color: var(--muted);
            line-height: 1.4;
        }

        /* Status */
        .status-success {
            background: var(--green-soft);
            border: 1px solid #80CC8C;
            border-radius: 8px;
            padding: 14px 16px;
            color: var(--green);
            font-size: 13px;
            font-weight: 500;
        }

        .status-running {
            background: var(--red-soft);
            border: 1px solid #B34040;
            border-radius: 6px;
            padding: 10px 14px;
            color: var(--red);
            font-size: 12px;
            font-weight: 500;
        }

        /* Person cards */
        .person-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
        }

        .person-card.selected {
            background: #F0F5FF;
            border: 2px solid var(--blue);
        }

        .person-thumb {
            height: 128px;
            background: var(--surface-2);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 44px;
            margin-bottom: 14px;
        }

        .person-name {
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .person-meta {
            font-size: 12px;
            color: var(--muted);
        }

        /* Timeline */
        .timeline-panel {
            background: #FFFFFF;
            border: 1.5px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }

        .timeline-row {
            display: grid;
            grid-template-columns: 80px 1fr;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }

        .timeline-label {
            font-size: 11px;
            color: var(--muted);
        }

        .timeline-track {
            height: 24px;
            background: var(--surface);
            border-radius: 4px;
            position: relative;
        }

        .timeline-segment {
            position: absolute;
            top: 0;
            height: 24px;
            border-radius: 4px;
            background: #EBF2FF;
        }

        .timeline-segment.object {
            background: rgba(65, 31, 140, 0.17);
        }

        /* Footer / helper */
        .hint-box {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 12px;
            color: var(--muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )