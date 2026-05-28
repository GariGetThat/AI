import streamlit as st

from ui.styles import inject_global_styles
from ui.pages import render_current_page


st.set_page_config(
    page_title="Gari-Get-That",
    page_icon="🎥",
    layout="wide",
)

inject_global_styles()
render_current_page()