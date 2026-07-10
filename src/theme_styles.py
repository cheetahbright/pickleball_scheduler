#!/usr/bin/env python3
"""Dark theme CSS, injected on demand based on the user's saved preference.

Streamlit's default look is already light, so "light" means "inject
nothing" - only "dark" needs an override. This targets the same kind of
concrete selectors as mobile_styles.py rather than Streamlit's internal CSS
custom properties, which vary between Streamlit versions.
"""

from __future__ import annotations

THEME_CSS_DARK = """
<style>
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}

section[data-testid="stSidebar"] {
    background-color: #161a23;
}

.stMarkdown, .stMarkdown p, label, h1, h2, h3, h4, h5, h6 {
    color: #fafafa !important;
}

.stButton > button, .stDownloadButton > button {
    background-color: #262730;
    color: #fafafa;
    border: 1px solid #3d4048;
}

div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: #262730;
    color: #fafafa;
}

.stDataFrame, .stTable {
    background-color: #161a23;
    color: #fafafa;
}

div[data-baseweb="tab-list"] {
    background-color: #0e1117;
}

div[data-testid="stMetricValue"] {
    color: #fafafa;
}
</style>
"""


def inject_theme_css(st_module, theme: str) -> None:
    """Render the dark-theme CSS block when theme == "dark". A no-op for
    "light" (or any other value) since Streamlit's default is already light."""
    if theme == "dark":
        st_module.markdown(THEME_CSS_DARK, unsafe_allow_html=True)
