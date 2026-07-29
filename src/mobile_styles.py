#!/usr/bin/env python3
"""Mobile-responsive CSS injected once at app startup.

Streamlit's `st.columns` already stacks vertically below its own internal
breakpoint, and `layout="wide"` is otherwise fine on mobile - but the default
touch targets (buttons, selects, sliders) are sized for a mouse, and wide
tables/dataframes overflow the viewport instead of scrolling. This is scoped
to a single `@media (max-width: 640px)` block so it changes nothing on
desktop.
"""

from __future__ import annotations

MOBILE_BREAKPOINT_PX = 640

MOBILE_CSS = f"""
<style>
@media (max-width: {MOBILE_BREAKPOINT_PX}px) {{
    /* Larger touch targets for buttons and inputs - the default sizing
       assumes mouse precision, which mobile taps don't have. */
    .stButton > button,
    .stDownloadButton > button {{
        min-height: 44px;
        font-size: 1rem;
        width: 100%;
    }}

    div[data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input {{
        min-height: 44px;
        font-size: 1rem;
    }}

    /* Sliders' default thumb is far smaller than a comfortable 44px tap
       target - the weight and skill-rating sliders are otherwise unusable
       with a fingertip. Padding on an ANCESTOR container doesn't extend a
       descendant element's own hit-box, so it has to go on the thumb
       itself: content-box sizing keeps the visible thumb at its normal 24px
       while padding grows the actual clickable/tappable box to 44px, and
       background-clip keeps the painted color from spreading into that
       padding so it still looks like a 24px thumb. */
    .stSlider {{
        padding-top: 12px;
        padding-bottom: 12px;
    }}
    .stSlider [role="slider"] {{
        box-sizing: content-box;
        width: 24px;
        height: 24px;
        padding: 10px;
        background-clip: content-box;
    }}

    /* Wide tables/dataframes must scroll horizontally instead of forcing
       the whole page to overflow sideways. */
    .stDataFrame, .stTable {{
        overflow-x: auto;
    }}

    /* Reclaim horizontal space on narrow screens. */
    .block-container {{
        padding-left: 1rem;
        padding-right: 1rem;
    }}

    /* Metric values shrink so 3-4 column metric rows don't wrap
       into an unreadable stack of overlapping text. */
    div[data-testid="stMetricValue"] {{
        font-size: 1.25rem;
    }}
}}
</style>
"""


def inject_mobile_css(st_module) -> None:
    """Render the mobile CSS block. Call once per app render (idempotent - it's
    just a <style> tag, safe to emit on every Streamlit rerun)."""
    st_module.markdown(MOBILE_CSS, unsafe_allow_html=True)
