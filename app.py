import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Page config
st.set_page_config(
    page_title="F1 Podium Prediction Dashboard",
    layout="wide"
)

# Basic styling 
F1_CSS = """
<style>
body {
    background-color: #0c0c0c;
}
.reportview-container .main .block-container {
    background-color: #111111;
    color: #e0e0e0;
}
h1, h2, h3, h4 {
    color: #f44336;
}
div.stMetric {
    background-color: #1b1b1b;
    border-radius: 8px;
    padding: 8px;
}
.sidebar .sidebar-content {
    background-color: #151515;
}
.stButton>button {
    background-color: #f44336;
    color: white;
    border-radius: 4px;
    border: none;
}
.stButton>button:hover {
    background-color: #d32f2f;
}
.streamlit-expanderHeader {
    font-weight: 600;
}
</style>
"""
st.markdown(F1_CSS, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("rf_podium_model.joblib")
    return model

rf_model = load_model()

# Layout: Tabs
st.title("F1 Podium Prediction Dashboard")

tabs = st.tabs(["Podium Prediction", "Model & Data Insights", "Weather Impact Analysis"])

# TAB 1: Podium Prediction
with tabs[0]:
    st.subheader("Podium probability for a configured race scenario")

    st.sidebar.title("Race Scenario Inputs")
    st.sidebar.markdown("Configure the pre-race conditions for a single driver below.")

    grid = st.sidebar.slider("Grid position", 1, 20, 5)

    driver_avg_finish_last3 = st.sidebar.slider(
        "Driver average finish (last 3 races)",
        1.0, 20.0, 7.0, 0.5,
        help="Lower is better. For example, 3.0 ≈ consistently finishing around P3."
    )

    driver_points_last5 = st.sidebar.slider(
        "Driver points (last 5 races)",
        0.0, 125.0, 30.0, 1.0,
        help="Maximum is 125 (25 points per win × 5 races)."
    )

    driver_dnfs_last5 = st.sidebar.slider(
        "Driver DNFs (last 5 races)",
        0, 5, 0,
        help="How many times the driver did not finish in the last 5 races."
    )

    constructor_points_last5 = st.sidebar.slider(
        "Constructor points (last 5 races)",
        0.0, 150.0, 50.0, 1.0,
        help="Total points scored by the team in the last 5 races."
    )

    constructor_wins_last10 = st.sidebar.slider(
        "Constructor wins (last 10 races)",
        0, 10, 2,
        help="Number of wins for this team across the last 10 races."
    )

    constructor_dnfs_last5 = st.sidebar.slider(
        "Constructor DNFs (last 5 races)",
        0, 5, 1,
        help="Total DNFs for either car from this team in the last 5 races."
    )

    predict_button = st.sidebar.button("Predict podium probability")

    st.markdown(
        """
This tab estimates the probability that a driver will finish on the podium
based on grid position, recent driver form and recent constructor performance.
The underlying model is a Random Forest classifier trained on historical F1 race data.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="ROC–AUC", value="0.95")
    with col2:
        st.metric(label="Podium recall (class 1)", value="0.94")
    with col3:
        st.metric(label="Model", value="Random Forest")

    st.markdown("---")

    if predict_button:
        features = np.array([[
            grid,
            driver_avg_finish_last3,
            driver_points_last5,
            driver_dnfs_last5,
            constructor_points_last5,
            constructor_wins_last10,
            constructor_dnfs_last5
        ]])

        prob = rf_model.predict_proba(features)[0, 1]
        prob_pct = prob * 100

        st.markdown(f"### Predicted podium probability: **{prob_pct:.1f}%**")

        if prob > 0.70:
            st.success(
                "High podium likelihood. The combination of grid, recent form and team performance "
                "is strongly favourable."
            )
        elif prob > 0.40:
            st.warning(
                "Moderate podium likelihood. A strong result is possible, depending on strategy and race events."
            )
        else:
            st.info(
                "Low podium likelihood under typical race conditions. A podium would likely require "
                "unusual circumstances or high attrition ahead."
            )

        with st.expander("Input summary"):
            st.write({
                "grid": grid,
                "driver_avg_finish_last3": driver_avg_finish_last3,
                "driver_points_last5": driver_points_last5,
                "driver_dnfs_last5": driver_dnfs_last5,
                "constructor_points_last5": constructor_points_last5,
                "constructor_wins_last10": constructor_wins_last10,
                "constructor_dnfs_last5": constructor_dnfs_last5,
            })
    else:
        st.info("Use the controls in the sidebar and click 'Predict podium probability' to generate a prediction.")
        
    st.subheader("Model interpretation")

    st.markdown(
    """
**Key drivers of the prediction:**

- **Grid position** – starting closer to the front naturally increases podium chances.  
- **Driver recent form** – strong average finishing positions and consistent points in the last few races
  signal good momentum.  
- **Constructor performance** – high recent team points and wins indicate a competitive car package.  
- **Reliability** – repeated DNFs for either the driver or team tend to reduce expected performance.

This application is intended as an **analytics and exploration tool**, not a deterministic race result
generator. It shows how different pre-race factors can influence the podium probability according to
the trained model.
"""
)

# TAB 2: Model & Data Insights
with tabs[1]:
    st.subheader("Model and data insights")

    st.markdown(
        """
This section summarises key patterns from the historical dataset and how the model
uses them to estimate podium probability.
"""
    )

    st.markdown("#### Podium probability vs grid position")
    try:
        st.image("podium_by_grid.png", caption="Podium probability as a function of starting grid position.")
    except Exception:
        st.info("Add 'podium_by_grid.png' to the app folder to display this plot.")

    st.markdown("#### Feature importance (Random Forest)")
    try:
        st.image("feature_importance.png", caption="Relative importance of each input feature in the Random Forest model.")
    except Exception:
        st.info("Add 'feature_importance.png' to the app folder to display this plot.")

    st.markdown(
        """
Key observations:

- Starting closer to the front significantly increases podium probability.  
- Recent driver and constructor performance contribute strongly to the prediction.  
- Reliability-related features (DNFs) act as penalties rather than primary drivers.
"""
    )

# TAB 3: Weather Impact Analysis
with tabs[2]:
    st.subheader("Weather impact (exploratory analysis)")

    st.markdown(
        """
This section compares dry and wet races using a proxy-based labelling of historically
rain-affected Grands Prix. It is treated as **exploratory** rather than a predictive input.
"""
    )

    # Attempt to show summary stats table if available
    try:
        wet_stats = pd.read_csv("wet_stats_summary.csv", index_col=0)
        st.markdown("#### Summary statistics: dry vs wet races")
        st.dataframe(wet_stats)
    except Exception:
        st.info("Add 'wet_stats_summary.csv' to the app folder to display summary statistics.")

    st.markdown("#### DNF rate: wet vs dry")
    try:
        st.image("wet_dnf_rate.png", caption="Comparison of DNF rate between dry and wet races.")
    except Exception:
        st.info("Add 'wet_dnf_rate.png' to the app folder to display this plot.")

    st.markdown("#### Average position change: wet vs dry")
    try:
        st.image("wet_position_gain.png", caption="Average (grid - finish) comparison between dry and wet races.")
    except Exception:
        st.info("Add 'wet_position_gain.png' to the app folder to display this plot.")

    st.markdown(
        """
Note: due to the limited number of identified wet races and proxy-based labelling,
the weather analysis is subject to sampling bias and should be interpreted with caution.
"""
    )


