"""
Streamlit Web App — AQI Predictor Dashboard

Shows:
1. Current AQI for Karachi
2. 3-day AQI forecast
3. AQI trend chart
4. Health recommendations based on AQI level
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_next_3_days, fetch_current_conditions, get_aqi_category


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)


# ── Helper Functions ──────────────────────────────────────────────────────────

def get_health_advice(category: str) -> str:
    """Return health advice based on AQI category."""
    advice = {
        "Good":                              "✅ Air quality is good. Enjoy outdoor activities!",
        "Moderate":                          "⚠️ Acceptable air quality. Unusually sensitive people should limit prolonged outdoor exertion.",
        "Unhealthy for Sensitive Groups":    "🟠 Sensitive groups should reduce outdoor activity. Others can continue normally.",
        "Unhealthy":                         "🔴 Everyone should reduce prolonged outdoor exertion. Wear a mask outdoors.",
        "Very Unhealthy":                    "🟣 Health alert! Avoid outdoor activities. Keep windows closed.",
        "Hazardous":                         "⚫ Emergency conditions! Everyone should stay indoors. Avoid all outdoor activity.",
    }
    return advice.get(category, "No advice available.")


def aqi_gauge(aqi: float, title: str):
    """Create a gauge chart for AQI value."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 300]},
            "bar":  {"color": "darkblue"},
            "steps": [
                {"range": [0, 50],   "color": "#00e400"},
                {"range": [50, 100], "color": "#ffff00"},
                {"range": [100, 150],"color": "#ff7e00"},
                {"range": [150, 200],"color": "#ff0000"},
                {"range": [200, 300],"color": "#8f3f97"},
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # Header
    st.title("🌫️ Karachi AQI Predictor")
    st.markdown("**Real-time Air Quality Index monitoring and 3-day forecast for Karachi, Pakistan**")
    st.markdown("---")

    # Load current conditions
    with st.spinner("Fetching current air quality data..."):
        try:
            conditions = fetch_current_conditions()
            current_aqi = conditions["aqi"]
            category, color = get_aqi_category(current_aqi)
        except Exception as e:
            st.error(f"Failed to fetch current data: {e}")
            return

    # ── Current AQI Section ───────────────────────────────────────────────────
    st.subheader("📍 Current Air Quality — Karachi")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.plotly_chart(aqi_gauge(current_aqi, "Current AQI"), use_container_width=True)

    with col2:
        st.markdown(f"### AQI: **{current_aqi}**")
        st.markdown(f"**Category:** <span style='color:{color}'>{category}</span>", unsafe_allow_html=True)
        st.markdown(f"**Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        st.markdown("---")
        st.markdown(f"🌡️ Temperature: **{conditions['temperature']}°C**")
        st.markdown(f"💧 Humidity: **{conditions['humidity']}%**")
        st.markdown(f"💨 Wind Speed: **{conditions['wind_speed']} km/h**")

    with col3:
        st.info(get_health_advice(category))
        st.markdown("**Pollutant Levels:**")
        pol_col1, pol_col2 = st.columns(2)
        with pol_col1:
            st.metric("PM2.5", f"{conditions['pm2_5']} µg/m³")
            st.metric("PM10",  f"{conditions['pm10']} µg/m³")
        with pol_col2:
            st.metric("NO₂",   f"{conditions['no2']} µg/m³")
            st.metric("Ozone", f"{conditions['ozone']} µg/m³")

    st.markdown("---")

    # ── 3-Day Forecast Section ────────────────────────────────────────────────
    st.subheader("📅 3-Day AQI Forecast")

    with st.spinner("Running AI predictions..."):
        try:
            predictions = predict_next_3_days()
        except Exception as e:
            st.error(f"Failed to generate predictions: {e}")
            st.info("Make sure models are trained. Run: python -m src.training_pipeline")
            return

    # Show forecast cards
    cols = st.columns(3)
    for i, (_, row) in enumerate(predictions.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div style='background-color:{row['color']}22; border-left: 5px solid {row['color']};
                        padding: 15px; border-radius: 8px; text-align: center;'>
                <h3>{row['day']}</h3>
                <p style='font-size:12px; color:gray;'>{row['date']}</p>
                <h1 style='color:{row['color']};'>{row['aqi']}</h1>
                <p><b>{row['category']}</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(get_health_advice(row["category"]))

    st.markdown("---")

    # ── Forecast Chart ────────────────────────────────────────────────────────
    st.subheader("📈 AQI Forecast Chart")

    fig = go.Figure()

    # Current AQI point
    fig.add_trace(go.Scatter(
        x=["Now"],
        y=[current_aqi],
        mode="markers+text",
        marker=dict(size=12, color=color),
        text=[f"Now: {current_aqi}"],
        textposition="top center",
        name="Current"
    ))

    # Forecast points
    fig.add_trace(go.Scatter(
        x=predictions["date"].tolist(),
        y=predictions["aqi"].tolist(),
        mode="lines+markers+text",
        marker=dict(size=10),
        line=dict(width=2, dash="dash"),
        text=[str(v) for v in predictions["aqi"].tolist()],
        textposition="top center",
        name="Forecast"
    ))

    # AQI level bands
    fig.add_hrect(y0=0,   y1=50,  fillcolor="#00e400", opacity=0.1, line_width=0, annotation_text="Good")
    fig.add_hrect(y0=50,  y1=100, fillcolor="#ffff00", opacity=0.1, line_width=0, annotation_text="Moderate")
    fig.add_hrect(y0=100, y1=150, fillcolor="#ff7e00", opacity=0.1, line_width=0, annotation_text="Unhealthy (Sensitive)")
    fig.add_hrect(y0=150, y1=200, fillcolor="#ff0000", opacity=0.1, line_width=0, annotation_text="Unhealthy")
    fig.add_hrect(y0=200, y1=300, fillcolor="#8f3f97", opacity=0.1, line_width=0, annotation_text="Very Unhealthy")

    fig.update_layout(
        title="Current + 3-Day AQI Forecast",
        xaxis_title="Date",
        yaxis_title="AQI",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Historical Data ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Historical AQI Data")

    try:
        df = pd.read_parquet("data/features.parquet")
        df = df.sort_values("timestamp").tail(168)  # last 7 days hourly

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["aqi"],
            mode="lines",
            line=dict(color="#1f77b4", width=1.5),
            name="Historical AQI"
        ))
        fig2.update_layout(
            title="Last 7 Days — Hourly AQI",
            xaxis_title="Date",
            yaxis_title="AQI",
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        st.info("Historical chart will appear after data is collected.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("Built with ❤️ | Data: OpenMeteo | Models: XGBoost, Random Forest | Registry: DagHub MLflow")


if __name__ == "__main__":
    main()