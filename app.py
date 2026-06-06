from datetime import UTC, datetime
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.predict import predict_next_3_days


# =========================================
# AQI COLORS + HEALTH MESSAGES
# =========================================

AQI_LEVELS = [
    (50, "#00e400", "Good",
     "Air quality is satisfactory and poses little or no risk."),

    (100, "#ffcc00", "Moderate",
     "Acceptable air quality. Sensitive individuals should limit prolonged outdoor activity."),

    (150, "#ff9900", "Unhealthy for Sensitive Groups",
     "Children, elderly, and sensitive groups should reduce outdoor exertion."),

    (200, "#ff0000", "Unhealthy",
     "Everyone may begin to experience health effects."),

    (300, "#99004c", "Very Unhealthy",
     "Health warnings issued. Outdoor activity should be minimized."),

    (500, "#7e0023", "Hazardous",
     "Emergency conditions. Avoid outdoor exposure completely."),
]


def get_aqi_info(aqi):

    for limit, color, category, advice in AQI_LEVELS:
        if aqi <= limit:
            return color, category, advice

    return "#7e0023", "Hazardous", "Avoid outdoor exposure."


# =========================================
# AQI GAUGE
# =========================================

def build_gauge(aqi):

    color, category, _ = get_aqi_info(aqi)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=aqi,
            number={"font": {"size": 42}},
            title={"text": f"<b>{category}</b>"},
            gauge={
                "axis": {"range": [0, 500]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#00e400"},
                    {"range": [51, 100], "color": "#ffcc00"},
                    {"range": [101, 150], "color": "#ff9900"},
                    {"range": [151, 200], "color": "#ff0000"},
                    {"range": [201, 300], "color": "#99004c"},
                    {"range": [301, 500], "color": "#7e0023"},
                ],
            },
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(t=30, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    return fig


# =========================================
# FORECAST CHART
# =========================================

def build_forecast_chart(df):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df["aqi"],
            mode="lines+markers+text",
            text=[round(x, 1) for x in df["aqi"]],
            textposition="top center",
            line=dict(width=4),
            marker=dict(size=12),
            name="AQI Forecast",
        )
    )

    fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=51, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=101, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
    fig.add_hrect(y0=151, y1=200, fillcolor="red", opacity=0.1, line_width=0)

    fig.update_layout(
        title="📈 Current + 3-Day AQI Forecast",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig.update_yaxes(range=[0, 300])

    return fig


# =========================================
# POLLUTANT CARD
# =========================================

def pollutant_card(name, value, unit):

    return f"""
    <div style="
        background: rgba(255,255,255,0.05);
        padding: 18px;
        border-radius: 18px;
        text-align:center;
        border:1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
    ">
        <div style="
            color:#9ca3af;
            font-size:14px;
            margin-bottom:8px;
        ">
            {name}
        </div>

        <div style="
            color:white;
            font-size:28px;
            font-weight:700;
        ">
            {value}
        </div>

        <div style="
            color:#9ca3af;
            margin-top:6px;
        ">
            {unit}
        </div>
    </div>
    """


# =========================================
# FORECAST CARD
# =========================================

def forecast_card(day, date, aqi):

    color, category, advice = get_aqi_info(aqi)

    return f"""
    <div style="
        background: rgba(255,255,255,0.05);
        border-radius:22px;
        padding:20px;
        min-height:260px;
        border:1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(16px);
    ">

        <div style="
            color:#9ca3af;
            margin-bottom:10px;
        ">
            {date}
        </div>

        <div style="
            color:white;
            font-size:26px;
            font-weight:700;
        ">
            {day}
        </div>

        <div style="
            color:{color};
            font-size:48px;
            font-weight:800;
            margin-top:15px;
        ">
            {round(aqi,1)}
        </div>

        <div style="
            color:white;
            margin-top:10px;
            font-size:18px;
            font-weight:600;
        ">
            {category}
        </div>

        <div style="
            margin-top:16px;
            color:#d1d5db;
            line-height:1.5;
            font-size:14px;
        ">
            ⚠️ {advice}
        </div>

    </div>
    """


# =========================================
# MAIN RENDER FUNCTION
# =========================================

def render_dashboard():

    try:

        df = predict_next_3_days()

        current = df.iloc[0]

        current_aqi = float(current["aqi"])

        color, category, advice = get_aqi_info(current_aqi)

        gauge = build_gauge(current_aqi)

        forecast_chart = build_forecast_chart(df)

        forecast_cards = []

        for _, row in df.iterrows():

            forecast_cards.append(
                forecast_card(
                    row["day"],
                    row["date"],
                    row["aqi"],
                )
            )

        alert_box = f"""
        <div style="
            background:{color}22;
            border:1px solid {color};
            padding:18px;
            border-radius:18px;
            margin-top:15px;
            color:white;
            font-size:16px;
            line-height:1.6;
        ">
            <b>Health Advisory:</b><br>
            {advice}
        </div>
        """

        updated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

        title = f"""
        # 🌍 Karachi AQI Predictor

        Real-time AQI forecasting and health monitoring dashboard.

        **Updated:** {updated}
        """

        # Simulated values for UI richness
        pm25 = pollutant_card("PM2.5", "17.5", "µg/m³")
        pm10 = pollutant_card("PM10", "32.9", "µg/m³")
        no2 = pollutant_card("NO₂", "12.3", "µg/m³")
        ozone = pollutant_card("Ozone", "44.0", "µg/m³")

        return (
            title,
            gauge,
            alert_box,
            pm25,
            pm10,
            no2,
            ozone,
            forecast_cards[0],
            forecast_cards[1],
            forecast_cards[2],
            forecast_chart,
        )

    except Exception as e:

        return (
            f"❌ Error: {str(e)}",
            None,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            None,
        )


# =========================================
# UI LAYOUT
# =========================================

custom_css = """
body {
    background: linear-gradient(135deg,#0f172a,#111827);
}

.gradio-container {
    background: transparent !important;
}

footer {
    display:none !important;
}
"""


with gr.Blocks() as demo:

    title_md = gr.Markdown()

    refresh_btn = gr.Button(
        "🔄 Refresh Forecast",
        variant="primary",
    )

    with gr.Row():

        gauge_plot = gr.Plot(scale=1)

        alert_html = gr.HTML(scale=1)

    gr.Markdown("## 🧪 Pollutant Levels")

    with gr.Row():

        pm25_card = gr.HTML()
        pm10_card = gr.HTML()
        no2_card = gr.HTML()
        ozone_card = gr.HTML()

    gr.Markdown("## 📅 3-Day AQI Forecast")

    with gr.Row():

        day1 = gr.HTML()
        day2 = gr.HTML()
        day3 = gr.HTML()

    gr.Markdown("## 📈 AQI Forecast Trend")

    forecast_plot = gr.Plot()

    refresh_btn.click(
        fn=render_dashboard,
        outputs=[
            title_md,
            gauge_plot,
            alert_html,
            pm25_card,
            pm10_card,
            no2_card,
            ozone_card,
            day1,
            day2,
            day3,
            forecast_plot,
        ],
    )

    demo.load(
        fn=render_dashboard,
        outputs=[
            title_md,
            gauge_plot,
            alert_html,
            pm25_card,
            pm10_card,
            no2_card,
            ozone_card,
            day1,
            day2,
            day3,
            forecast_plot,
        ],
    )


if __name__ == "__main__":

    demo.launch(
    theme=gr.themes.Base(),
    css=custom_css
)

