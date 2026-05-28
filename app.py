from datetime import datetime

import gradio as gr
import plotly.express as px

from src.predict import predict_next_3_days


def badge_html(category: str, color: str) -> str:
    return (
        f"<span style='background:{color}; padding:4px 10px; border-radius:999px;"
        "font-weight:600; color:#111;'>"
        f"{category}</span>"
    )


def build_alerts(df):
    hazardous = df[df["aqi"] >= 301]
    very_unhealthy = df[(df["aqi"] >= 201) & (df["aqi"] < 301)]

    if not hazardous.empty:
        return "🚨 Hazardous AQI forecast detected. Minimize outdoor exposure."
    if not very_unhealthy.empty:
        return "⚠️ Very Unhealthy AQI forecast detected. Sensitive groups should stay indoors."
    return "✅ No hazardous AQI level forecast in the next 3 days."


def render_dashboard():
    df = predict_next_3_days()
    df["badge"] = [badge_html(cat, color) for cat, color in zip(df["category"], df["color"])]

    fig = px.line(df, x="date", y="aqi", markers=True, title="Karachi AQI Forecast (Next 3 Days)")
    fig.update_yaxes(range=[0, 500])

    summary = (
        f"### Karachi AQI Predictor\n"
        f"Last updated (UTC): **{datetime.utcnow().strftime('%Y-%m-%d %H:%M')}**\n\n"
        f"{build_alerts(df)}"
    )

    table = df[["day", "date", "aqi", "category"]]
    return summary, table, fig


with gr.Blocks(title="Karachi AQI Predictor") as demo:
    gr.Markdown("# 🌍 Karachi AQI Predictor (Serverless)")
    gr.Markdown("Predicts AQI for the next 3 days using latest production models from DagsHub MLflow registry.")

    refresh = gr.Button("Refresh Forecast")
    summary = gr.Markdown()
    table = gr.Dataframe(headers=["day", "date", "aqi", "category"], interactive=False)
    chart = gr.Plot()

    refresh.click(fn=render_dashboard, outputs=[summary, table, chart])
    demo.load(fn=render_dashboard, outputs=[summary, table, chart])


if __name__ == "__main__":
    demo.launch()
