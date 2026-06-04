"""
CyberGuard – AI-Based Cyberbullying Detection System
Streamlit demo with interactive dashboard.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import LABEL_COLORS, LABELS
from src.predict import load_model_bundle, predict_text
from src.utils import load_metrics

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberGuard | AI Cyberbullying Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: linear-gradient(165deg, #0a0e1a 0%, #111827 40%, #0f172a 100%);
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(139,92,246,0.12) 50%, rgba(16,185,129,0.08) 100%);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}

.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}

.metric-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(71, 85, 105, 0.5);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(8px);
}

.metric-card h3 {
    color: #e2e8f0;
    font-size: 1.75rem;
    margin: 0.25rem 0;
}

.metric-card span {
    color: #64748b;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.result-safe {
    background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.05));
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 1.5rem;
}

.result-toxic {
    background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(245,158,11,0.05));
    border: 2px solid #f59e0b;
    border-radius: 16px;
    padding: 1.5rem;
}

.result-severe {
    background: linear-gradient(135deg, rgba(239,68,68,0.25), rgba(239,68,68,0.05));
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 1.5rem;
}

.stTextArea textarea {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}

div[data-testid="stSidebar"] .stMarkdown h1 {
    color: #60a5fa;
}

.sample-btn {
    margin: 0.15rem 0;
}

/* Hide Streamlit Deploy button */
[data-testid="stAppDeployButton"],
.stAppDeployButton {
    display: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model_bundle()


def result_css(label: str) -> str:
    if label == "Safe":
        return "result-safe"
    if label == "Toxic":
        return "result-toxic"
    return "result-severe"


def render_hero(subtitle: str) -> None:
    st.markdown(
        f"""
<div class="hero">
    <h1>🛡️ CyberGuard</h1>
    <p>{subtitle}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def probability_chart(probabilities: dict) -> go.Figure:
    labels = list(probabilities.keys())
    values = [probabilities[l] * 100 for l in labels]
    colors = [LABEL_COLORS.get(l, "#64748b") for l in labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=10, r=60, t=20, b=20),
        xaxis=dict(range=[0, 105], title="Confidence %"),
        showlegend=False,
        font=dict(family="Space Grotesk"),
    )
    return fig


def gauge_chart(confidence: float, label: str) -> go.Figure:
    color = LABEL_COLORS.get(label, "#64748b")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": "Prediction Confidence", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "bgcolor": "rgba(30,41,59,0.5)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(34,197,94,0.15)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.15)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.2)"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def show_prediction(result: dict, key_prefix: str = "") -> None:
    label = result["label"]
    conf = result["confidence"]
    css = result_css(label)

    emoji = {"Safe": "✅", "Toxic": "⚠️", "Severe Bullying": "🚨"}.get(label, "❓")

    st.markdown(
        f"""
<div class="{css}">
    <h2 style="margin:0;color:{LABEL_COLORS.get(label,'#fff')}">{emoji} {label}</h2>
    <p style="color:#94a3b8;margin:0.5rem 0 0 0">Model confidence: <strong>{conf*100:.1f}%</strong></p>
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.plotly_chart(
            probability_chart(result["probabilities"]),
            use_container_width=True,
            key=f"{key_prefix}_bar",
        )
    with col2:
        st.plotly_chart(
            gauge_chart(conf, label),
            use_container_width=True,
            key=f"{key_prefix}_gauge",
        )

    if label != "Safe":
        st.warning(
            "⚠️ This content may be harmful. Consider reporting it to a moderator, "
            "parent, or school counselor. If you feel unsafe, reach out to a trusted adult."
        )


# ─── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🛡️ CyberGuard")
    st.caption("AI-Powered Cyberbullying Detection")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Analyze Text", "📁 Batch Analysis", "📊 Model Insights", "ℹ️ About Project"],
        label_visibility="collapsed",
    )

    st.divider()
    metrics = load_metrics()
    if metrics:
        st.metric("Test Accuracy", f"{metrics['test']['accuracy']*100:.1f}%")
        st.metric("F1 Score", f"{metrics['test']['f1_macro']*100:.1f}%")
    else:
        st.info("Train the model to see metrics.")


# ─── Load model ────────────────────────────────────────────────────────────────
try:
    model_bundle = get_model()
    model_ready = True
except FileNotFoundError:
    model_ready = False
    model_bundle = None


# ─── Pages ─────────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    render_hero(
        "Real-time NLP & Machine Learning system to detect toxic and bullying content online."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><span>Categories</span><h3>3</h3></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><span>Technique</span><h3>TF-IDF</h3></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><span>Classifier</span><h3>LogReg</h3></div>', unsafe_allow_html=True)
    with c4:
        status = "Ready" if model_ready else "Not Trained"
        st.markdown(
            f'<div class="metric-card"><span>Model</span><h3>{status}</h3></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### How it works")
    st.markdown(
        """
1. **Input** – User enters a tweet, comment, or message.  
2. **Preprocessing** – NLTK cleans text (lowercase, remove URLs, lemmatize).  
3. **Vectorization** – TF-IDF converts text to numerical features.  
4. **Classification** – Logistic Regression predicts: **Safe**, **Toxic**, or **Severe Bullying**.
        """
    )

    if not model_ready:
        st.error("Model not found. Run `python -m src.train` then refresh this page.")
    else:
        st.success("Model loaded. Go to **Analyze Text** to try a live prediction.")

    if st.session_state.history:
        st.markdown("### Recent analyses")
        hist_df = pd.DataFrame(st.session_state.history[-8:])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


elif page == "🔍 Analyze Text":
    render_hero("Paste any social media comment or message for instant AI classification.")

    if not model_ready:
        st.error("Please train the model first: `python -m src.train`")
        st.stop()

    samples = {
        "Safe example": "Had an amazing day at school with my friends!",
        "Toxic example": "You're so annoying, nobody likes you here.",
        "Severe example": "I hope you die, nobody would miss you at all.",
    }

    st.markdown("**Quick samples**")
    scols = st.columns(3)
    for col, (name, text) in zip(scols, samples.items()):
        with col:
            if st.button(name, use_container_width=True):
                st.session_state["input_text"] = text

    default = st.session_state.get("input_text", "")
    user_text = st.text_area(
        "Enter text to analyze",
        value=default,
        height=140,
        placeholder="Type or paste a comment, tweet, or chat message...",
    )

    analyze = st.button("🔍 Analyze Now", type="primary", use_container_width=True)

    if analyze:
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing with NLP + ML pipeline..."):
                result = predict_text(user_text, model_bundle)

            show_prediction(result, key_prefix="single")

            st.session_state.history.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "text": user_text[:80] + ("..." if len(user_text) > 80 else ""),
                    "prediction": result["label"],
                    "confidence": f"{result['confidence']*100:.1f}%",
                }
            )

            with st.expander("Preprocessed text (debug)"):
                st.code(result.get("processed_text", ""))


elif page == "📁 Batch Analysis":
    render_hero("Upload a CSV file with a text column for bulk cyberbullying screening.")

    if not model_ready:
        st.error("Please train the model first: `python -m src.train`")
        st.stop()

    st.markdown(
        "CSV must contain a column named `text`, `tweet`, or `comment`. "
        "Optional: download [sample template](data/sample_upload_template.csv)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        text_cols = [c for c in df.columns if c.lower() in ("text", "tweet", "comment", "message")]
        text_col = text_cols[0] if text_cols else df.columns[0]

        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Run batch analysis", type="primary"):
            texts = df[text_col].astype(str).tolist()
            progress = st.progress(0)
            results = []
            for i, t in enumerate(texts):
                results.append(predict_text(t, model_bundle))
                progress.progress((i + 1) / len(texts))

            df["prediction"] = [r["label"] for r in results]
            df["confidence"] = [f"{r['confidence']*100:.1f}%" for r in results]

            st.success(f"Analyzed {len(df)} rows.")
            st.dataframe(df, use_container_width=True)

            counts = df["prediction"].value_counts().reset_index()
            counts.columns = ["Category", "Count"]
            fig = px.pie(
                counts,
                values="Count",
                names="Category",
                color="Category",
                color_discrete_map=LABEL_COLORS,
                hole=0.45,
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                csv_out,
                file_name="cyberguard_results.csv",
                mime="text/csv",
            )


elif page == "📊 Model Insights":
    render_hero("Training metrics and class distribution from the ML pipeline.")

    metrics = load_metrics()
    if not metrics:
        st.warning("No training metrics yet. Run `python -m src.train`.")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Training samples", f"{metrics['samples']:,}")
    m2.metric("Test accuracy", f"{metrics['test']['accuracy']*100:.2f}%")
    m3.metric("Test F1 (macro)", f"{metrics['test']['f1_macro']*100:.2f}%")

    report = metrics["test"]["report"]
    per_class = pd.DataFrame(
        {
            "Class": LABELS,
            "Precision": [report[l]["precision"] * 100 for l in LABELS],
            "Recall": [report[l]["recall"] * 100 for l in LABELS],
            "F1": [report[l]["f1-score"] * 100 for l in LABELS],
        }
    )

    fig_bar = px.bar(
        per_class.melt(id_vars="Class", var_name="Metric", value_name="Score"),
        x="Class",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=["#60a5fa", "#a78bfa", "#34d399"],
    )
    fig_bar.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", yaxis_title="Score (%)")
    st.plotly_chart(fig_bar, use_container_width=True)

    cm = metrics["test"]["confusion_matrix"]
    fig_cm = px.imshow(
        cm,
        x=LABELS,
        y=LABELS,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual"),
    )
    fig_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("Full metrics JSON"):
        st.json(metrics)


else:  # About
    render_hero("AI-powered cyberbullying detection for safer online communities")

    st.markdown(
        """
### Purpose
Cyberbullying affects students' mental health on social media. **CyberGuard** automatically
analyzes text to detect bullying, toxic, or abusive content so schools, parents, and platforms
can take timely action.

### AI Techniques Used
- **Natural Language Processing (NLP)** – tokenization, stopwords, lemmatization (NLTK)
- **Machine Learning** – TF-IDF feature extraction + Logistic Regression (Scikit-learn)
- **Dataset** – Kaggle Cyberbullying Tweet Dataset (47k+ labeled tweets)

### 4-Week Timeline
| Week | Activity |
|------|----------|
| 1 | Data preprocessing |
| 2 | Model training |
| 3 | Web app (Streamlit) |
| 4 | Testing & report |
        """
    )

    st.markdown("### Tools & Technologies")
    tools = pd.DataFrame(
        {
            "Tool": ["Python", "Jupyter Notebook", "Scikit-learn", "NLTK", "Pandas", "NumPy", "Streamlit", "Kaggle Dataset"],
            "Purpose": [
                "Programming",
                "Training & EDA",
                "Machine Learning",
                "NLP preprocessing",
                "Data handling",
                "Numerical operations",
                "Web demo interface",
                "Labeled tweets",
            ],
        }
    )
    st.table(tools)
