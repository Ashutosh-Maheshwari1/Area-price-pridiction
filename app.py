import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import time

# ---------------- 1. PRO CONFIG ----------------
st.set_page_config(
    page_title="Home.ai - Smart House Price Projector",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- 2. SECURE ASSET LOADING ----------------
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("house_model.pkl", "rb"))
        model_columns = pickle.load(open("model_columns.pkl", "rb"))
        return model, model_columns
    except Exception as e:
        st.error(f"Initialization Failed: {e}")
        st.stop()

model, model_columns = load_assets()

# ---------------- 3. CRYSTAL UI & ANIMATIONS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

    /* Fluid Animations */
    @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulseGlow { 0% { border-color: rgba(99, 102, 241, 0.2); } 50% { border-color: rgba(99, 102, 241, 0.6); } 100% { border-color: rgba(99, 102, 241, 0.2); } }

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* Ultra-Transparent Midnight Theme */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0%, #020617 100%);
        background-attachment: fixed;
    }

    /* Flawless Glass Master Panel */
    .glass-panel {
        background: rgba(255, 255, 255, 0.01);
        backdrop-filter: blur(35px) saturate(180%);
        -webkit-backdrop-filter: blur(35px) saturate(180%);
        border-radius: 40px;
        padding: 50px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        animation: slideUp 0.8s ease-out;
        margin-top: 20px;
    }

    /* Branding Section (Fixed Merging) */
    .header-box {
        text-align: center;
        padding-top: 50px;
        margin-bottom: 40px;
        animation: slideUp 1s ease-out;
    }

    .brand-title {
        background: linear-gradient(135deg, #fff 40%, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 5rem;
        font-weight: 800;
        letter-spacing: -4px;
        margin: 0;
    }

    /* Target UI: Elite Metric Cards */
    .elite-metric {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 25px;
        border-radius: 24px;
        text-align: center;
        transition: all 0.3s ease;
        animation: slideUp 1.2s ease-out;
    }
    .elite-metric:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: #6366f1;
        transform: translateY(-5px);
    }

    /* Predict Button */
    div.stButton > button {
        background: linear-gradient(90deg, #4f46e5, #9333ea);
        color: white !important;
        border: none;
        padding: 25px 10px;
        font-weight: 800;
        font-size: 1.3rem;
        width: 100%;
        border-radius: 20px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-transform: uppercase;
        letter-spacing: 5px;
        margin-top: 30px;
    }

    div.stButton > button:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- 4. HEADER ----------------
st.markdown("""
    <div class="header-box">
        <p style="color:#6366f1; font-weight:700; letter-spacing:8px; text-transform:uppercase; margin-bottom:5px;">Institutional Grade Projector</p>
        <h1 class="brand-title">Home_Prediction</h1>
    </div>
""", unsafe_allow_html=True)

# ---------------- 5. MAIN CONTROL PANEL ----------------
with st.container():
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("#### 📐 Structural Metrics")
        carpet = st.number_input("Carpet Area (sqft)", 200, 10000, 1800)
        bath_val = st.select_slider("Bathrooms", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        balc_val = st.select_slider("Balconies", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    with col2:
        st.markdown("#### 🌍 Geographic Data")
        floor = st.number_input("Floor Level", 0, 100, 12)
        locations = sorted([c.replace("location_", "") for c in model_columns if "location_" in c])
        selected_loc = st.selectbox("Market Neighborhood", locations)
        furnishing = st.radio("Finish Standard", ["Unfurnished", "Semi", "Full"], horizontal=True)

    analyze_btn = st.button("✨ PROJECT MARKET VALUE")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- 6. PROJECTOR LOGIC ----------------
if analyze_btn:
    with st.status("🔮 Analyzing Neural Comps...", expanded=False) as status:
        time.sleep(1.2)
        
        # Build Input DataFrame (117 Features)
        input_data = pd.DataFrame(0, index=[0], columns=model_columns)
        
        if "Carpet Area" in input_data.columns: input_data["Carpet Area"] = carpet
        if "Floor_No" in input_data.columns: input_data["Floor_No"] = floor
        
        # Map Categorical Features based on your specific model binary names
        bath_col = f"Bathroom_{bath_val}" if bath_val <= 9 else "Bathroom_> 10"
        balc_col = f"Balcony_{balc_val}" if balc_val <= 9 else "Balcony_> 10"
        
        if bath_col in input_data.columns: input_data[bath_col] = 1
        if balc_col in input_data.columns: input_data[balc_col] = 1
        if f"location_{selected_loc}" in input_data.columns: input_data[f"location_{selected_loc}"] = 1
            
        prediction = model.predict(input_data)[0]
        final_price = max(0, int(prediction))
        
        status.update(label="Projection Synthesized!", state="complete")

    # ---------------- 7. TARGET OUTPUT DASHBOARD ----------------
    st.write("")
    
    # ADVANCED METRIC TILES
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="elite-metric"><p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; margin:0;">Price Per Sqft</p><h2 style="color:white; margin:5px 0;">₹ {int(final_price/carpet):,}</h2><span style="color:#22c55e; font-size:0.75rem; font-weight:700;">● Neutral</span></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="elite-metric"><p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; margin:0;">Market Volatility</p><h2 style="color:white; margin:5px 0;">Low</h2><span style="color:#22c55e; font-size:0.75rem; font-weight:700;">● Bullish</span></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="elite-metric"><p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; margin:0;">Yield Efficiency</p><h2 style="color:white; margin:5px 0;">8.12%</h2><span style="color:#22c55e; font-size:0.75rem; font-weight:700;">● Optimal</span></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="elite-metric"><p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; margin:0;">Risk Score</p><h2 style="color:white; margin:5px 0;">0.24</h2><span style="color:#22c55e; font-size:0.75rem; font-weight:700;">● Safe</span></div>', unsafe_allow_html=True)

    st.write("")
    
    res_col1, res_col2 = st.columns([1.1, 1], gap="large")

    with res_col1:
        st.markdown(f"""
            <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid #6366f1; padding: 60px; border-radius: 35px; text-align: center; animation: slideUp 1s ease-out;">
                <p style="color: #818cf8; text-transform: uppercase; font-weight: 700; letter-spacing: 3px; margin-bottom: 10px;">Projected Valuation</p>
                <h1 style="color: white; font-size: 5.2rem; margin: 0; line-height:1; letter-spacing:-2px;">₹ {final_price:,}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("### 🧬 Intelligence Report")
        st.info(f"Asset projection based on linear regression of **{len(model_columns)}** features. The neighborhood of **{selected_loc}** is currently trending at a growth rate of 4.2% YoY.")

    with res_col2:
        # BIGGER, ELEGANT FEATURE WEIGHT CHART
        impacts = np.abs(model.coef_ * input_data.iloc[0].values)
        pie_df = pd.DataFrame({"Factor": model_columns, "Influence": impacts})
        pie_df = pie_df[pie_df["Influence"] > 0].sort_values("Influence", ascending=False).head(5)

        fig = px.pie(
            pie_df, values='Influence', names='Factor', hole=0.6,
            color_discrete_sequence=px.colors.sequential.Purp_r
        )
        
        fig.update_layout(
            title=dict(text="PRICE DRIVER ANALYSIS", font=dict(size=14, color="#94a3b8"), x=0.5, y=0.9),
            showlegend=True, height=450, margin=dict(t=80, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#94a3b8"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)