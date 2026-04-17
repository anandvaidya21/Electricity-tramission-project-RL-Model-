"""
⚡ Electricity Transmission Dashboard — Streamlit Frontend
A premium, real-time power grid simulation interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from environment.env import ElectricityTransmissionEnv, ActionModel
from environment.config import config
import time

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Electricity Transmission Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Root variables ── */
:root {
    --accent: #6C63FF;
    --accent-glow: rgba(108, 99, 255, 0.35);
    --success: #00E676;
    --warning: #FFD600;
    --danger: #FF5252;
    --card-bg: rgba(30, 30, 46, 0.7);
    --glass-border: rgba(255, 255, 255, 0.08);
}

/* ── Global ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%) !important;
    border-right: 1px solid var(--glass-border);
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px var(--accent-glow);
}
div[data-testid="stMetric"] label {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
}

/* ── Header banner ── */
.hero-header {
    background: linear-gradient(135deg, #6C63FF 0%, #3F3D56 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 40px rgba(108, 99, 255, 0.25);
    border: 1px solid rgba(255,255,255,0.1);
}
.hero-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #fff, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-header p {
    margin: 0.4rem 0 0 0;
    color: rgba(255,255,255,0.7);
    font-size: 1rem;
}

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.status-live { background: rgba(0,230,118,0.15); color: #00E676; border: 1px solid rgba(0,230,118,0.3); }
.status-idle { background: rgba(255,214,0,0.15); color: #FFD600; border: 1px solid rgba(255,214,0,0.3); }
.status-done { background: rgba(108,99,255,0.15); color: #6C63FF; border: 1px solid rgba(108,99,255,0.3); }

/* ── Section cards ── */
.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
}

/* ── Plotly chart containers ── */
.stPlotlyChart {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF 0%, #5a52e0 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108, 99, 255, 0.5) !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background-color: #6C63FF !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 20px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--glass-border);
    color: rgba(255,255,255,0.6);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6C63FF 0%, #5a52e0 100%) !important;
    color: white !important;
    border-color: transparent !important;
}

/* ── Dividers ── */
hr {
    border-color: var(--glass-border) !important;
}

/* ── Generator cards ── */
.gen-card {
    background: linear-gradient(135deg, rgba(30,30,46,0.8), rgba(30,30,46,0.5));
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}
.gen-card:hover {
    border-color: rgba(108,99,255,0.4);
    box-shadow: 0 0 20px var(--accent-glow);
}
</style>
""", unsafe_allow_html=True)

# ─── Plotly Theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="rgba(255,255,255,0.8)"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.1)",
        font=dict(size=11),
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
)

COLORS = {
    "accent": "#6C63FF",
    "cyan": "#00BCD4",
    "green": "#00E676",
    "amber": "#FFD600",
    "pink": "#FF4081",
    "orange": "#FF9100",
    "purple": "#AA00FF",
    "red": "#FF5252",
}

FUEL_COLORS = {
    "coal": "#78909C",
    "gas": "#FFB74D",
    "renewable": "#66BB6A",
    "nuclear": "#42A5F5",
}

FUEL_ICONS = {
    "coal": "🏭",
    "gas": "🔥",
    "renewable": "🌿",
    "nuclear": "⚛️",
}

# ─── Session State Initialisation ──────────────────────────────────────────────
def init_session_state():
    defaults = {
        "env": None,
        "running": False,
        "step_count": 0,
        "episode_done": False,
        "history": [],           # list of observation dicts
        "rewards": [],
        "cumulative_reward": 0.0,
        "gen_targets": {},
        "auto_run": False,
        "auto_speed": 0.3,
        "total_episodes": 0,
        "episode_rewards": [],   # total reward per episode
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ─── Environment helpers ───────────────────────────────────────────────────────
def create_env():
    env = ElectricityTransmissionEnv()
    st.session_state.env = env
    # Pre-set generator targets
    st.session_state.gen_targets = {
        gen.id: gen.max_output * 0.5 for gen in env.simulator.generators
    }
    return env


def reset_env():
    env = st.session_state.env
    if env is None:
        env = create_env()
    reset_resp, _ = env.reset()
    obs = reset_resp.observation.model_dump()
    st.session_state.history = [obs]
    st.session_state.rewards = []
    st.session_state.cumulative_reward = 0.0
    st.session_state.step_count = 0
    st.session_state.episode_done = False
    st.session_state.running = True
    return obs


def do_step():
    env = st.session_state.env
    if env is None or st.session_state.episode_done:
        return None
    action = ActionModel(generator_actions=st.session_state.gen_targets)
    step_resp = env.step(action)
    obs = step_resp.observation.model_dump()
    reward = step_resp.reward
    done = step_resp.terminated

    st.session_state.history.append(obs)
    st.session_state.rewards.append(reward)
    st.session_state.cumulative_reward += reward
    st.session_state.step_count += 1
    st.session_state.episode_done = done

    if done:
        st.session_state.total_episodes += 1
        st.session_state.episode_rewards.append(st.session_state.cumulative_reward)

    return obs


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚡ Control Panel")
    st.markdown("---")

    # — Environment controls —
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset", use_container_width=True, key="btn_reset"):
            reset_env()
    with col2:
        if st.button("▶ Step", use_container_width=True, key="btn_step",
                      disabled=st.session_state.episode_done or not st.session_state.running):
            do_step()

    st.markdown("")

    # — Auto-run toggle —
    st.session_state.auto_run = st.toggle("🤖  Auto-Run Agent", value=st.session_state.auto_run)
    st.session_state.auto_speed = st.slider(
        "Speed (steps/sec)", 1, 20, 5, key="speed_slider"
    )

    st.markdown("---")

    # — Generator Dispatch Controls —
    st.markdown("### 🔧 Generator Dispatch")

    if st.session_state.env is not None:
        generators = st.session_state.env.simulator.generators
        for gen in generators:
            icon = FUEL_ICONS.get(gen.fuel_type, "⚙️")
            label = f"{icon} Gen {gen.id} ({gen.fuel_type.title()})"
            st.session_state.gen_targets[gen.id] = st.slider(
                label,
                min_value=0.0,
                max_value=float(gen.max_output),
                value=float(st.session_state.gen_targets.get(gen.id, gen.max_output * 0.5)),
                step=1.0,
                key=f"gen_slider_{gen.id}",
            )
            # show cost
            cost = gen.cost_per_mwh
            st.caption(f"  Cost: ${cost}/MWh  ·  Ramp: {gen.ramp_rate} MW/min")
    else:
        st.info("Press **🔄 Reset** to initialize the environment.")

    st.markdown("---")

    # — Environment Info —
    st.markdown("### 📋 Environment Info")
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | Generators | `{config.NUM_GENERATORS}` |
    | Consumers | `{config.NUM_CONSUMERS}` |
    | Substations | `{config.NUM_SUBSTATIONS}` |
    | Episode Length | `{config.EPISODE_LENGTH} steps` |
    | Step Duration | `{config.SIMULATION_STEP_MINUTES} min` |
    | Max Gen Output | `{config.GENERATOR_MAX_OUTPUT} MW` |
    | Line Capacity | `{config.TRANSMISSION_LINE_CAPACITY} MW` |
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Hero Header ───────────────────────────────────────────────────────────────
if st.session_state.episode_done:
    badge = '<span class="status-badge status-done">EPISODE COMPLETE</span>'
elif st.session_state.running:
    badge = '<span class="status-badge status-live">● LIVE</span>'
else:
    badge = '<span class="status-badge status-idle">IDLE</span>'

st.markdown(f"""
<div class="hero-header">
    <h1>⚡ Electricity Transmission Dashboard</h1>
    <p>Real-time power grid simulation &amp; optimization &nbsp;&nbsp; {badge}</p>
</div>
""", unsafe_allow_html=True)


# ─── If no env, show welcome ──────────────────────────────────────────────────
if st.session_state.env is None or not st.session_state.running:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:3rem;">
        <h2 style="color:white; margin-bottom:0.5rem;">Welcome to the Grid Simulator</h2>
        <p style="color:rgba(255,255,255,0.6); font-size:1.1rem;">
            Press <strong>🔄 Reset</strong> in the sidebar to initialize the environment and start simulating.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Current Observation ──────────────────────────────────────────────────────
obs = st.session_state.history[-1] if st.session_state.history else {}

# ─── KPI Row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

prev_obs = st.session_state.history[-2] if len(st.session_state.history) >= 2 else None

def delta(key):
    if prev_obs and key in prev_obs:
        return round(obs.get(key, 0) - prev_obs[key], 2)
    return None

k1.metric("Generation", f"{obs.get('total_generation', 0):.1f} MW", delta("total_generation"))
k2.metric("Demand", f"{obs.get('total_demand', 0):.1f} MW", delta("total_demand"))
k3.metric("Demand Met", f"{obs.get('total_met', 0):.1f} MW", delta("total_met"))
k4.metric("Frequency", f"{obs.get('grid_frequency', 50):.2f} Hz", delta("grid_frequency"))
k5.metric("Voltage Stability", f"{obs.get('voltage_stability', 1):.3f}", delta("voltage_stability"))

last_reward = st.session_state.rewards[-1] if st.session_state.rewards else 0.0
k6.metric("Last Reward", f"{last_reward:.4f}",
          f"Σ {st.session_state.cumulative_reward:.2f}")

st.markdown("")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_generation, tab_priority, tab_agent, tab_history = st.tabs([
    "📊 Overview", "⚡ Generation", "🎯 Priority Analysis", "🤖 Agent Performance", "📜 History"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    if len(st.session_state.history) >= 2:
        df = pd.DataFrame(st.session_state.history)
        col_left, col_right = st.columns(2)

        with col_left:
            # Generation vs Demand
            fig_gd = go.Figure()
            fig_gd.add_trace(go.Scatter(
                y=df["total_generation"], mode="lines",
                name="Generation",
                line=dict(color=COLORS["green"], width=2),
                fill="tozeroy", fillcolor="rgba(0,230,118,0.08)",
            ))
            fig_gd.add_trace(go.Scatter(
                y=df["total_demand"], mode="lines",
                name="Demand",
                line=dict(color=COLORS["pink"], width=2, dash="dot"),
                fill="tozeroy", fillcolor="rgba(255,64,129,0.06)",
            ))
            fig_gd.add_trace(go.Scatter(
                y=df["total_met"], mode="lines",
                name="Met",
                line=dict(color=COLORS["cyan"], width=2),
            ))
            fig_gd.update_layout(**PLOTLY_LAYOUT, title="Generation vs Demand (MW)", height=380)
            st.plotly_chart(fig_gd, use_container_width=True)

        with col_right:
            # Grid Stability
            fig_stab = make_subplots(specs=[[{"secondary_y": True}]])
            fig_stab.add_trace(go.Scatter(
                y=df["grid_frequency"], mode="lines",
                name="Frequency (Hz)",
                line=dict(color=COLORS["amber"], width=2),
            ), secondary_y=False)
            fig_stab.add_trace(go.Scatter(
                y=df["voltage_stability"], mode="lines",
                name="Voltage Stability",
                line=dict(color=COLORS["purple"], width=2),
            ), secondary_y=True)
            # Target frequency line
            fig_stab.add_hline(y=50.0, line_dash="dash",
                               line_color="rgba(255,255,255,0.2)",
                               annotation_text="50 Hz Target",
                               annotation_font_color="rgba(255,255,255,0.4)")
            fig_stab.update_layout(**PLOTLY_LAYOUT, title="Grid Stability", height=380)
            fig_stab.update_yaxes(title_text="Frequency (Hz)", secondary_y=False)
            fig_stab.update_yaxes(title_text="Voltage Stability", secondary_y=True)
            st.plotly_chart(fig_stab, use_container_width=True)

        # Transmission & Cost
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=df["transmission_losses"], mode="lines+markers",
                name="Losses (MW)",
                line=dict(color=COLORS["orange"], width=2),
                marker=dict(size=3),
                fill="tozeroy", fillcolor="rgba(255,145,0,0.08)",
            ))
            fig_loss.update_layout(**PLOTLY_LAYOUT, title="Transmission Losses (MW)", height=320)
            st.plotly_chart(fig_loss, use_container_width=True)

        with col_r2:
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(
                y=df["total_cost"], mode="lines",
                name="Cost ($)",
                line=dict(color=COLORS["accent"], width=2),
                fill="tozeroy", fillcolor="rgba(108,99,255,0.1)",
            ))
            fig_cost.update_layout(**PLOTLY_LAYOUT, title="Generation Cost ($)", height=320)
            st.plotly_chart(fig_cost, use_container_width=True)
    else:
        st.info("▶ Step through the simulation to see real-time charts.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_generation:
    if st.session_state.env is not None:
        generators = st.session_state.env.simulator.generators

        # Generator status cards
        gen_cols = st.columns(len(generators))
        for i, gen in enumerate(generators):
            with gen_cols[i]:
                util_pct = (gen.current_output / gen.max_output * 100) if gen.max_output > 0 else 0
                color = COLORS["green"] if util_pct < 70 else (COLORS["amber"] if util_pct < 90 else COLORS["red"])
                icon = FUEL_ICONS.get(gen.fuel_type, "⚙️")
                st.markdown(f"""
                <div class="gen-card">
                    <div style="font-size:2.2rem;">{icon}</div>
                    <div style="color:white; font-weight:700; font-size:1.1rem; margin:0.3rem 0;">
                        Generator {gen.id}
                    </div>
                    <div style="color:rgba(255,255,255,0.5); font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;">
                        {gen.fuel_type}
                    </div>
                    <div style="color:{color}; font-size:1.6rem; font-weight:700; margin:0.5rem 0;">
                        {gen.current_output:.1f} MW
                    </div>
                    <div style="color:rgba(255,255,255,0.4); font-size:0.75rem;">
                        {util_pct:.0f}% of {gen.max_output} MW  ·  ${gen.cost_per_mwh}/MWh
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # Fuel mix pie chart
        col_pie, col_bar = st.columns(2)
        with col_pie:
            fuel_data = {}
            for gen in generators:
                fuel_data[gen.fuel_type] = fuel_data.get(gen.fuel_type, 0) + gen.current_output
            fig_pie = go.Figure(go.Pie(
                labels=list(fuel_data.keys()),
                values=list(fuel_data.values()),
                marker=dict(colors=[FUEL_COLORS.get(f, "#aaa") for f in fuel_data.keys()]),
                hole=0.55,
                textfont=dict(color="white"),
            ))
            fig_pie.update_layout(**PLOTLY_LAYOUT, title="Fuel Mix (Current Output)", height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            # Target vs actual output
            fig_bar = go.Figure()
            gen_ids = [f"Gen {g.id}" for g in generators]
            fig_bar.add_trace(go.Bar(
                x=gen_ids,
                y=[st.session_state.gen_targets.get(g.id, 0) for g in generators],
                name="Target",
                marker_color=COLORS["accent"],
                opacity=0.6,
            ))
            fig_bar.add_trace(go.Bar(
                x=gen_ids,
                y=[g.current_output for g in generators],
                name="Actual",
                marker_color=COLORS["green"],
            ))
            fig_bar.update_layout(**PLOTLY_LAYOUT, title="Target vs Actual Output", height=350,
                                  barmode="group")
            st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRIORITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_priority:
    if len(st.session_state.history) >= 2:
        df = pd.DataFrame(st.session_state.history)
        priority_cols = ["critical_demand_met", "high_demand_met", "medium_demand_met", "low_demand_met"]
        priority_labels = ["Critical", "High", "Medium", "Low"]
        priority_colors = [COLORS["red"], COLORS["orange"], COLORS["amber"], COLORS["green"]]

        # Current priority satisfaction gauge
        st.markdown("#### Current Priority Satisfaction")
        gauge_cols = st.columns(4)
        for i, (col_name, label, color) in enumerate(zip(priority_cols, priority_labels, priority_colors)):
            with gauge_cols[i]:
                val = obs.get(col_name, 0)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=val,
                    number=dict(suffix="%", font=dict(size=28, color="white")),
                    title=dict(text=label, font=dict(size=14, color="rgba(255,255,255,0.6)")),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="rgba(255,255,255,0.3)"),
                        bar=dict(color=color),
                        bgcolor="rgba(255,255,255,0.05)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 50], color="rgba(255,82,82,0.1)"),
                            dict(range=[50, 80], color="rgba(255,214,0,0.1)"),
                            dict(range=[80, 100], color="rgba(0,230,118,0.1)"),
                        ],
                        threshold=dict(
                            line=dict(color="white", width=2),
                            thickness=0.75,
                            value=95,
                        ),
                    ),
                ))
                gauge_layout = {**PLOTLY_LAYOUT, "height": 220, "margin": dict(l=20, r=20, t=40, b=10)}
                fig_gauge.update_layout(**gauge_layout)
                st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("")

        # Priority over time
        fig_pri = go.Figure()
        for col_name, label, color in zip(priority_cols, priority_labels, priority_colors):
            fig_pri.add_trace(go.Scatter(
                y=df[col_name], mode="lines",
                name=label,
                line=dict(color=color, width=2),
            ))
        fig_pri.add_hline(y=95, line_dash="dash", line_color="rgba(255,255,255,0.15)",
                          annotation_text="95% Target")
        fig_pri.update_layout(**PLOTLY_LAYOUT, title="Priority Demand Met Over Time (%)", height=380)
        st.plotly_chart(fig_pri, use_container_width=True)
    else:
        st.info("▶ Step through the simulation to see priority analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AGENT PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_agent:
    if st.session_state.rewards:
        col_rew, col_cum = st.columns(2)

        with col_rew:
            fig_rew = go.Figure()
            fig_rew.add_trace(go.Scatter(
                y=st.session_state.rewards, mode="lines",
                name="Step Reward",
                line=dict(color=COLORS["cyan"], width=2),
                fill="tozeroy", fillcolor="rgba(0,188,212,0.08)",
            ))
            fig_rew.update_layout(**PLOTLY_LAYOUT, title="Step Reward", height=350)
            st.plotly_chart(fig_rew, use_container_width=True)

        with col_cum:
            cumulative = np.cumsum(st.session_state.rewards).tolist()
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                y=cumulative, mode="lines",
                name="Cumulative Reward",
                line=dict(color=COLORS["accent"], width=2.5),
                fill="tozeroy", fillcolor="rgba(108,99,255,0.1)",
            ))
            fig_cum.update_layout(**PLOTLY_LAYOUT, title="Cumulative Reward", height=350)
            st.plotly_chart(fig_cum, use_container_width=True)

        # Summary stats
        st.markdown("#### 📈 Session Statistics")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Steps", st.session_state.step_count)
        s2.metric("Episodes", st.session_state.total_episodes)
        s3.metric("Avg Reward", f"{np.mean(st.session_state.rewards):.4f}")
        s4.metric("Max Reward", f"{np.max(st.session_state.rewards):.4f}")
        s5.metric("Min Reward", f"{np.min(st.session_state.rewards):.4f}")

        # Reward distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=st.session_state.rewards,
            nbinsx=30,
            marker_color=COLORS["accent"],
            opacity=0.8,
        ))
        fig_dist.update_layout(**PLOTLY_LAYOUT, title="Reward Distribution", height=300)
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("▶ Step through the simulation to see agent performance.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HISTORY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_history:
    if len(st.session_state.history) >= 2:
        df_hist = pd.DataFrame(st.session_state.history)
        # Add reward column
        if st.session_state.rewards:
            df_hist["reward"] = [0.0] + st.session_state.rewards
        st.dataframe(
            df_hist.style.format({
                "total_generation": "{:.2f}",
                "total_demand": "{:.2f}",
                "total_met": "{:.2f}",
                "grid_frequency": "{:.3f}",
                "voltage_stability": "{:.4f}",
                "transmission_losses": "{:.4f}",
                "total_cost": "{:.2f}",
                "critical_demand_met": "{:.1f}",
                "high_demand_met": "{:.1f}",
                "medium_demand_met": "{:.1f}",
                "low_demand_met": "{:.1f}",
            }).background_gradient(subset=["reward"] if "reward" in df_hist.columns else [], cmap="RdYlGn"),
            use_container_width=True,
            height=500,
        )

        # Download button
        csv = df_hist.to_csv(index=False)
        st.download_button(
            "📥 Download History as CSV",
            csv,
            "grid_simulation_history.csv",
            "text/csv",
        )
    else:
        st.info("▶ Step through the simulation to see history data.")


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-RUN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.auto_run and st.session_state.running and not st.session_state.episode_done:
    do_step()
    time.sleep(1.0 / st.session_state.auto_speed)
    st.rerun()
