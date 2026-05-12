import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import random
import time

warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NetGuard — Network Anomaly Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@400;600;700&display=swap');

:root {
    --bg-base:       #060a10;
    --bg-panel:      #0b1120;
    --bg-card:       #0f1828;
    --bg-hover:      #131e30;
    --border:        #1a2d45;
    --border-bright: #1e3d5c;
    --green:         #00e5a0;
    --green-dim:     #00b87d;
    --green-glow:    rgba(0,229,160,0.15);
    --red:           #ff3c5a;
    --red-dim:       #cc2040;
    --red-glow:      rgba(255,60,90,0.15);
    --orange:        #ff8c00;
    --blue:          #38b6ff;
    --purple:        #b44dff;
    --yellow:        #ffe033;
    --text-primary:  #d4e0f0;
    --text-secondary:#7a95b8;
    --text-muted:    #3d5570;
}

.stApp {
    background: var(--bg-base) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,100,180,0.08), transparent),
        radial-gradient(ellipse 50% 30% at 90% 80%, rgba(0,229,160,0.04), transparent);
}

section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.5);
}
section[data-testid="stSidebar"] * { color: var(--text-primary); }

div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 18px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover {
    border-color: var(--border-bright) !important;
    box-shadow: 0 0 20px rgba(0,229,160,0.07);
}
div[data-testid="metric-container"] > div > div > div {
    color: var(--green) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    padding: 12px 20px !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.2s, border-color 0.2s;
}
.stTabs [aria-selected="true"] {
    color: var(--green) !important;
    border-bottom: 2px solid var(--green) !important;
    background: rgba(0,229,160,0.04) !important;
}

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--green) !important;
    color: var(--green) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em;
    border-radius: 6px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--green-glow) !important;
    box-shadow: 0 0 16px rgba(0,229,160,0.3) !important;
    transform: translateY(-1px) !important;
}

.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 6px !important;
}
.stSlider > div > div > div > div { background: var(--green) !important; }

hr { border-color: var(--border) !important; margin: 20px 0 !important; }

.ng-header {
    font-family: 'Orbitron', monospace;
    color: var(--green);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 10px;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--border);
}
.ng-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

.alert-critical {
    background: linear-gradient(135deg, rgba(255,60,90,0.12), rgba(180,20,40,0.06));
    border: 1px solid rgba(255,60,90,0.4);
    border-left: 3px solid var(--red);
    padding: 14px 18px; border-radius: 8px; margin: 8px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #ff7a8a;
    animation: pulse-red 2s infinite;
}
.alert-high {
    background: linear-gradient(135deg, rgba(255,140,0,0.10), rgba(180,90,0,0.05));
    border: 1px solid rgba(255,140,0,0.35);
    border-left: 3px solid var(--orange);
    padding: 14px 18px; border-radius: 8px; margin: 8px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #ffa040;
}
.alert-normal {
    background: linear-gradient(135deg, rgba(0,229,160,0.07), rgba(0,160,110,0.03));
    border: 1px solid rgba(0,229,160,0.2);
    border-left: 3px solid var(--green);
    padding: 14px 18px; border-radius: 8px; margin: 8px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--green);
}

.feed-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px; margin: 6px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    display: flex; align-items: center; gap: 12px;
    transition: border-color 0.2s;
}
.feed-item:hover { border-color: var(--border-bright); }

.badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    font-weight: 500; letter-spacing: 0.08em;
}
.badge-critical { background: rgba(255,60,90,0.2); color: var(--red); border: 1px solid rgba(255,60,90,0.4); }
.badge-high     { background: rgba(255,140,0,0.2); color: var(--orange); border: 1px solid rgba(255,140,0,0.4); }
.badge-medium   { background: rgba(255,224,51,0.15); color: var(--yellow); border: 1px solid rgba(255,224,51,0.3); }

.detect-result-safe {
    background: linear-gradient(135deg, rgba(0,229,160,0.08), rgba(0,160,110,0.04));
    border: 1px solid rgba(0,229,160,0.35); border-radius: 12px; padding: 28px; text-align: center;
}
.detect-result-danger {
    background: linear-gradient(135deg, rgba(255,60,90,0.1), rgba(180,20,40,0.05));
    border: 1px solid rgba(255,60,90,0.4); border-radius: 12px; padding: 28px; text-align: center;
    animation: pulse-red 2s infinite;
}

.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); animation: blink 2s infinite; box-shadow: 0 0 6px var(--green);
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 rgba(255,60,90,0); }
    50%       { box-shadow: 0 0 20px rgba(255,60,90,0.2); }
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── PLOT THEME ────────────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor='#0b1120', plot_bgcolor='#060a10',
    font=dict(color='#7a95b8', family='JetBrains Mono'),
    xaxis=dict(gridcolor='#1a2d45', color='#7a95b8', linecolor='#1a2d45', zerolinecolor='#1a2d45'),
    yaxis=dict(gridcolor='#1a2d45', color='#7a95b8', linecolor='#1a2d45', zerolinecolor='#1a2d45'),
    margin=dict(t=50, b=40, l=40, r=20),
)
COLORS = {
    'BENIGN': '#00e5a0', 'DDoS': '#ff3c5a',
    'PortScan': '#ff8c00', 'BruteForce': '#b44dff', 'Bot': '#38b6ff',
}

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:24px 0 16px;'>
        <div style='font-size:3rem; filter:drop-shadow(0 0 12px rgba(0,229,160,0.6));'>🛡️</div>
        <div style='font-family:Orbitron,monospace; color:#00e5a0; font-size:1.2rem;
                    font-weight:700; letter-spacing:0.2em; margin-top:10px;'>NETGUARD</div>
        <div style='font-family:JetBrains Mono,monospace; color:#3d5570; font-size:0.65rem;
                    letter-spacing:0.18em; margin-top:4px;'>NETWORK ANOMALY DETECTOR</div>
        <div style='margin-top:12px;'>
            <span class='status-dot'></span>
            <span style='font-family:JetBrains Mono,monospace; color:#7a95b8;
                         font-size:0.7rem; margin-left:8px;'>SYSTEM ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="ng-header">📡 Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio("", ["Use Demo Dataset", "Upload CSV File"], label_visibility="collapsed")
    if data_source == "Upload CSV File":
        uploaded_csv = st.file_uploader("Upload network traffic CSV", type=["csv"])
        st.markdown("""
        <div style='font-family:JetBrains Mono,monospace; font-size:0.68rem; color:#3d5570; margin-top:8px; line-height:1.6;'>
        Requires a <span style='color:#7a95b8'>label</span> column with values:<br>
        BENIGN · DDoS · PortScan · BruteForce · Bot
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded_csv = None
    st.markdown("---")
    st.markdown('<div class="ng-header">⚙️ Model Config</div>', unsafe_allow_html=True)
    n_trees   = st.slider("Decision Trees", 50, 300, 100, 50)
    test_size = st.slider("Test Split %", 10, 40, 20, 5)
    max_depth = st.slider("Max Tree Depth", 5, 30, 15, 5)
    st.markdown("---")
    st.markdown('<div class="ng-header">🎛️ Display</div>', unsafe_allow_html=True)
    show_raw     = st.checkbox("Show raw feature values", value=False)
    auto_refresh = st.checkbox("Simulate live feed updates", value=False)
    st.markdown("---")
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.65rem; color:#3d5570; text-align:center; line-height:2;'>
        v3.0 · Random Forest Classifier<br>CICIDS2017 Dataset Structure<br>
        <span style='color:#7a95b8'>Built with Streamlit + Plotly</span>
    </div>
    """, unsafe_allow_html=True)

# ─── DATA GENERATION ───────────────────────────────────────────────────────────
@st.cache_data
def generate_demo_dataset():
    np.random.seed(42)
    samples = {'BENIGN': 3000, 'DDoS': 800, 'PortScan': 400, 'BruteForce': 300, 'Bot': 200}
    patterns = {
        'BENIGN':     dict(dur=(2,1),    pkt=(500,100),    bps=(5000,2000),    pps=(10,5),    fwd=(20,5),   bwd=(18,5),  syn=(1,0.5),   rst=(0,0.01)),
        'DDoS':       dict(dur=(0.1,.05),pkt=(60,10),      bps=(500000,100000),pps=(1000,200),fwd=(500,100),bwd=(2,1),   syn=(500,100), rst=(100,20)),
        'PortScan':   dict(dur=(0.5,.2), pkt=(40,5),       bps=(1000,500),     pps=(50,10),   fwd=(1,.5),   bwd=(1,.5),  syn=(1,.5),    rst=(1,.5)),
        'BruteForce': dict(dur=(5,2),    pkt=(200,50),     bps=(2000,500),     pps=(20,5),    fwd=(30,10),  bwd=(30,10), syn=(1,.5),    rst=(5,2)),
        'Bot':        dict(dur=(10,3),   pkt=(300,50),     bps=(3000,500),     pps=(15,3),    fwd=(25,8),   bwd=(25,8),  syn=(1,.5),    rst=(0,0.01)),
    }
    dfs = []
    for label, n in samples.items():
        p = patterns[label]
        df = pd.DataFrame({
            'duration':               abs(np.random.normal(p['dur'][0], p['dur'][1], n)),
            'packet_length_mean':     abs(np.random.normal(p['pkt'][0], p['pkt'][1], n)),
            'packet_length_std':      abs(np.random.normal(p['pkt'][1], p['pkt'][1]/2, n)),
            'flow_bytes_per_sec':     abs(np.random.normal(p['bps'][0], p['bps'][1], n)),
            'flow_packets_per_sec':   abs(np.random.normal(p['pps'][0], p['pps'][1], n)),
            'fwd_packets':            abs(np.random.normal(p['fwd'][0], p['fwd'][1], n)),
            'bwd_packets':            abs(np.random.normal(p['bwd'][0], p['bwd'][1], n)),
            'fwd_packet_length_mean': abs(np.random.normal(p['pkt'][0]*0.8, p['pkt'][1], n)),
            'bwd_packet_length_mean': abs(np.random.normal(p['pkt'][0]*1.2, p['pkt'][1], n)),
            'syn_flag_count':         abs(np.random.normal(p['syn'][0], p['syn'][1], n)),
            'rst_flag_count':         abs(np.random.normal(p['rst'][0], p['rst'][1], n)),
            'fin_flag_count':         abs(np.random.poisson(1, n).astype(float)),
            'idle_mean':              abs(np.random.exponential(1, n)),
            'active_mean':            abs(np.random.exponential(3, n)),
            'label': label
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

@st.cache_resource
def train_model(_df, n_estimators, test_split, depth):
    feature_cols = [c for c in _df.columns if c != 'label']
    X = _df[feature_cols]
    le = LabelEncoder()
    y = le.fit_transform(_df['label'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split/100, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=depth,
        random_state=42, n_jobs=-1, class_weight='balanced'
    )
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec   = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1    = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    fi = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    cm = confusion_matrix(y_test, y_pred)
    return model, le, acc, prec, rec, f1, fi, X_test, y_test, y_pred, y_proba, cm, feature_cols

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("Initializing NetGuard detection engine..."):
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            col_map = {c: c.strip().lower().replace(' ', '_') for c in df.columns}
            df = df.rename(columns=col_map)
            if 'label' not in df.columns:
                possible = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]
                if possible:
                    df = df.rename(columns={possible[0]: 'label'})
                else:
                    st.error("No 'label' column found. Using demo dataset.")
                    df = generate_demo_dataset()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            df[numeric_cols] = df[numeric_cols].abs()
            df['label'] = df['label'].str.strip()
            st.success(f"✅ Loaded {len(df):,} records · {df['label'].nunique()} classes detected")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            df = generate_demo_dataset()
    else:
        df = generate_demo_dataset()

    model, le, accuracy, precision, recall, f1, fi, X_test, y_test, y_pred, y_proba, cm, feature_cols = \
        train_model(df, n_trees, test_size, max_depth)

label_counts = df['label'].value_counts()
total      = len(df)
benign     = int(label_counts.get('BENIGN', label_counts.get('benign', 0)))
attacks    = total - benign
threat_pct = attacks / total * 100

# ─── HEADER ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <div style='padding:8px 0 4px;'>
        <div style='display:flex; align-items:center; gap:14px;'>
            <span style='font-size:2.2rem; filter:drop-shadow(0 0 14px rgba(0,229,160,0.7));'>🛡️</span>
            <div>
                <div style='font-family:Orbitron,monospace; color:#00e5a0; font-size:1.6rem;
                            font-weight:700; letter-spacing:0.06em; line-height:1.1;'>
                    NetGuard
                    <span style='font-size:0.85rem; color:#3d5570; font-weight:400;'>— Network Anomaly Detector</span>
                </div>
                <div style='font-family:JetBrains Mono,monospace; color:#3d5570; font-size:0.72rem;
                            letter-spacing:0.1em; margin-top:3px;'>
                    ML-POWERED INTRUSION DETECTION · RANDOM FOREST CLASSIFIER · CICIDS2017
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    now = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style='text-align:right; font-family:JetBrains Mono,monospace; padding-top:14px; line-height:2;'>
        <div style='font-size:0.68rem; color:#3d5570; letter-spacing:0.12em;'>STATUS</div>
        <div style='font-size:0.82rem; color:#00e5a0;'><span class='status-dot'></span> ACTIVE</div>
        <div style='font-size:0.68rem; color:#3d5570; margin-top:4px;'>{now} UTC</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── THREAT BANNER ─────────────────────────────────────────────────────────────
if threat_pct > 40:
    st.markdown(f'<div class="alert-critical">🚨 <strong>CRITICAL THREAT LEVEL</strong> — {threat_pct:.1f}% of traffic classified as malicious · Immediate incident response recommended <span style="float:right">● ELEVATED RISK</span></div>', unsafe_allow_html=True)
elif threat_pct > 20:
    st.markdown(f'<div class="alert-high">⚠️ <strong>HIGH THREAT LEVEL</strong> — {threat_pct:.1f}% of traffic flagged as suspicious · Review live feed <span style="float:right">● MONITORING</span></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="alert-normal">✅ <strong>NORMAL OPERATIONS</strong> — {threat_pct:.1f}% threat level — System nominal <span style="float:right">● NOMINAL</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "[ OVERVIEW ]", "[ LIVE DETECTOR ]", "[ MODEL INTEL ]",
    "[ TRAFFIC ANALYSIS ]", "[ TIMELINE ]", "[ ALERTS LOG ]",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("TOTAL CONNECTIONS", f"{total:,}",   delta=f"+{int(total*0.02):,} /hr")
    c2.metric("NORMAL TRAFFIC",    f"{benign:,}",  delta=f"+{int(benign*0.015):,} /hr")
    c3.metric("ATTACKS DETECTED",  f"{attacks:,}", delta=f"+{random.randint(5,30)} /min", delta_color="inverse")
    c4.metric("THREAT LEVEL",      f"{threat_pct:.1f}%")
    c5.metric("MODEL ACCURACY",    f"{accuracy:.1%}")
    c6.metric("F1 SCORE",          f"{f1:.3f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        color_list = [COLORS.get(l, '#888') for l in label_counts.index]
        fig_pie = go.Figure(go.Pie(
            labels=label_counts.index, values=label_counts.values, hole=0.65,
            marker=dict(colors=color_list, line=dict(color='#060a10', width=2)),
            textfont=dict(family='JetBrains Mono', size=11),
        ))
        fig_pie.update_layout(
            title=dict(text="Traffic Distribution", font=dict(color='#00e5a0', family='Orbitron', size=13)),
            legend=dict(bgcolor='#0b1120', bordercolor='#1a2d45', borderwidth=1, font=dict(family='JetBrains Mono', size=11)),
            annotations=[dict(text=f'<b>{threat_pct:.0f}%</b><br>threats', x=0.5, y=0.5, showarrow=False,
                              font=dict(size=15, color='#ff3c5a', family='Orbitron'))],
            **{k:v for k,v in DARK_LAYOUT.items() if k not in ['xaxis','yaxis']}, height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        attack_only = label_counts[label_counts.index != 'BENIGN']
        if len(attack_only):
            fig_bar = go.Figure(go.Bar(
                x=attack_only.values, y=attack_only.index, orientation='h',
                marker=dict(color=[COLORS.get(l,'#888') for l in attack_only.index], opacity=0.85,
                            line=dict(color='#1a2d45', width=1)),
                text=attack_only.values, textposition='outside',
                textfont=dict(family='JetBrains Mono', size=11, color='#7a95b8'),
            ))
            fig_bar.update_layout(
                title=dict(text="Attack Vector Frequency", font=dict(color='#00e5a0', family='Orbitron', size=13)),
                **DARK_LAYOUT, height=320,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="ng-header">🔴 Live Threat Feed</div>', unsafe_allow_html=True)
    attack_types = [l for l in label_counts.index if l.upper() != 'BENIGN']
    if attack_types:
        rng = random.Random(int(time.time() // 30) if auto_refresh else 42)
        protocols = ['TCP','UDP','ICMP','HTTP','HTTPS','DNS','FTP','SSH']
        countries = ['CN','RU','US','DE','BR','KR','UA','NL']
        for i in range(10):
            attack  = rng.choice(attack_types)
            src_ip  = f"{rng.randint(1,223)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
            dst_ip  = f"10.0.{rng.randint(1,10)}.{rng.randint(1,50)}"
            port    = rng.choice([80,443,22,21,3389,8080,53,445])
            proto   = rng.choice(protocols)
            country = rng.choice(countries)
            ts      = (datetime.now() - timedelta(seconds=rng.randint(0,300))).strftime("%H:%M:%S")
            if attack.upper() == 'DDOS':
                sev, badge_cls, color = "CRITICAL", "badge-critical", "#ff3c5a"
            elif attack.upper() in ['BRUTEFORCE','BOT']:
                sev, badge_cls, color = "HIGH", "badge-high", "#ff8c00"
            else:
                sev, badge_cls, color = "MEDIUM", "badge-medium", "#ffe033"
            st.markdown(f"""
            <div class='feed-item' style='border-left:3px solid {color};'>
                <span class='badge {badge_cls}'>{sev}</span>
                <span style='color:#d4e0f0; font-weight:500;'>{attack}</span>
                <span style='color:#3d5570;'>·</span>
                <span style='color:#7a95b8;'>{src_ip}</span>
                <span style='color:#3d5570;'>→</span>
                <span style='color:#d4e0f0;'>{dst_ip}:{port}</span>
                <span style='color:#3d5570;'>·</span>
                <span style='color:#38b6ff; font-size:0.72rem;'>{proto}</span>
                <span style='color:#3d5570;'>·</span>
                <span style='color:#3d5570; font-size:0.72rem;'>[{country}]</span>
                <span style='margin-left:auto; color:#3d5570; font-size:0.72rem;'>{ts}</span>
                <span style='background:rgba(255,60,90,0.15); color:#ff3c5a; padding:2px 8px; border-radius:4px; font-size:0.68rem;'>BLOCKED</span>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="ng-header">🔍 Real-Time Traffic Classifier</div>', unsafe_allow_html=True)

    st.markdown("**Quick Scenarios:**")
    preset_cols = st.columns(6)
    presets = {
        "Normal":     dict(duration=2.0, packet_length_mean=500, packet_length_std=100, flow_bytes_per_sec=5000, flow_packets_per_sec=10, fwd_packets=20, bwd_packets=18, fwd_packet_length_mean=400, bwd_packet_length_mean=600, syn_flag_count=1, rst_flag_count=0, fin_flag_count=1, idle_mean=1.0, active_mean=3.0),
        "DDoS":       dict(duration=0.1, packet_length_mean=60,  packet_length_std=10,  flow_bytes_per_sec=500000, flow_packets_per_sec=1000, fwd_packets=500, bwd_packets=2, fwd_packet_length_mean=48, bwd_packet_length_mean=72, syn_flag_count=500, rst_flag_count=100, fin_flag_count=0, idle_mean=0.01, active_mean=0.1),
        "PortScan":   dict(duration=0.5, packet_length_mean=40,  packet_length_std=5,   flow_bytes_per_sec=1000, flow_packets_per_sec=50, fwd_packets=1, bwd_packets=1, fwd_packet_length_mean=32, bwd_packet_length_mean=48, syn_flag_count=1, rst_flag_count=1, fin_flag_count=0, idle_mean=0.5, active_mean=0.5),
        "BruteForce": dict(duration=5.0, packet_length_mean=200, packet_length_std=50,  flow_bytes_per_sec=2000, flow_packets_per_sec=20, fwd_packets=30, bwd_packets=30, fwd_packet_length_mean=160, bwd_packet_length_mean=240, syn_flag_count=1, rst_flag_count=5, fin_flag_count=1, idle_mean=2.0, active_mean=3.0),
        "Bot":        dict(duration=10.0,packet_length_mean=300, packet_length_std=50,  flow_bytes_per_sec=3000, flow_packets_per_sec=15, fwd_packets=25, bwd_packets=25, fwd_packet_length_mean=240, bwd_packet_length_mean=360, syn_flag_count=1, rst_flag_count=0, fin_flag_count=1, idle_mean=1.0, active_mean=3.0),
        "Random":     None,
    }
    if 'preset_inputs' not in st.session_state:
        st.session_state.preset_inputs = presets['Normal'].copy()

    for idx, (name, vals) in enumerate(presets.items()):
        with preset_cols[idx]:
            if st.button(name, key=f"preset_{name}"):
                if vals is None:
                    st.session_state.preset_inputs = {k: random.uniform(mn, mx) for k,(_, mn, mx) in {
                        'duration':(None,0,20),'packet_length_mean':(None,0,1500),
                        'packet_length_std':(None,0,500),'flow_bytes_per_sec':(None,0,1000000),
                        'flow_packets_per_sec':(None,0,2000),'fwd_packets':(None,0,1000),
                        'bwd_packets':(None,0,1000),'fwd_packet_length_mean':(None,0,1500),
                        'bwd_packet_length_mean':(None,0,1500),'syn_flag_count':(None,0,1000),
                        'rst_flag_count':(None,0,500),'fin_flag_count':(None,0,10),
                        'idle_mean':(None,0.0,10.0),'active_mean':(None,0.0,10.0),
                    }.items()}
                else:
                    st.session_state.preset_inputs = vals.copy()
                st.rerun()

    st.markdown("---")
    slider_config = {
        'duration':               ("Connection Duration (s)", 0.0, 20.0),
        'packet_length_mean':     ("Mean Packet Length (bytes)", 0, 1500),
        'packet_length_std':      ("Packet Length Std Dev", 0, 500),
        'flow_bytes_per_sec':     ("Flow Bytes / sec", 0, 1000000),
        'flow_packets_per_sec':   ("Flow Packets / sec", 0, 2000),
        'fwd_packets':            ("Forward Packets", 0, 1000),
        'bwd_packets':            ("Backward Packets", 0, 1000),
        'fwd_packet_length_mean': ("Fwd Packet Length Mean", 0, 1500),
        'bwd_packet_length_mean': ("Bwd Packet Length Mean", 0, 1500),
        'syn_flag_count':         ("SYN Flag Count", 0, 1000),
        'rst_flag_count':         ("RST Flag Count", 0, 500),
        'fin_flag_count':         ("FIN Flag Count", 0, 10),
        'idle_mean':              ("Idle Mean", 0.0, 10.0),
        'active_mean':            ("Active Mean", 0.0, 10.0),
    }
    inputs = {}
    keys = list(slider_config.keys())
    col_a, col_b = st.columns(2)
    with col_a:
        for k in keys[:7]:
            label, mn, mx = slider_config[k]
            default_val = float(np.clip(st.session_state.preset_inputs.get(k, (mn+mx)/2), mn, mx))
            inputs[k] = st.slider(label, float(mn), float(mx), default_val, key=f"slider_{k}")
    with col_b:
        for k in keys[7:]:
            label, mn, mx = slider_config[k]
            default_val = float(np.clip(st.session_state.preset_inputs.get(k, (mn+mx)/2), mn, mx))
            inputs[k] = st.slider(label, float(mn), float(mx), default_val, key=f"slider_{k}")

    input_df   = pd.DataFrame([[inputs[f] for f in feature_cols]], columns=feature_cols)
    pred       = model.predict(input_df)[0]
    probs      = model.predict_proba(input_df)[0]
    pred_label = le.inverse_transform([pred])[0]
    confidence = float(max(probs)) * 100

    st.markdown("---")
    if pred_label.upper() == 'BENIGN':
        st.markdown(f"""
        <div class='detect-result-safe'>
            <div style='font-family:Orbitron,monospace; font-size:1.6rem; color:#00e5a0; letter-spacing:0.08em;'>✅ BENIGN TRAFFIC</div>
            <div style='font-family:JetBrains Mono,monospace; color:#7a95b8; margin-top:10px; font-size:0.85rem;'>
                Confidence: <span style='color:#00e5a0;'>{confidence:.1f}%</span> · No threat signatures detected · Connection allowed
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        threat_color = "#ff3c5a" if pred_label.upper() in ['DDOS','BRUTEFORCE'] else "#ff8c00"
        st.markdown(f"""
        <div class='detect-result-danger'>
            <div style='font-family:Orbitron,monospace; font-size:1.6rem; color:{threat_color}; letter-spacing:0.08em;'>🚨 ATTACK DETECTED: {pred_label.upper()}</div>
            <div style='font-family:JetBrains Mono,monospace; color:#7a95b8; margin-top:10px; font-size:0.85rem;'>
                Confidence: <span style='color:{threat_color};'>{confidence:.1f}%</span> · Threat signature matched · Connection blocked
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    prob_df = pd.DataFrame({'Type': le.classes_, 'Probability': probs}).sort_values('Probability', ascending=True)
    fig_prob = go.Figure(go.Bar(
        x=prob_df['Probability'], y=prob_df['Type'], orientation='h',
        marker=dict(color=[COLORS.get(t,'#888') for t in prob_df['Type']], opacity=0.85, line=dict(color='#1a2d45', width=1)),
        text=[f"{p:.1%}" for p in prob_df['Probability']], textposition='outside',
        textfont=dict(family='JetBrains Mono', size=11),
    ))
    fig_prob.update_layout(
        title=dict(text="Classification Confidence Breakdown", font=dict(color='#00e5a0', family='Orbitron', size=13)),
        xaxis=dict(gridcolor='#1a2d45', color='#7a95b8', tickformat='.0%', range=[0,1.1]),
        yaxis=dict(gridcolor='#1a2d45', color='#7a95b8'),
        **{k:v for k,v in DARK_LAYOUT.items() if k not in ['xaxis','yaxis']}, height=280,
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    if show_raw:
        st.markdown('<div class="ng-header">📋 Raw Feature Values</div>', unsafe_allow_html=True)
        st.dataframe(input_df.T.rename(columns={0:'Value'}), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INTEL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="ng-header">🧠 Model Intelligence Report</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("ACCURACY",       f"{accuracy:.1%}")
    c2.metric("PRECISION",      f"{precision:.1%}")
    c3.metric("RECALL",         f"{recall:.1%}")
    c4.metric("F1 SCORE",       f"{f1:.3f}")
    c5.metric("TEST SAMPLES",   f"{len(y_test):,}")
    c6.metric("DECISION TREES", f"{n_trees}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        labels  = le.classes_
        cm_pct  = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        text_lbl= [[f"{cm[i,j]}<br>({cm_pct[i,j]:.0%})" for j in range(len(labels))] for i in range(len(labels))]
        fig_cm  = go.Figure(go.Heatmap(
            z=cm_pct, x=labels, y=labels,
            colorscale=[[0,'#060a10'],[0.4,'#003a25'],[0.7,'#006644'],[1,'#00e5a0']],
            text=text_lbl, texttemplate="%{text}",
            textfont=dict(color='white', family='JetBrains Mono', size=10),
        ))
        fig_cm.update_layout(
            title=dict(text="Confusion Matrix (row-normalized)", font=dict(color='#00e5a0', family='Orbitron', size=13)),
            xaxis=dict(title='Predicted', color='#7a95b8', gridcolor='#1a2d45'),
            yaxis=dict(title='Actual',    color='#7a95b8', gridcolor='#1a2d45'),
            **{k:v for k,v in DARK_LAYOUT.items() if k not in ['xaxis','yaxis']}, height=380,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_r:
        fig_fi = go.Figure(go.Bar(
            x=fi.head(12)['importance'], y=fi.head(12)['feature'], orientation='h',
            marker=dict(color=fi.head(12)['importance'],
                        colorscale=[[0,'#003a25'],[0.5,'#00b87d'],[1,'#00e5a0']],
                        line=dict(color='#1a2d45', width=1), showscale=False),
            text=[f"{v:.3f}" for v in fi.head(12)['importance']], textposition='outside',
            textfont=dict(family='JetBrains Mono', size=10),
        ))
        fig_fi.update_layout(
            title=dict(text="Top Feature Importances", font=dict(color='#00e5a0', family='Orbitron', size=13)),
            xaxis=dict(gridcolor='#1a2d45', color='#7a95b8'),
            yaxis=dict(gridcolor='#1a2d45', color='#7a95b8', categoryorder='total ascending'),
            **{k:v for k,v in DARK_LAYOUT.items() if k not in ['xaxis','yaxis']}, height=380,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    report    = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_df = report_df.drop(['accuracy','macro avg','weighted avg'], errors='ignore')
    report_df = report_df[['precision','recall','f1-score','support']].round(3)
    report_df.columns = ['Precision','Recall','F1-Score','Support']
    report_df['Support'] = report_df['Support'].astype(int)

    fig_rep = go.Figure()
    for m, mc in zip(['Precision','Recall','F1-Score'], ['#38b6ff','#b44dff','#00e5a0']):
        fig_rep.add_trace(go.Bar(
            name=m, x=report_df.index, y=report_df[m], marker_color=mc, opacity=0.8,
            text=[f"{v:.3f}" for v in report_df[m]], textposition='outside',
            textfont=dict(family='JetBrains Mono', size=10),
        ))
    fig_rep.update_layout(
        barmode='group',
        title=dict(text="Precision / Recall / F1 by Class", font=dict(color='#00e5a0', family='Orbitron', size=13)),
        legend=dict(bgcolor='#0b1120', bordercolor='#1a2d45', borderwidth=1, font=dict(family='JetBrains Mono', size=11)),
        paper_bgcolor='#0b1120', plot_bgcolor='#060a10',
        font=dict(color='#7a95b8', family='JetBrains Mono'),
        xaxis=dict(gridcolor='#1a2d45', color='#7a95b8'),
        yaxis=dict(gridcolor='#1a2d45', color='#7a95b8', range=[0,1.15]),
        margin=dict(t=50, b=40, l=40, r=20),
        height=300,
    )
    st.plotly_chart(fig_rep, use_container_width=True)
    st.dataframe(report_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRAFFIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="ng-header">📡 Traffic Analysis Console</div>', unsafe_allow_html=True)
    col_f1, col_f2, col_f3 = st.columns([1,1,2])
    with col_f1:
        filter_type = st.selectbox("Filter by type:", ['All'] + sorted(df['label'].unique().tolist()))
    with col_f2:
        sort_feat = st.selectbox("Sort by:", ['(none)'] + feature_cols)
    with col_f3:
        display_df = df if filter_type == 'All' else df[df['label'] == filter_type]
        if sort_feat != '(none)':
            display_df = display_df.sort_values(sort_feat, ascending=False)
        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace; font-size:0.78rem; color:#7a95b8; padding-top:28px;'>
        Showing: <span style='color:#00e5a0;'>{filter_type}</span> · Records: <span style='color:#d4e0f0;'>{len(display_df):,}</span>
        </div>
        """, unsafe_allow_html=True)
    st.dataframe(display_df.head(200), use_container_width=True, height=280)

    st.markdown("<br>", unsafe_allow_html=True)
    col_x, col_y, col_sz = st.columns(3)
    with col_x: x_feat = st.selectbox("X axis:", feature_cols, index=3)
    with col_y: y_feat = st.selectbox("Y axis:", feature_cols, index=4)
    with col_sz: sz_feat = st.selectbox("Size by:", ['(none)'] + feature_cols)

    sample_df = df.sample(min(1200, len(df)), random_state=42)
    sz_vals   = None
    if sz_feat != '(none)':
        raw_sz = sample_df[sz_feat].fillna(0)
        sz_vals = (raw_sz - raw_sz.min()) / (raw_sz.max() - raw_sz.min() + 1e-9) * 15 + 3

    fig_scatter = px.scatter(
        sample_df, x=x_feat, y=y_feat, color='label',
        color_discrete_map=COLORS, opacity=0.65,
        size=sz_vals if sz_feat != '(none)' else None,
        title=f"{x_feat} vs {y_feat}" + (f" (size={sz_feat})" if sz_feat != '(none)' else ""),
    )
    fig_scatter.update_layout(
        title=dict(font=dict(color='#00e5a0', family='Orbitron', size=13)),
        legend=dict(bgcolor='#0b1120', bordercolor='#1a2d45', borderwidth=1, font=dict(family='JetBrains Mono', size=11)),
        **DARK_LAYOUT, height=420,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('<div class="ng-header">📦 Feature Distribution by Class</div>', unsafe_allow_html=True)
    box_feat = st.selectbox("Feature to inspect:", feature_cols, index=3)
    fig_box  = go.Figure()
    for lbl in sorted(df['label'].unique()):
        vals = df[df['label'] == lbl][box_feat].dropna()
        fig_box.add_trace(go.Box(
            y=vals, name=lbl,
            marker_color=COLORS.get(lbl,'#888'),
            line=dict(color=COLORS.get(lbl,'#888')), boxmean='sd',
        ))
    fig_box.update_layout(
        title=dict(text=f"Distribution of '{box_feat}' by Class", font=dict(color='#00e5a0', family='Orbitron', size=13)),
        legend=dict(bgcolor='#0b1120', bordercolor='#1a2d45', borderwidth=1, font=dict(family='JetBrains Mono', size=11)),
        **DARK_LAYOUT, height=360,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="ng-header">📅 Attack Timeline Analysis</div>', unsafe_allow_html=True)
    np.random.seed(42)
    timeline_data = []
    for hour in range(24):
        base = 12 if 9 <= hour <= 17 else 5
        for attack_type in ['DDoS','PortScan','BruteForce','Bot']:
            mult = {'DDoS':2.5,'BruteForce':1.2,'PortScan':1.0,'Bot':0.7}[attack_type]
            cnt  = max(0, int(np.random.normal(base * mult, 4)))
            timeline_data.append({'Hour': f"{hour:02d}:00", 'Count': cnt, 'Attack Type': attack_type})
    timeline_df = pd.DataFrame(timeline_data)

    fig_line = px.line(
        timeline_df, x='Hour', y='Count', color='Attack Type',
        title="Attack Frequency Over 24 Hours",
        color_discrete_map={k: COLORS[k] for k in ['DDoS','PortScan','BruteForce','Bot']},
        markers=True,
    )
    fig_line.update_traces(line=dict(width=2), marker=dict(size=6))
    fig_line.update_layout(
        title=dict(font=dict(color='#00e5a0', family='Orbitron', size=13)),
        legend=dict(bgcolor='#0b1120', bordercolor='#1a2d45', borderwidth=1, font=dict(family='JetBrains Mono', size=11)),
        **DARK_LAYOUT, height=360,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    peak_df   = timeline_df.groupby('Hour')['Count'].sum().reset_index()
    peak_hour = peak_df.loc[peak_df['Count'].idxmax(), 'Hour']
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Peak Attack Hour",    peak_hour)
    c2.metric("Total Attacks (24h)", f"{peak_df['Count'].sum():,}")
    c3.metric("Avg Attacks / Hour",  f"{peak_df['Count'].mean():.0f}")
    c4.metric("Max Single Hour",     f"{peak_df['Count'].max():,}")

    st.markdown("<br>", unsafe_allow_html=True)
    pivot    = timeline_df.pivot(index='Attack Type', columns='Hour', values='Count')
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0,'#060a10'],[0.35,'#003a25'],[0.7,'#006644'],[1,'#00e5a0']],
        text=pivot.values, texttemplate="%{text}",
        textfont=dict(color='white', size=9, family='JetBrains Mono'),
    ))
    fig_heat.update_layout(
        title=dict(text="Attack Heatmap by Hour of Day", font=dict(color='#00e5a0', family='Orbitron', size=13)),
        xaxis=dict(color='#7a95b8', title='Hour'),
        yaxis=dict(color='#7a95b8', title=''),
        **{k:v for k,v in DARK_LAYOUT.items() if k not in ['xaxis','yaxis']}, height=260,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="ng-header">📈 Cumulative Attack Volume</div>', unsafe_allow_html=True)
    fig_area = go.Figure()
    for attack_type in ['DDoS','PortScan','BruteForce','Bot']:
        sub = timeline_df[timeline_df['Attack Type'] == attack_type]
        c   = COLORS[attack_type]
        r,g,b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
        fig_area.add_trace(go.Scatter(
            x=sub['Hour'], y=sub['Count'].cumsum(), name=attack_type,
            fill='tonexty', line=dict(color=c, width=2),
            fillcolor=f"rgba({r},{g},{b},0.12)",
        ))
    fig_area.update_layout(
        title=dict(text="Cumulative Attacks per Type (24h)", font=dict(color='#00e5a0', family='Orbitron', size=13)),
        legend=dict(bgcolor='#0b1120', bordercolor='#1a2d45', borderwidth=1, font=dict(family='JetBrains Mono', size=11)),
        **DARK_LAYOUT, height=300,
    )
    st.plotly_chart(fig_area, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ALERTS LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="ng-header">🔔 Alerts Log & Incident Tracker</div>', unsafe_allow_html=True)
    a1,a2,a3,a4 = st.columns(4)
    a1.metric("Open Incidents",    f"{random.randint(3,8)}",  delta="+2 today", delta_color="inverse")
    a2.metric("Blocked IPs",       f"{random.randint(40,80)}", delta=f"+{random.randint(3,10)} today")
    a3.metric("Avg Response Time", f"{random.uniform(0.3,1.2):.2f}s")
    a4.metric("False Positives",   f"{random.randint(1,5)}",  delta=f"-{random.randint(1,3)} vs yesterday")

    st.markdown("<br>", unsafe_allow_html=True)
    col_sev, col_atk = st.columns(2)
    with col_sev:
        sev_filter = st.multiselect("Severity:", ["CRITICAL","HIGH","MEDIUM"], default=["CRITICAL","HIGH","MEDIUM"])
    with col_atk:
        atk_filter = st.multiselect("Attack Type:", list(label_counts[label_counts.index!='BENIGN'].index),
                                    default=list(label_counts[label_counts.index!='BENIGN'].index))

    rng3 = random.Random(99)
    alert_records = []
    for i in range(40):
        atk  = rng3.choice([l for l in label_counts.index if l.upper() != 'BENIGN'])
        sev  = "CRITICAL" if atk.upper()=='DDOS' else "HIGH" if atk.upper() in ['BRUTEFORCE','BOT'] else "MEDIUM"
        src  = f"{rng3.randint(1,223)}.{rng3.randint(0,255)}.{rng3.randint(0,255)}.{rng3.randint(1,254)}"
        dst  = f"10.0.{rng3.randint(1,10)}.{rng3.randint(1,50)}"
        port = rng3.choice([22,80,443,3389,21,8080,53])
        ts   = (datetime.now() - timedelta(minutes=rng3.randint(0,480))).strftime("%Y-%m-%d %H:%M:%S")
        alert_records.append({
            'Timestamp': ts, 'Severity': sev, 'Attack': atk,
            'Source IP': src, 'Dest IP': dst, 'Port': port,
            'Status': rng3.choice(['BLOCKED','BLOCKED','BLOCKED','REVIEWING']),
            'Duration (s)': round(rng3.uniform(0.1,60), 2),
        })

    alerts_df = pd.DataFrame(alert_records)
    if sev_filter:
        alerts_df = alerts_df[alerts_df['Severity'].isin(sev_filter)]
    if atk_filter:
        alerts_df = alerts_df[alerts_df['Attack'].isin(atk_filter)]
    alerts_df = alerts_df.sort_values('Timestamp', ascending=False)
    st.dataframe(alerts_df, use_container_width=True, height=350)
    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.72rem; color:#3d5570; margin-top:8px;'>
    Showing {len(alerts_df)} alerts · Last updated: {datetime.now().strftime("%H:%M:%S")}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_p, col_q = st.columns(2)
    with col_p:
        sev_counts = pd.DataFrame(alert_records)['Severity'].value_counts()
        fig_sev = go.Figure(go.Pie(
            labels=sev_counts.index, values=sev_counts.values, hole=0.55,
            marker=dict(colors=['#ff3c5a','#ff8c00','#ffe033'], line=dict(color='#060a10', width=2)),
        ))
        fig_sev.update_layout(
            title=dict(text="Alert Severity Distribution", font=dict(color='#00e5a0', family='Orbitron', size=12)),
            legend=dict(bgcolor='#0b1120', bordercolor='#1a2d45', borderwidth=1, font=dict(family='JetBrains Mono', size=11)),
            **{k:v for k,v in DARK_LAYOUT.items() if k not in ['xaxis','yaxis']}, height=280,
        )
        st.plotly_chart(fig_sev, use_container_width=True)
    with col_q:
        atk_counts = pd.DataFrame(alert_records)['Attack'].value_counts()
        fig_atk = go.Figure(go.Bar(
            x=atk_counts.values, y=atk_counts.index, orientation='h',
            marker=dict(color=[COLORS.get(l,'#888') for l in atk_counts.index], opacity=0.85),
            text=atk_counts.values, textposition='outside',
            textfont=dict(family='JetBrains Mono', size=11),
        ))
        fig_atk.update_layout(
            title=dict(text="Alerts by Attack Type", font=dict(color='#00e5a0', family='Orbitron', size=12)),
            **DARK_LAYOUT, height=280,
        )
        st.plotly_chart(fig_atk, use_container_width=True)