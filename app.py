import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import warnings
import random
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="NetGuard — Network Anomaly Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;600&display=swap');
.stApp { background-color: #0a0e1a; }
section[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #1a2a1a;
}
.stApp, .stMarkdown, p, div { color: #c9d1d9; font-family: 'Inter', sans-serif; }
div[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1e3a1e;
    border-radius: 8px;
    padding: 16px;
}
div[data-testid="metric-container"] > div > div > div {
    color: #00ff90 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 2rem !important;
}
div[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #0d1117;
    border-bottom: 1px solid #1e3a1e;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    color: #00ff90 !important;
    border-bottom: 2px solid #00ff90 !important;
}
.stButton > button {
    background: transparent;
    border: 1px solid #00ff90;
    color: #00ff90;
    font-family: 'Share Tech Mono', monospace;
    border-radius: 4px;
    padding: 8px 24px;
}
.stButton > button:hover { background: #00ff9015; }
.threat-critical {
    background: #2a0a0a;
    border-left: 3px solid #ff0000;
    padding: 12px 16px; border-radius: 4px; margin: 6px 0;
    font-family: 'Share Tech Mono', monospace; font-size: 0.85rem; color: #ff6666;
}
.threat-high {
    background: #2a1a00;
    border-left: 3px solid #ff6600;
    padding: 12px 16px; border-radius: 4px; margin: 6px 0;
    font-family: 'Share Tech Mono', monospace; font-size: 0.85rem; color: #ffaa44;
}
.threat-normal {
    background: #0a1a0a;
    border-left: 3px solid #00ff90;
    padding: 12px 16px; border-radius: 4px; margin: 6px 0;
    font-family: 'Share Tech Mono', monospace; font-size: 0.85rem; color: #00ff90;
}
.section-header {
    font-family: 'Share Tech Mono', monospace;
    color: #00ff90; font-size: 0.75rem;
    letter-spacing: 0.15em; text-transform: uppercase;
    border-bottom: 1px solid #1e3a1e;
    padding-bottom: 8px; margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0;'>
        <div style='font-size:2.5rem'>🛡️</div>
        <div style='font-family: Share Tech Mono, monospace; color: #00ff90; font-size: 1.1rem; margin-top: 8px;'>NETGUARD</div>
        <div style='color: #8b949e; font-size: 0.7rem; letter-spacing: 0.1em;'>NETWORK ANOMALY DETECTOR</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header">📡 Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio("", ["Use Demo Dataset", "Upload CSV File"], label_visibility="collapsed")
    if data_source == "Upload CSV File":
        uploaded_csv = st.file_uploader("Upload network traffic CSV", type=["csv"])
        st.markdown("""
        <div style='font-size:0.72rem; color:#8b949e; margin-top:8px;'>
        CSV must have a 'label' column with traffic types like BENIGN, DDoS, PortScan etc.
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded_csv = None
    st.markdown("---")
    st.markdown('<div class="section-header">⚙️ Model Settings</div>', unsafe_allow_html=True)
    n_trees = st.slider("Number of Trees", 50, 200, 100, 50)
    test_size = st.slider("Test Split %", 10, 40, 20, 5)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#8b949e; text-align:center;'>
        v2.0 | Random Forest Classifier<br>
        CICIDS2017 Dataset Structure<br><br>
        <span style='color:#00ff90'>● SYSTEM ONLINE</span>
    </div>
    """, unsafe_allow_html=True)

# ── HEADER ──
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <h1 style='font-family: Share Tech Mono, monospace; color: #00ff90; margin-bottom: 4px; font-size: 1.8rem;'>
        🛡️ NetGuard — Network Anomaly Detector
    </h1>
    <p style='color: #8b949e; font-size: 0.85rem; margin: 0;'>
        ML-powered intrusion detection system | Random Forest Classifier | CICIDS2017 Structure
    </p>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style='text-align:right; font-family: Share Tech Mono, monospace; font-size: 0.75rem; color: #8b949e; padding-top: 16px;'>
        STATUS: <span style='color:#00ff90'>● ACTIVE</span><br>
        MODE: DETECTION
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

# ── DATASET GENERATION ──
@st.cache_data
def generate_demo_dataset():
    np.random.seed(42)
    samples = {'BENIGN': 3000, 'DDoS': 800, 'PortScan': 400, 'BruteForce': 300, 'Bot': 200}
    patterns = {
        'BENIGN':     dict(dur=(2,1),   pkt=(500,100),    bps=(5000,2000),    pps=(10,5),    fwd=(20,5),   bwd=(18,5),  syn=(1,0.5),   rst=(0,0.01)),
        'DDoS':       dict(dur=(0.1,.05),pkt=(60,10),     bps=(500000,100000),pps=(1000,200),fwd=(500,100),bwd=(2,1),   syn=(500,100), rst=(100,20)),
        'PortScan':   dict(dur=(0.5,.2), pkt=(40,5),      bps=(1000,500),     pps=(50,10),   fwd=(1,.5),   bwd=(1,.5),  syn=(1,.5),    rst=(1,.5)),
        'BruteForce': dict(dur=(5,2),    pkt=(200,50),    bps=(2000,500),     pps=(20,5),    fwd=(30,10),  bwd=(30,10), syn=(1,.5),    rst=(5,2)),
        'Bot':        dict(dur=(10,3),   pkt=(300,50),    bps=(3000,500),     pps=(15,3),    fwd=(25,8),   bwd=(25,8),  syn=(1,.5),    rst=(0,0.01)),
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
def train_model(_df, n_estimators, test_split):
    feature_cols = [c for c in _df.columns if c != 'label']
    X = _df[feature_cols]
    le = LabelEncoder()
    y = le.fit_transform(_df['label'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split/100, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    fi = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    cm = confusion_matrix(y_test, y_pred)
    return model, le, acc, fi, X_test, y_test, y_pred, cm, feature_cols

# ── LOAD DATA ──
with st.spinner("🔄 Initializing detection system..."):
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            if 'label' not in df.columns and ' Label' in df.columns:
                df = df.rename(columns={' Label': 'label'})
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0).abs()
            st.success(f"✅ Loaded {len(df):,} records from uploaded file")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            df = generate_demo_dataset()
    else:
        df = generate_demo_dataset()
    model, le, accuracy, fi, X_test, y_test, y_pred, cm, feature_cols = train_model(df, n_trees, test_size)

label_counts = df['label'].value_counts()
total = len(df)
benign = label_counts.get('BENIGN', label_counts.get('benign', 0))
attacks = total - benign
threat_pct = attacks / total * 100

# ── THREAT BANNER ──
if threat_pct > 40:
    st.markdown(f'<div class="threat-critical">🚨 CRITICAL THREAT LEVEL — {threat_pct:.1f}% of traffic is malicious — Immediate action required</div>', unsafe_allow_html=True)
elif threat_pct > 20:
    st.markdown(f'<div class="threat-high">⚠️ HIGH THREAT LEVEL — {threat_pct:.1f}% of traffic flagged as suspicious</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="threat-normal">✅ NORMAL OPERATIONS — {threat_pct:.1f}% threat level — System nominal</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──
tab1, tab2, tab3, tab4 = st.tabs(["[ OVERVIEW ]", "[ LIVE DETECTOR ]", "[ MODEL INTEL ]", "[ TRAFFIC ANALYSIS ]"])

PLOT_DARK = dict(paper_bgcolor='#0d1117', plot_bgcolor='#0a0e1a',
                 font=dict(color='#c9d1d9'),
                 xaxis=dict(gridcolor='#1e2a1e', color='#8b949e'),
                 yaxis=dict(gridcolor='#1e2a1e', color='#8b949e'))

# ── TAB 1: OVERVIEW ──
with tab1:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("TOTAL CONNECTIONS", f"{total:,}")
    c2.metric("NORMAL TRAFFIC", f"{benign:,}")
    c3.metric("ATTACKS DETECTED", f"{attacks:,}")
    c4.metric("THREAT LEVEL", f"{threat_pct:.1f}%")
    c5.metric("MODEL ACCURACY", f"{accuracy:.1%}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        colors = {'BENIGN':'#00ff90','DDoS':'#ff4444','PortScan':'#ff9900','BruteForce':'#ff44ff','Bot':'#44aaff'}
        color_list = [colors.get(l, '#888888') for l in label_counts.index]
        fig = go.Figure(data=[go.Pie(
            labels=label_counts.index, values=label_counts.values, hole=0.6,
            marker=dict(colors=color_list, line=dict(color='#0a0e1a', width=2))
        )])
        fig.update_layout(
            title=dict(text="Traffic Distribution", font=dict(color='#00ff90', family='Share Tech Mono')),
            legend=dict(bgcolor='#0d1117', bordercolor='#2a3a2a', borderwidth=1),
            annotations=[dict(text=f'{threat_pct:.0f}%<br>threats', x=0.5, y=0.5,
                             font=dict(size=14, color='#ff4444', family='Share Tech Mono'), showarrow=False)],
            **{k: v for k, v in PLOT_DARK.items() if k not in ['xaxis', 'yaxis']}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        attack_only = label_counts[label_counts.index != 'BENIGN']
        if len(attack_only) > 0:
            fig2 = go.Figure(go.Bar(
                x=attack_only.values, y=attack_only.index, orientation='h',
                marker=dict(color=['#ff4444','#ff9900','#ff44ff','#44aaff'][:len(attack_only)],
                           line=dict(color='#333333', width=1))
            ))
            fig2.update_layout(
                title=dict(text="Attack Vector Frequency", font=dict(color='#00ff90', family='Share Tech Mono')),
                **PLOT_DARK
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Threat feed
    st.markdown('<div class="section-header">🔴 Live Threat Feed</div>', unsafe_allow_html=True)
    attack_types = [l for l in label_counts.index if l != 'BENIGN']
    if attack_types:
        random.seed(42)
        fake_ips = [f"192.168.{random.randint(1,255)}.{random.randint(1,255)}" for _ in range(10)]
        dest_ips = [f"10.0.{random.randint(1,10)}.{random.randint(1,50)}" for _ in range(10)]
        for i in range(min(8, attacks)):
            attack = random.choice(attack_types)
            src = fake_ips[i % len(fake_ips)]
            dst = dest_ips[i % len(dest_ips)]
            severity = "CRITICAL" if attack == 'DDoS' else "HIGH" if attack == 'BruteForce' else "MEDIUM"
            color = "#ff4444" if severity == "CRITICAL" else "#ff9900" if severity == "HIGH" else "#ffff44"
            st.markdown(f"""
            <div style='background:#0d1117; border:1px solid #1e2a1e; border-left:3px solid {color};
                        padding:10px 14px; margin:4px 0; border-radius:4px;
                        font-family:Share Tech Mono,monospace; font-size:0.8rem;'>
                <span style='color:{color}'>[{severity}]</span>
                <span style='color:#8b949e'> {attack} detected |</span>
                <span style='color:#c9d1d9'> SRC: {src} → DST: {dst}</span>
                <span style='color:#8b949e; float:right'>⬤ BLOCKED</span>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2: LIVE DETECTOR ──
with tab2:
    st.markdown('<div class="section-header">🔍 Real-Time Traffic Classifier</div>', unsafe_allow_html=True)
    st.markdown("Adjust the parameters below to simulate network traffic. The model predicts in real time.")

    col1, col2 = st.columns(2)
    inputs = {}
    slider_config = {
        'duration':               ("Connection Duration (s)", 0.0, 20.0, 2.0),
        'packet_length_mean':     ("Mean Packet Length (bytes)", 0, 1500, 500),
        'packet_length_std':      ("Packet Length Std Dev", 0, 500, 200),
        'flow_bytes_per_sec':     ("Flow Bytes/sec", 0, 1000000, 5000),
        'flow_packets_per_sec':   ("Flow Packets/sec", 0, 2000, 10),
        'fwd_packets':            ("Forward Packets", 0, 1000, 20),
        'bwd_packets':            ("Backward Packets", 0, 1000, 18),
        'fwd_packet_length_mean': ("Fwd Packet Length Mean", 0, 1500, 400),
        'bwd_packet_length_mean': ("Bwd Packet Length Mean", 0, 1500, 600),
        'syn_flag_count':         ("SYN Flag Count", 0, 1000, 1),
        'rst_flag_count':         ("RST Flag Count", 0, 500, 0),
        'fin_flag_count':         ("FIN Flag Count", 0, 10, 1),
        'idle_mean':              ("Idle Mean", 0.0, 10.0, 1.0),
        'active_mean':            ("Active Mean", 0.0, 10.0, 3.0),
    }
    keys = list(slider_config.keys())
    with col1:
        for k in keys[:7]:
            label, mn, mx, val = slider_config[k]
            inputs[k] = st.slider(label, mn, mx, val)
    with col2:
        for k in keys[7:]:
            label, mn, mx, val = slider_config[k]
            inputs[k] = st.slider(label, mn, mx, val)

    input_df = pd.DataFrame([[inputs[f] for f in feature_cols]], columns=feature_cols)
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    pred_label = le.inverse_transform([pred])[0]
    confidence = max(probs) * 100

    st.markdown("---")
    if pred_label == 'BENIGN':
        st.markdown(f"""
        <div style='background:#0a1a0a; border:1px solid #00ff90; border-radius:8px; padding:20px; text-align:center;'>
            <div style='font-family:Share Tech Mono,monospace; font-size:1.5rem; color:#00ff90;'>✅ BENIGN TRAFFIC</div>
            <div style='color:#8b949e; margin-top:8px;'>Confidence: {confidence:.1f}% — No threat detected</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#1a0a0a; border:1px solid #ff4444; border-radius:8px; padding:20px; text-align:center;'>
            <div style='font-family:Share Tech Mono,monospace; font-size:1.5rem; color:#ff4444;'>🚨 ATTACK DETECTED: {pred_label}</div>
            <div style='color:#8b949e; margin-top:8px;'>Confidence: {confidence:.1f}% — Immediate action recommended</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    prob_df = pd.DataFrame({'Type': le.classes_, 'Probability': probs}).sort_values('Probability', ascending=True)
    fig_prob = go.Figure(go.Bar(
        x=prob_df['Probability'], y=prob_df['Type'], orientation='h',
        marker=dict(color=['#00ff90','#44aaff','#ff44ff','#ff9900','#ff4444'][:len(prob_df)],
                   line=dict(color='#1e2a1e', width=1))
    ))
    fig_prob.update_layout(
        title=dict(text="Classification Confidence", font=dict(color='#00ff90', family='Share Tech Mono')),
        xaxis=dict(gridcolor='#1e2a1e', color='#8b949e', tickformat='.0%'),
        yaxis=dict(gridcolor='#1e2a1e', color='#8b949e'),
        **{k: v for k, v in PLOT_DARK.items() if k not in ['xaxis','yaxis']}
    )
    st.plotly_chart(fig_prob, use_container_width=True)

# ── TAB 3: MODEL INTEL ──
with tab3:
    st.markdown('<div class="section-header">🧠 Model Intelligence Report</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ACCURACY", f"{accuracy:.1%}")
    c2.metric("TEST SAMPLES", f"{len(y_test):,}")
    c3.metric("ATTACK CLASSES", f"{len(le.classes_)}")
    c4.metric("DECISION TREES", f"{n_trees}")

    col1, col2 = st.columns(2)
    with col1:
        labels = le.classes_
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0,'#0a0e1a'],[0.5,'#004400'],[1,'#00ff90']],
            text=cm, texttemplate="%{text}",
            textfont=dict(color='white', family='Share Tech Mono')
        ))
        fig_cm.update_layout(
            title=dict(text="Confusion Matrix", font=dict(color='#00ff90', family='Share Tech Mono')),
            xaxis=dict(title='Predicted', color='#8b949e', gridcolor='#1e2a1e'),
            yaxis=dict(title='Actual', color='#8b949e', gridcolor='#1e2a1e'),
            **{k: v for k, v in PLOT_DARK.items() if k not in ['xaxis','yaxis']}
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        fig_fi = go.Figure(go.Bar(
            x=fi.head(10)['importance'], y=fi.head(10)['feature'], orientation='h',
            marker=dict(color='#00ff90', line=dict(color='#004400', width=1))
        ))
        fig_fi.update_layout(
            title=dict(text="Top 10 Detection Features", font=dict(color='#00ff90', family='Share Tech Mono')),
            xaxis=dict(gridcolor='#1e2a1e', color='#8b949e'),
            yaxis=dict(gridcolor='#1e2a1e', color='#8b949e', categoryorder='total ascending'),
            **{k: v for k, v in PLOT_DARK.items() if k not in ['xaxis','yaxis']}
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df = report_df.drop(['accuracy','macro avg','weighted avg'], errors='ignore')
    report_df = report_df[['precision','recall','f1-score']].round(3)
    st.markdown('<div class="section-header">📊 Per-Class Performance</div>', unsafe_allow_html=True)
    st.dataframe(report_df, use_container_width=True)

# ── TAB 4: TRAFFIC ANALYSIS ──
with tab4:
    st.markdown('<div class="section-header">📡 Traffic Analysis Console</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        filter_type = st.selectbox("Filter by type:", ['All'] + list(df['label'].unique()))
    with col2:
        count = len(df[df['label']==filter_type]) if filter_type != 'All' else len(df)
        st.markdown(f"""
        <div style='font-family:Share Tech Mono,monospace; font-size:0.8rem; color:#8b949e; padding-top:32px;'>
        Showing: {filter_type} | Records: {count:,}
        </div>
        """, unsafe_allow_html=True)

    display_df = df if filter_type == 'All' else df[df['label'] == filter_type]
    st.dataframe(display_df.head(200), use_container_width=True, height=300)

    col1, col2 = st.columns(2)
    with col1:
        x_feat = st.selectbox("X axis:", feature_cols, index=3)
    with col2:
        y_feat = st.selectbox("Y axis:", feature_cols, index=4)

    color_map = {'BENIGN':'#00ff90','DDoS':'#ff4444','PortScan':'#ff9900','BruteForce':'#ff44ff','Bot':'#44aaff'}
    sample = df.sample(min(800, len(df)), random_state=42)
    fig_s = px.scatter(
        sample, x=x_feat, y=y_feat, color='label',
        color_discrete_map=color_map, opacity=0.7,
        title=f"{x_feat} vs {y_feat}"
    )
    fig_s.update_layout(
        title=dict(font=dict(color='#00ff90', family='Share Tech Mono')),
        legend=dict(bgcolor='#0d1117', bordercolor='#2a3a2a', borderwidth=1),
        **PLOT_DARK
    )
    st.plotly_chart(fig_s, use_container_width=True)