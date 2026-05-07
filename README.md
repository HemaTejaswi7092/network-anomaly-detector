# 🛡️ NetGuard — Network Traffic Anomaly Detector

An ML-powered cybersecurity dashboard that detects network intrusions in real time using Random Forest classification.

## 🚀 Live Demo
👉 **[Try it live here](https://netguard-anomaly-detector.streamlit.app/)**

---

## ✨ Features
- 📊 Real-time threat dashboard with attack breakdown
- 🔍 Live traffic simulator — adjust sliders and get instant predictions
- 🧠 99.9% accurate Random Forest classifier
- 📡 Detects 5 attack types: DDoS, PortScan, BruteForce, Bot, and normal traffic
- 📈 Confusion matrix and feature importance analysis
- 📁 Upload your own CSV network traffic data

## 🛠️ Tech Stack
- **Python** — core language
- **Scikit-learn** — Random Forest classifier
- **Streamlit** — interactive dashboard
- **Plotly** — cybersecurity-themed visualizations
- **Pandas + NumPy** — data processing

## 🔐 Attack Types Detected
| Attack | Description |
|--------|-------------|
| DDoS | Floods server with fake traffic to crash it |
| PortScan | Probes network looking for weak entry points |
| BruteForce | Tries thousands of passwords to break in |
| Bot | Automated malicious programs communicating |
| BENIGN | Normal legitimate traffic |

## 🚀 Run Locally
```bash
git clone https://github.com/HemaTejaswi7092/network-anomaly-detector.git
cd network-anomaly-detector
pip install -r requirements.txt
streamlit run app.py
```

## 🔮 Future Improvements
- Real-time packet capture using Scapy
- Email/Slack alerts when attack detected
- Deep learning upgrade with neural networks
- REST API endpoint for integration
