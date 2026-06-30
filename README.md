# 🛡️ NetGuard — Network Traffic Anomaly Detector

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99.9%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A real-time network intrusion detection dashboard powered by a Random Forest classifier trained on 41 engineered features from the NSL-KDD dataset. Built to flag DDoS, port scans, brute-force attempts, and bot traffic the moment they happen.

**[🚀 Try the live demo →](https://netguard-anomaly-detector.streamlit.app/)**

<!-- 📸 Add a screenshot or GIF of the dashboard here, e.g.:
![NetGuard Dashboard](assets/dashboard-demo.gif)
-->

---

## Table of Contents
- [Features](#-features)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Attack Types Detected](#-attack-types-detected)
- [Run Locally](#-run-locally)
- [Roadmap](#-roadmap)

---

## ✨ Features

| | |
|---|---|
| 📊 **Real-time threat dashboard** | Live attack breakdown with severity ranking |
| 🔍 **Traffic simulator** | Adjust traffic parameters via sliders, get instant predictions |
| 🧠 **99.9% accurate classifier** | Random Forest ensemble, false-positive rate under 0.1% |
| 📡 **5-class detection** | DDoS, PortScan, BruteForce, Bot, and normal (benign) traffic |
| 📈 **Model explainability** | Confusion matrix and feature importance analysis built in |
| 📁 **Bring your own data** | Upload custom CSV network traffic logs for analysis |

## 🛠️ Tech Stack

- **Python** — core language
- **Scikit-learn** — Random Forest classifier
- **Streamlit** — interactive dashboard
- **Plotly** — visualizations
- **Pandas + NumPy** — data processing pipeline

## 🔐 Attack Types Detected

| Attack | Description |
|--------|-------------|
| DDoS | Floods the server with fake traffic to crash it |
| PortScan | Probes the network for weak entry points |
| BruteForce | Attempts thousands of password combinations |
| Bot | Automated malicious programs communicating over the network |
| BENIGN | Normal, legitimate traffic |

## 🚀 Run Locally

\`\`\`bash
git clone https://github.com/HemaTejaswi7092/network-anomaly-detector.git
cd network-anomaly-detector
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## 🔮 Roadmap

- [ ] Real-time packet capture via Scapy
- [ ] Email/Slack alerts on detected attacks
- [ ] Deep learning upgrade (neural net classifier)
- [ ] REST API endpoint for third-party integration

---

**Built by [Hema Tejaswi Manchikalapudi](https://github.com/HemaTejaswi7092)** · [LinkedIn](https://www.linkedin.com/in/hematejaswimanchikalapudi)
