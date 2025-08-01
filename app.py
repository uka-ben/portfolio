import streamlit as st
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Benjamin Uka Imo | AI & Data Engineering Portfolio",
    page_icon="📊",
    layout="wide"
)

# --- PROFILE IMAGE ---
image = Image.open("profile.png")
st.image(image, width=130)

# --- HEADER ---
st.title("Benjamin Uka Imo")
st.subheader("AI & Data Engineer | Machine Learning Systems | LLMs | MLOps | Databricks | PySpark")

st.markdown("""
📍 Port Harcourt, Nigeria  
✉ [benjaminukaimo@gmail.com](mailto:benjaminukaimo@gmail.com)  
📞 +234 706 719 3071  
🔗 [LinkedIn](https://www.linkedin.com/in/benjamin-uka-imo)  
💻 [GitHub](https://github.com/uka-ben)  
📺 [YouTube](https://youtube.com/@blackdatascience)
""")

# --- ABOUT ME ---
st.markdown("## 🧠 About Me")
st.markdown("""
Dynamic and solutions-oriented AI & Data Engineer with experience in end-to-end machine learning systems, MLOps pipelines, real-time big data processing, and scalable AI deployments. Skilled in Databricks, PySpark, Delta Live Tables, and large-scale transformer-based architectures (LLMs, RAGs). 

My mission is to build intelligent, production-ready systems across FinTech, HealthTech, and infrastructure analytics, leveraging GNNs, Deep RL, NLP, and CI/CD pipelines.
""")

# --- TECHNICAL SKILLS ---
st.markdown("## 🛠️ Technical Skills")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Machine Learning & AI")
    st.markdown("""
    - Time Series (TFT, Prophet), Deep RL (PPO, SAC), Quantum ML (Basic)
    - Anomaly Detection, Risk Modeling, Cybersecurity Analytics
    - NLP: LLM, RAG, Chatbots, Sentiment Analysis
    - AutoML: AutoGluon, H2O.ai, PyCaret
    """)

with col2:
    st.markdown("### Programming & Toolkits")
    st.markdown("""
    - Python, SQL, PySpark, Spark SQL
    - Hugging Face, Stable-Baselines3, Gymnasium
    - TensorFlow, PyTorch, scikit-learn, RLlib
    - Data Viz: Matplotlib, Seaborn, Plotly, Power BI, YData Profiler
    """)

with col3:
    st.markdown("### Data Engineering & Cloud MLOps")
    st.markdown("""
    - Databricks, Delta Live Tables, AutoLoader, Apache Beam
    - GCP (BigQuery, Dataflow), Azure Databricks
    - CI/CD, MLflow, Docker, Airflow, Flask API
    - Warehousing: Snowflake, Delta Lake
    """)

# --- EXPERIENCE ---
st.markdown("## 💼 Professional Experience")
st.markdown("""
**Data Scientist**  
*Miracle Health Recruitment, UK (Remote | Nov 2024 – Feb 2025)*  
- Built streaming pipelines using Delta Live Tables and AutoLoader.  
- Integrated data science workflows across team operations.

**Junior Data Scientist**  
*Baknance Technology (Remote | Feb 2023 – Apr 2024)*  
- Developed ML systems using Databricks ML Runtime for real-time inference.  
- Created RAG-based chatbots using OpenAI + Hugging Face integrated with PySpark ETL pipelines.  
- Built fraud detection system using GNNs + DRL reducing false positives by 50%.  
- Delivered financial risk models & sentiment analysis pipelines.  
- Implemented Delta Lake workflows for continuous model monitoring.
""")

# --- PROJECT HIGHLIGHTS ---
st.markdown("## 🚀 Projects")

st.markdown("### 📈 Finance & Risk Modeling")
st.markdown("""
- [Stock Market Analysis](https://timetion.streamlit.app/): Detected void anti-symmetric patterns in time series.
- [Portfolio Optimizer](https://ben-stock-deep-learning.streamlit.app/): DRL-based asset allocator.
- Sixfold strategy model: Ensemble + QNN + DL + DRL + LSTM + Time Series.
""")

st.markdown("### 🤖 AI & NLP Chatbots")
st.markdown("""
- [benGPT](https://benhealthcare.streamlit.app/): Deployed LLM chatbot using OpenAI API.
- Document-aware chatbots with RAG architecture + vector search.
- [Healthcare NLP Tool](https://benhealthcare.streamlit.app/): Multimodal diagnosis assistant (↑ prediction accuracy by 35%).
""")

st.markdown("### 🔐 Anomaly Detection Systems")
st.markdown("""
- Built DRL-GNN-Transformer hybrid model for fraud detection.
- Designed real-time anomaly detection system for e-commerce + healthcare claims.
- Set up threat detection using user behavior and traffic monitoring.
""")

st.markdown("### ♻️ Optimization & Predictive Tools")
st.markdown("""
- [CO₂ Optimizer](https://ben-co2optimization.streamlit.app/): DRL for global carbon policy modeling.
- [ML Analytics Suite](https://benjitable-ds.streamlit.app/): Built unified system for clustering, classification & anomaly detection.
- IoT Failure Prediction: RL-powered predictive maintenance for industrial IoT.
""")

st.markdown("### 🕹️ Robotics & Game AI")
st.markdown("""
- PPO and SAC-trained RL agents in robotics control environments.
- Multi-agent RL systems for autonomous decision-making and coordination.
""")

# --- EDUCATION & CERTIFICATIONS ---
st.markdown("## 🎓 Education & Certifications")
st.markdown("""
- **B.Sc. in Accountancy**, Imo State University (2016)  
- **National Diploma in Accountancy**, Imo State Polytechnic (2012)
- Deep Reinforcement Learning – Hugging Face *(in progress)*  
- Neural Networks – SIMPLILEARN
""")

# --- COMMUNITY & EXTRAS ---
st.markdown("## 🌍 Community & Open Source")
st.markdown("""
- 📺 [YouTube: Black Data Science](https://youtube.com/@blackdatascience): Tutorials on AI, ML & Data Engineering.
- 💻 [GitHub](https://github.com/uka-ben): Open source contributions to AI/ML.
""")

# --- CONTACT ---
st.markdown("## 📬 Get in Touch")
st.markdown("""
- ✉ [benjaminukaimo@gmail.com](mailto:benjaminukaimo@gmail.com)  
- 🔗 [LinkedIn](https://www.linkedin.com/in/benjamin-uka-imo)  
- 💻 [GitHub](https://github.com/uka-ben)  
- 📺 [YouTube](https://youtube.com/@blackdatascience)  


st.success("Thank you for visiting my AI & Data Engineering Portfolio! Let's collaborate or innovate together.")
