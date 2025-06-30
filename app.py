import streamlit as st
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Benjamin Imo Uka | Data Science Portfolio",
    page_icon="📊",
    layout="wide"
)

# --- PROFILE IMAGE ---
st.image("profile.png", width=100)
st.markdown("""
**Benjamin Imo Uka**  
Data Scientist | AI & Data Engineer  
📍 Port Harcourt, Nigeria

✉ [Email](mailto:benjaminukaimo@gmail.com)  
📞 +234 706 719 3071  
🔗 [LinkedIn](https://www.linkedin.com/in/benjamin-uka-imo)  
💻 [GitHub](https://github.com/uka-ben)  
📺 [YouTube](https://youtube.com/@blackdatascience)
""")

# --- HEADER ---
st.title("Benjamin Imo Uka")
st.subheader("Data Scientist | AI & Data Engineer | Specializing in DRL, NLP & Anomaly Detection")

# --- ABOUT ME ---
st.markdown("## 🧠 About Me")
st.markdown("""
I am a dynamic and solutions-oriented Data Scientist and AI Engineer with strong expertise in machine learning, deep reinforcement learning (DRL), natural language processing (NLP), anomaly detection, and end-to-end data engineering. I am skilled in deploying scalable ML systems and building real-time analytics platforms using cloud-native tools such as Databricks, GCP, and Azure.

My core strength lies in solving complex business problems using AI-first methodologies—leveraging technologies like GNNs, Transformers, LLMs, and DRL to deliver intelligent insights and automation at scale.
""")

# --- TECHNICAL SKILLS ---
st.markdown("## 🛠️ Technical Skills")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Machine Learning & AI")
    st.markdown("""
    - Time Series (TFT, Prophet)
    - Graph Neural Networks (Graphormer, DGL)
    - Deep Reinforcement Learning (PPO, A2C, DDPG)
    - NLP: LLMs, RAG
    - Anomaly Detection & Cybersecurity
    - AutoML: AutoGluon, H2O.ai, PyCaret
    """)

with col2:
    st.markdown("### Programming & Tools")
    st.markdown("""
    - Python, SQL, PySpark, Spark SQL
    - TensorFlow, PyTorch, scikit-learn, Hugging Face
    - Gymnasium, RLlib, Stable-Baselines3
    - Visualization: Matplotlib, Seaborn, Plotly, Power BI
    """)

with col3:
    st.markdown("### Data Engineering & MLOps")
    st.markdown("""
    - Delta Live Tables, Spark Streaming, ETL, Airflow
    - Databricks, GCP (BigQuery, Dataflow), Azure
    - MLflow, Docker, CI/CD Pipelines, Flask APIs
    - Warehousing: Snowflake, Delta Lake, BigQuery
    """)

# --- PROFESSIONAL EXPERIENCE ---
st.markdown("## 💼 Professional Experience")

st.markdown("""
**Data Scientist**  
*Miracle Health Recruitment, UK (Remote | Nov 2024 – Feb 2025)*  
- Collaborated with cross-functional teams to embed ML solutions into operational systems.

**Junior Data Scientist**  
*Baknance Technology (Remote | Feb 2023 – Apr 2024)*  
- Developed RAG-based chatbot using Hugging Face and OpenAI.
- Built a GNN and DRL-powered fraud detection system with 50% improvement in accuracy.
- Led sentiment analysis for financial predictions, reducing default risks by 30%.
- Delivered forecasting models that optimized decision-making efficiency.
""")

# --- PROJECT HIGHLIGHTS ---
st.markdown("## 🚀 Key Projects")

st.markdown("### 📈 Financial Systems")
st.markdown("""
- [Stock Market Analysis](https://timetion.streamlit.app/): Anti-symmetric pattern detection for real-time insights.
- [Portfolio Optimizer](https://ben-stock-deep-learning.streamlit.app/): DRL-based investment strategy model.
- Sixfold trading model with DRL, LSTM, QNN, Sentiment Analysis.
""")

st.markdown("### 🤖 AI & NLP Solutions")
st.markdown("""
- [benGPT](https://benhealthcare.streamlit.app/): Conversational LLM chatbot for personalized Q&A.
- Multimodal healthcare assistant that improves diagnosis accuracy by 35%.
""")

st.markdown("### 🔐 Anomaly Detection Systems")
st.markdown("""
- Real-time fraud detection using GNNs and Transformers.
- Cybersecurity threat modeling with DRL and behavioral analytics.
""")

st.markdown("### ♻️ Optimization & Predictive Tools")
st.markdown("""
- [Carbon Emission Optimizer](https://ben-co2optimization.streamlit.app/): DRL agent for global CO₂ reduction strategies.
- [ML Decision Suite](https://benjitable-ds.streamlit.app/): Real-time analytics tool for classification, regression, clustering.
- RL-based IoT Predictive Maintenance System.
""")

st.markdown("### 🕹️ Robotics & Game AI")
st.markdown("""
- Trained agents in robotic simulations using PPO and SAC.
- Developed multi-agent game-playing reinforcement learning environments.
""")

# --- EDUCATION ---
st.markdown("## 🎓 Education")
st.markdown("""
- **B.Sc. in Accountancy**, Imo State University, Nigeria (2016)
- **National Diploma in Accountancy**, Imo State Polytechnic, Nigeria (2012)
""")

# --- CERTIFICATIONS ---
st.markdown("## 📄 Certifications")
st.markdown("""
- Neural Networks – Simplilearn  
- Deep Reinforcement Learning – Hugging Face *(In Progress)*
""")

# --- ADDITIONAL ACTIVITIES ---
st.markdown("## 🌍 Additional Involvement")
st.markdown("""
- 🎥 Content Creator: [Black Data Science](https://youtube.com/@blackdatascience) – Tutorials on ML, AI, and Data Engineering.
- 💻 Contributor: Actively involved in open-source ML/AI projects on GitHub.
""")

st.success("Thanks for viewing my professional portfolio! Feel free to reach out for collaborations or opportunities.")
