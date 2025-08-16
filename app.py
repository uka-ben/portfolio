import streamlit as st
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Benjamin Uka Imo | AI & Data Engineering Portfolio",
    page_icon="📊",
    layout="wide"
)

# --- PROFILE SECTION ---
col1, col2 = st.columns([1, 3])
with col1:
    image = Image.open("profile.png")
    st.image(image, width=150)
with col2:
    st.title("Benjamin Imo Uka")
    st.subheader("AI & Data Engineering | ML Systems, DRL | Databricks, MLOps")
    st.markdown("""
    📍 Port Harcourt, Nigeria  
    ✉ [benjaminukaimo@gmail.com](mailto:benjaminukaimo@gmail.com)  
    📞 +234 706 719 3071  
    🔗 [LinkedIn](https://www.linkedin.com/in/benjamin-uka-imo)  
    💻 [GitHub](https://github.com/uka-ben)  
    📺 [YouTube](https://youtube.com/@blackdatascience)  
    🌐 [Portfolio](https://benjaminuka.streamlit.app/)
    """)

    # --- RESUME DOWNLOAD BUTTON ---
    with open("Benjamin_Uka_Resume.pdf", "rb") as file:
        st.download_button(
            label="📄 Download My Resume",
            data=file,
            file_name="Benjamin_Uka_Resume.pdf",
            mime="application/pdf"
        )

# --- ABOUT ME ---
st.markdown("## 🧠 Qualification Summary")
st.markdown("""
Dynamic and solutions-oriented AI & Data Engineer with robust experience in machine learning, data engineering, 
MLOps, and cloud-native analytics. Proficient in designing scalable pipelines, deploying production-grade ML systems, 
and leveraging cutting-edge AI technologies including GNNs, Transformers, RL, and LLMs. Strong background in statistical 
modeling, big data processing, and decision intelligence.
""")

# --- TECHNICAL SKILLS ---
st.markdown("## 🛠️ Technical Skills")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Machine Learning & AI")
    st.markdown("""
    - Time Series (TFT, Prophet), Deep RL, Graph Neural Networks  
    - NLP: LLM, RAG, Chatbots  
    - AutoML: AutoGluon, H2O.ai, PyCaret  
    - Quantum ML (Basic)
    """)

with col2:
    st.markdown("### Programming & Toolkits")
    st.markdown("""
    - Python, SQL, PySpark, Spark SQL  
    - Numpy, Pandas, scikit-learn  
    - TensorFlow, PyTorch, Hugging Face, RLlib  
    - Stable-Baselines3, Gymnasium  
    - Data Viz: Matplotlib, Seaborn, Streamlit, Power BI, Looker  
    """)

with col3:
    st.markdown("### Data Engineering & Cloud MLOps")
    st.markdown("""
    - Databricks: LakeFlow, DLT, AutoLoader, CDC, SCD2  
    - Big Data: Structured Streaming, Delta Lake, ETL Pipelines  
    - GCP: BigQuery, Dataflow  
    - Azure Databricks  
    - CI/CD, MLflow, Airflow, Flask APIs  
    - Data Warehousing: Databricks SQL, BigQuery
    """)

# --- EXPERIENCE ---
st.markdown("## 💼 Professional Experience")

st.markdown("""
**AI & Data Engineer**  
*Miracle Health Recruitment, UK (Remote | Nov 2024 – Feb 2025)*  
- Worked with team to ensure seamless integration of data-driven solutions across operations.  
- Built streaming data pipelines using Delta Live Tables and AutoLoader.  

**Junior Data Scientist**  
*Baknance Technology (Remote | Feb 2023 – Apr 2024)*  
- Developed ML systems using Databricks ML Runtime for real-time inference.  
- Built a RAG-based chatbot using Hugging Face + OpenAI integrated with Databricks pipelines.  
- Designed end-to-end ETL pipelines for ingesting & transforming financial data.  
- Deployed predictive financial models for business forecasting optimization.  
- Implemented Delta Lake workflows enabling continuous model training & monitoring.  
""")

# --- PROJECTS ---
st.markdown("## 🚀 Projects")

st.markdown("### 📈 Finance & Risk Modeling")
st.markdown("""
- [Stock Market Analysis](https://timetion.streamlit.app/): Built a financial system using void anti-symmetric pattern detection.  
- Portfolio Optimization System: Designed a DRL-based solution for asset allocation.  
- Created a sixfold trading model (ensemble + DL + DRL + QNN + LSTM + Time Series).  
""")

st.markdown("### 🤖 AI & NLP Consultancy / Chatbots")
st.markdown("""
- [benGPT](https://benhealthcare.streamlit.app/): LLM-powered chatbot using Hugging Face + OpenAI.  
- Built multimodal healthcare diagnostic tool, improving prediction accuracy by 35%.  
- Designed conversational chatbots for real-time customer engagement.  
""")

st.markdown("### 🔐 Anomaly Detection Systems")
st.markdown("""
- Built fraud detection & risk modeling system using Databricks + DRL + GNN + Transformers.  
- Designed real-time anomaly detection for e-commerce & healthcare claims.  
- Built AI-driven cybersecurity threat detection with traffic + user behavior analytics.  
""")

st.markdown("### ♻️ Predictive & Optimization Systems")
st.markdown("""
- [CO₂ Optimizer](https://ben-co2optimization.streamlit.app/): DRL for global carbon emission policy modeling.  
- [ML Analytics Suite](https://benjitable-ds.streamlit.app/): Unified platform for anomaly detection, clustering, regression, classification.  
- IoT Predictive Maintenance: RL-powered system for industrial failure prevention.  
""")

st.markdown("### 🕹️ Robotics & Game AI")
st.markdown("""
- Trained custom RL agents (PPO, SAC) in robotics control environments.  
- Developed multi-agent reinforcement learning systems for autonomous decision-making.  
""")

# --- EDUCATION & CERTIFICATIONS ---
st.markdown("## 🎓 Education & Certifications")
st.markdown("""
- **B.Sc. in Accountancy**, Imo State University, Nigeria (2016)  
- **National Diploma in Accountancy**, Imo State Polytechnic, Nigeria (2012)  
- Certifications:  
  - Databricks Generative AI  
  - Neural Networks (Simplilearn)  
  - Deep Reinforcement Learning (Hugging Face – In Progress)  
""")

# --- COMMUNITY ---
st.markdown("## 🌍 Community & Open Source")
st.markdown("""
- 📺 [YouTube: Black Data Science](https://youtube.com/@blackdatascience): Tutorials on AI, ML & Data Engineering.  
- 💻 [GitHub](https://github.com/uka-ben): Open-source contributions to AI/ML.  
""")

# --- CONTACT ---
st.markdown("## 📬 Get in Touch")
st.markdown("""
✉ [benjaminukaimo@gmail.com](mailto:benjaminukaimo@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/benjamin-uka-imo)  
💻 [GitHub](https://github.com/uka-ben)  
📺 [YouTube](https://youtube.com/@blackdatascience)  
🌐 [Portfolio](https://benjaminuka.streamlit.app/)  
""")

st.success("Thank you for visiting my AI & Data Engineering Portfolio! Let's collaborate or innovate together 🚀")