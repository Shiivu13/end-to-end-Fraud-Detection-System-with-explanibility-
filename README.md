# end-to-end-Fraud-Detection-System-with-explanibility-

Fraud Detection System (End-to-End ML Project)
A production-style fraud detection system built step by step with a strong focus on real-world constraints such as imbalanced data, explainability, decision thresholds, and deployment reliability.
This project emphasizes engineering decisions and system ownership, not just model accuracy.

🎯 Project Goals
The primary goal of this project was to design and implement a complete ML system, not just train a model.
Key objectives:
Handle extremely imbalanced fraud data
Avoid misleading metrics (accuracy)
Use business-driven decision thresholds
Make predictions explainable and auditable
Deploy the model as a reliable API
Demonstrate the system via a simple frontend
Ensure reproducibility using Docker

🧩 Planned Architecture
Copy code

Input Transaction
        ↓
Data Preprocessing
        ↓
ML Model (LightGBM)
        ↓
Probability Score
        ↓
Business Threshold Logic
        ↓
Explainability (SHAP)
        ↓
FastAPI Backend
        ↓
Dockerized Service
        ↓
Streamlit Frontend Demo

🛠 Tech Stack
Machine Learning
Python
LightGBM
Scikit-learn
Pandas, NumPy
Explainability
SHAP (feature attribution)
Backend
FastAPI
REST APIs
Pydantic validation
Deployment
Docker
Linux-based containers
System-level dependency handling
Frontend
Streamlit (demo interface)

🧪 Key Engineering Decisions

1️⃣ Metrics over Accuracy
Fraud data is highly imbalanced, so accuracy was avoided.
Instead:
Precision–Recall Curve
PR-AUC
Threshold-based evaluation

2️⃣ Business-Driven Threshold
Instead of using a default probability threshold (0.5), a custom threshold was selected based on recall-precision trade-offs relevant to fraud detection.

3️⃣ Explainability First
Each prediction is accompanied by:
SHAP feature attributions
A human-readable explanation summarizing why the transaction was flagged

4️⃣ Robust API Design
The API:
Accepts partial feature inputs
Automatically fills missing features
Prevents inference crashes
Returns transparent prediction metadata

5️⃣ Production-Style Deployment
Dockerized the service
Resolved runtime issues such as OpenMP (libgomp) dependency required by LightGBM
Ensured reproducible builds

🛠 Build Log (Day-wise Development)

Day 1 – Problem Understanding & Baseline
Explored fraud dataset characteristics
Identified extreme class imbalance
Trained baseline model

Day 2 – Data Splitting & Evaluation Strategy
Train / validation split
Rejected accuracy as a metric
Adopted Precision–Recall based evaluation

Day 3 – Explainability Layer
Integrated SHAP for feature attribution
Designed human-readable explanations

Day 4 – Backend API
Built FastAPI backend
Defined request schemas
Added error handling and validation

Day 5 – Threshold Optimization
Performed PR analysis
Selected business-driven fraud threshold
Separated scoring from decision logic

Day 6 – Dockerization
Containerized the application
Debugged runtime issues inside Docker
Added system-level dependencies for LightGBM

Day 7 – Stability & Compatibility
Handled missing input features safely
Improved API robustness
Ensured model compatibility with frontend inputs

Day 8 – Frontend Demo
Built Streamlit UI
Connected frontend to backend API
Enabled end-to-end interaction for non-technical users
🖥 Running the Project
Backend (Docker)
Bash
Copy code
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
API available at:
Copy code

[http://localhost:8000](https://demofrauddetection.streamlit.app/)
Frontend (Streamlit)


📌 Notes
Feature values are PCA-transformed and used for demonstration purposes.
The project focuses on system behavior and design, not real financial decision-making.
The architecture supports future extensions such as monitoring and GenAI-based explanation layers.

🚀 Future Improvements
Full feature auto-loading in frontend
Model monitoring and drift detection
GenAI explanation layer
Cloud deployment

👤 Author
Shivansh Tripathi
Applied Machine Learning Practitioner
Focused on building real, explainable, production-style ML systems.
