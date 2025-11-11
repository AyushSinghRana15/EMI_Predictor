# EMI_Predictor
Overview

This project implements two machine learning models for the financial domain:

	•	EMI Eligibility Classification: Predicts whether a customer is eligible for an EMI.
	•	Maximum EMI Amount Regression: Predicts the maximum EMI amount a customer can afford.
  
Both models are built using real-world financial data, prepared with robust feature engineering, and trained using scalable ML pipelines. The models are tracked and registered with MLflow for reproducibility and deployment readiness.

Methodology and Architecture

	•	Data preprocessing includes handling missing values, feature encoding, and scaling.
	•	Classification model uses XGBoost classifier inside a sklearn pipeline.
	•	Regression model uses Random Forest regressor with similar preprocessing.
	•	MLflow is used for experiment tracking, model versioning, and deployment management.
	•	Streamlit is used to provide a multi-page web app interface with real-time prediction capabilities.
  
Exploratory Data Analysis (EDA)

	•	Conducted detailed feature distributions, correlation analysis, and outlier detection.
	•	Visualized relationships between customer financial attributes and EMI eligibility.
	•	Identified key drivers of EMI eligibility such as credit score, income, and existing debts.
	•	Visualizations include histograms, boxplots, heatmaps, and scatter plots.

Model Performance Analysis

	•	Classification metrics:
	•	Accuracy, Precision, Recall, F1 Score, ROC-AUC
	•	Regression metrics:
	•	RMSE, MAE, R2 Score
	•	MLflow experiments record hyperparameters, metrics, and artifacts for each run.
	•	Comparison between model versions highlights improvements via tuning and feature selection.
  
Business Impact and Recommendations

	•	Enables financial institutions to automate EMI eligibility assessment, reducing manual workload.
	•	Improves customer experience through faster decisions and personalized EMI offerings.
	•	Insights into factors affecting eligibility help shape targeted product designs.
	•	Suggests incorporating dynamic credit scoring and expense tracking for better model accuracy.
	•	Recommends continuous monitoring and retraining with new customer data for sustained performance.
  
Usage and Deployment

	•	Models are deployed as a multi-page Streamlit app accessible via Streamlit Cloud.
	•	Real-time classification and regression predictions supported with user-friendly input forms.
	•	MLflow Model Registry manages model versions and production transitions.
	•	Comprehensive codebase available on GitHub for easy cloning, testing, and extension.
  
Getting Started

	•	Clone this repository.
	•	Install requirements:  pip install -r requirements.txt 
  • Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate

  •  Set up MLflow tracking URI (if using local tracking)
      import mlflow
      mlflow.set_tracking_uri("file:///absolute/path/to/mlruns")

  •  streamlit run app.py
  
  
