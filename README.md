# Prediction-of-Chronic-Kidney-Disease-with-ANN
Machine Learning Project

# Chronic Kidney Disease (CKD) Prediction using Artificial Neural Networks

This project presents a Machine Learning solution for early prediction of Chronic Kidney Disease (CKD) using an Artificial Neural Network (ANN). CKD is a serious condition that affects kidney function over time. Early detection can significantly improve treatment outcomes and patient health.


📌 Project Overview

Objective: To build and evaluate an ANN model that accurately predicts the presence of CKD based on patient health data.

Dataset: The model is trained on a medical dataset containing various clinical attributes such as blood pressure, specific gravity, albumin, blood glucose levels, and more.

Tools Used: Python, Jupyter Notebook, TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.


🧠 Machine Learning Model

Algorithm: Artificial Neural Network (ANN)

Architecture:

  Input Layer: Accepts normalized patient features.

  Hidden Layers: Dense layers with ReLU activation.

  Output Layer: Binary classification with sigmoid activation (CKD or not).

Loss Function: Binary Crossentropy

Optimizer: Adam

Evaluation Metrics: Accuracy, Confusion Matrix, Classification Report, ROC-AUC.


📊 Dataset Features

The dataset includes clinical and physiological indicators such as:

Age, Blood Pressure, Hemoglobin, Serum Creatinine

Specific Gravity, Albumin, Sugar levels

Red Blood Cells (RBC), White Blood Cells (WBC)

Hypertension, Diabetes Mellitus, and other relevant features

-- Missing values were handled with appropriate imputation techniques, and data normalization was applied for consistent training.


🚀 Project Workflow

- Data Loading and Exploration

- Data Cleaning & Preprocessing 

- Feature Encoding and Normalization

- Model Building using ANN

- Training & Evaluation

- Performance Visualization


✅ Results

Model Accuracy: ~ 98% on test data.

Performance: High precision and recall, indicating the model's effectiveness in identifying CKD cases and minimizing false negatives.


🧪 Future Enhancements

- Deploy model using Flask or Streamlit for real-time predictions.

- Expand dataset for better generalization.

- Integrate additional diagnostic features or expert feedback.


👩‍⚕️ Use Cases

- Clinical Decision Support Systems

- Telehealth Diagnostics

- Public Health Screening Programs


📚 References

- UCI Machine Learning Repository (CKD dataset)

- TensorFlow

- Scikit-learn

