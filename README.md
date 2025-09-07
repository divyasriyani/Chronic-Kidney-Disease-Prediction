# Chronic-Kidney-Disease-Prediction
Machine learning model to predict chronic kidney disease using patient medical data.  A data science project that analyzes health parameters to predict the likelihood of CKD. Chronic Kidney Disease prediction system using Python, scikit-learn, and machine learning algorithms.
# Chronic Kidney Disease Detection

This project predicts the likelihood of Chronic Kidney Disease (CKD) using **machine learning algorithms**.  
It analyzes patient medical data (blood pressure, blood glucose, serum creatinine, hemoglobin, etc.) to identify early signs of CKD and assist healthcare professionals in decision-making.


## 📌 Features
- Data preprocessing and cleaning  
- Exploratory Data Analysis (EDA)  
- Model training with multiple ML algorithms  
- Performance evaluation and comparison  
- Prediction of CKD risk with high accuracy  


## 📂 Dataset
- **Name**: `chronic_kidney_disease.csv`  
- **Size**: ~400 patient records  
- **Attributes**: 24 medical features including age, blood pressure, blood glucose, hemoglobin, albumin, packed cell volume, etc.  
- **Target Variable**: Presence or absence of CKD (`yes` / `no`)  

*(Dataset source: UCI Machine Learning Repository or institutional dataset used in project)*  


## 🛠️ Technologies Used
- **Programming Language**: Python 3.7+  
- **Libraries**:  
  - Data Handling: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Machine Learning: `scikit-learn` (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes)  
  - Deep Learning: `TensorFlow` / `Keras` (for ANN models)  
  - Notebook: `Jupyter Notebook`  


## ⚙️ Implementation Steps
1. Data Collection → `chronic_kidney_disease.csv`  
2. Data Preprocessing → Handle missing values, clean typos, encode categorical variables  
3. Exploratory Data Analysis → Correlation heatmap, feature distributions, missing value analysis  
4. Model Building → Logistic Regression, SVM, Decision Tree, Random Forest, KNN, ANN, Naive Bayes  
5. Model Evaluation → Accuracy, Precision, Recall, F1-score, AUC-ROC  
6. Results Visualization → Accuracy comparison plots  
7. Deployment → (Optional: Flask/Streamlit for prediction UI)  


## 📊 Results
- **Logistic Regression**: 98.75% accuracy  
- **Support Vector Machine (SVC)**: 98.75% accuracy  
- **Decision Tree**: 97.5% accuracy  
- **Random Forest**: 96.25% accuracy  
- **Naive Bayes**: 93.75% accuracy  
- **KNN**: 85% accuracy  
- **ANN**: ~90% accuracy  

✅ The best performing models were **Logistic Regression** and **SVC**, achieving **98.75% accuracy**.  


## 🚀 Future Scope
- Integration with **Electronic Health Records (EHR)**  
- **Personalized medicine** and treatment recommendations  
- **Mobile app / IoT integration** for real-time patient monitoring  
- **Explainable AI** for better medical trust & transparency  
- Expansion with **genomic & proteomic data**  

