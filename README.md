
# **Churn Prediction using Random Forest**  
This repository contains a **machine learning model** that predicts customer churn using a **Random Forest Classifier**. The project includes **data preprocessing, model training, evaluation, and feature importance analysis** to help businesses identify customers likely to leave.  

## **Features**  
✅ **Data Preprocessing:** Handles missing values, encodes categorical variables, and prepares data for training.  
✅ **Random Forest Model:** Trains a classifier to predict churn with high accuracy.  
✅ **Model Evaluation:** Includes accuracy, classification report, and confusion matrix.  
✅ **Feature Importance Analysis:** Visualizes key features affecting churn predictions.  
✅ **Deployment-Ready:** Easily adaptable for API integration or real-time prediction.  

## **Installation & Usage**  
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/your-username/churn-prediction.git
   cd churn-prediction
   ```
2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the script:**  
   ```bash
   python churn_prediction.py
   ```

## **Dataset Details**  
The dataset should be a CSV file (`data.csv`) containing customer attributes and a **churn column** that indicates whether a customer has left.  

### **Example Structure:**  
| CustomerID | Age | Subscription_Length | Monthly_Charge | Total_Spend | Churn |  
|------------|-----|---------------------|---------------|------------|--------|  
| 1001       | 32  | 12                  | 25.99         | 311.88     | Yes    |  
| 1002       | 45  | 24                  | 39.99         | 959.76     | No     |  

- The `Churn` column is converted to a binary label (1 = Churn, 0 = Not Churn).  
- Ensure all categorical features are encoded before training.  

## **Model Deployment**  
This model can be deployed as an API using **FastAPI or Flask**. Example steps:  

1. Install **FastAPI** and **Uvicorn**:  
   ```bash
   pip install fastapi uvicorn
   ```
2. Create an API (`api.py`):  
   ```python
   from fastapi import FastAPI
   import joblib
   import pandas as pd

   app = FastAPI()
   model = joblib.load("random_forest_model.pkl")

   @app.post("/predict")
   def predict(data: dict):
       df = pd.DataFrame([data])
       prediction = model.predict(df)
       return {"churn_prediction": int(prediction[0])}
   ```
3. Run the API:  
   ```bash
   uvicorn api:app --reload
   ```
4. Send a request using **Postman** or **cURL**:  
   ```bash
   curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'Content-Type: application/json' -d '{"Age": 30, "Subscription_Length": 12, "Monthly_Charge": 19.99, "Total_Spend": 239.88}'
   ```

## **Technologies needed**  
🔹 **Python** (Pandas, NumPy, Scikit-learn)  
🔹 **Visualization** (Matplotlib, Seaborn)  
🔹 **FastAPI/Flask** (for deployment)  
🔹 **Random Forest Classifier** (for churn prediction)  

---

🚀
