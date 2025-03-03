import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset (replace with actual dataset path)
data = pd.read_csv('path/to/your/dataset.csv')

# Handle missing values (optional: you can replace with more sophisticated imputation if needed)
data.dropna(inplace=True)

# Encode categorical variables (except the target column)
data = pd.get_dummies(data, drop_first=True)

# Replace 'churn_column' with the actual churn column name
if 'churn_column' in data.columns:
    data['churn'] = data['churn_column'].apply(lambda x: 1 if x == 'yes' else 0)
    data.drop(columns=['churn_column'], inplace=True)
else:
    raise ValueError("Column 'churn_column' not found in dataset. Please check column names.")

# Split dataset into features (X) and target (y)
X = data.drop(columns=['churn'])
y = data['churn']

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("🔹 Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort features by importance
features = X.columns[indices]  # Ensure correct feature indexing

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
sns.barplot(x=importances[indices], y=features, palette="viridis")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
