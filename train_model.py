import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# ================== Load Dataset ==================
data = pd.read_csv("dataset/data.csv")

print("Dataset shape:", data.shape)
print(data.head())

# ================== Features & Labels ==================
X = data.drop("label", axis=1)
y = data["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Classes:", label_encoder.classes_)

# ================== Train-Test Split ==================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ================== Model Training ==================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ================== Predictions ==================
y_pred = model.predict(X_test)

# ================== Evaluation ==================
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Accuracy:", round(accuracy, 4))

print("\n📊 Classification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[str(c) for c in label_encoder.classes_]
    )
)

# ================== Additional Metrics ==================
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n📌 Additional Metrics:")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

# ================== Confusion Matrix ==================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ================== Feature Importance ==================
feature_names = X.columns
importances = model.feature_importances_

plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

# ================== Save Model ==================
joblib.dump(model, "sign_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n✅ Model and label encoder saved successfully")
