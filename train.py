import pandas as pd
import numpy as np
import joblib
from preprocess import load_and_preprocess, save_splits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Save splits once
save_splits()

#Load preprocessed data
X_train, X_test, y_train, y_test = load_and_preprocess()

#Training baseline model
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

#Save the model
joblib.dump(model, "models/baseline_model.pkl")

#Predictions
y_pred = model.predict(X_test)

#Basic evaluation
acs = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy:", acs)
print("Confusion Matrix:\n", cm)
print("Classification Report: \n", cr)