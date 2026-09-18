import pandas as pd
import numpy as np
import joblib
from preprocess import load_and_preprocess
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Load preprocessed data
X_train = pd.read_csv("preprocessed2/X_train.csv", index_col=0)
X_test = pd.read_csv("preprocessed2/X_test.csv", index_col=0)
y_train = pd.read_csv("preprocessed2/y_train.csv", index_col=0).squeeze()
y_test = pd.read_csv("preprocessed2/y_test.csv", index_col=0).squeeze()

#test
print("Unique y_train:", y_train.unique())
print("Unique y_test:", y_test.unique())

#Training baseline model
model = joblib.load("models/baseline_model.pkl")

#Predictions
y_pred = model.predict(X_test)

#Prepare LIME explainer
explainer_lime = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Normal", "Anomaly"],
    discretize_continuous=True,
    mode="classification"

)

#Sampling 500 rows for quicker result
X_shap = X_test.sample(500, random_state=42)

#Use training data as background for SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

print("RAW SHAP VALUES SHAPE:", shap_values.values.shape)

#Anomaly class = index 1
shap_anomaly = shap_values.values[:, :, 1]

#Safety check
print("SHAP:", shap_anomaly.shape)
print("X_test:", X_test.shape)

#Global feature importance (for anomaly class)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_anomaly, X_test, show=False)
plt.tight_layout()
plt.savefig("results/shap_global_summary_anomaly.png")
plt.close()

#Pick one anomaly sample for LIME
anomaly_idx = y_test[y_test == 1].index[0]
instance = X_test.loc[anomaly_idx].values

#Generate LIME explanation
exp = explainer_lime.explain_instance(
    instance,
    model.predict_proba,
    num_features = 10
)
with open("results/lime_local_anomaly_example.txt", "w") as f:  #Save text version
    for feature, weight in exp.as_list():
        f.write(f"{feature}: {weight}\n")
exp.save_to_file("results/lime_local_anomaly_example.html")  #Save interactive HTML version

#Local explaination for a single anomaly sample
idx = X_shap.index[0]
shap.force_plot(
    explainer.expected_value[1],
    shap_anomaly[0],
    X_shap.iloc[0],
    matplotlib=True
)
plt.savefig("results/shap_local_anomaly_example.png")
plt.close()

#Basic evaluation
acs = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy:", acs)
print("Confusion Matrix:\n", cm)
print("Classification Report: \n", cr)