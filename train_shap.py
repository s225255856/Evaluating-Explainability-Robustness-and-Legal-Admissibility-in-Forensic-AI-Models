import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Load preprocessed HDFS event occurence matrix
events = pd.read_csv("HDFS_v1/preprocessed/Event_occurrence_matrix.csv")

#Load anomaly labels
labels = pd.read_csv("HDFS_v1/preprocessed/anomaly_label.csv")

#Ensure BlockId column name's match
assert "BlockId" in events.columns
assert "BlockId" in labels.columns

#Merge features and labels
df = events.merge(labels, on="BlockId")

#Drop BlockId that are not useful for ML
df = df.drop(columns=["BlockId"])

#Use Label_y as the correct label column
df = df.rename(columns={"Label_y": "Label"})

#Drop unused columns
df = df.drop(columns=["Label_x", "Type"])

#Split into features (X) and labels (y)
X = df.drop("Label", axis=1)
y = df["Label"]

#Convert string labels to integers as SHAP takes integers
label_map = {"Normal": 0, "Anomaly": 1}
df["Label"] = df["Label"].map(label_map)

#Training and testing data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Training baseline model
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

#Predictions
y_pred = model.predict(X_test)

#Use training data as background for SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

#anomaly
shap_anomaly = shap_values[1]
shap_anomaly = shap_anomaly[:, :X_test.shape[1]]

#Global feature importance (for anomaly class)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_anomaly, X_test, show=False)
plt.tight_layout()
plt.savefig("results/shap_global_summary_anomaly.png")
plt.close()

#Local explaination for a single anomaly sample
idx = y_test[y_test == 1].index[0]
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][X_test.index.get_loc(idx)],
    X_test.iloc[X_test.index.get_loc(idx)],
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