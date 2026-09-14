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

#Convert string labels to integers as SHAP takes integers
label_map = {"Normal": 0, "Anomaly": 1}
labels["Label"] = labels["Label"].map(label_map)

#Merge features and labels
df = events.merge(labels, on="BlockId")

#test
print("MERGED COLUMNS:", df.columns.tolist())

#Drop BlockId that are not useful for ML
df = df.drop(columns=["BlockId", "Label_x", "Type"])

#Use Label_y as the correct label column
df = df.rename(columns={"Label_y": "Label"})

#Split into features (X) and labels (y)
X = df.drop("Label", axis=1)
y = df["Label"]

#Training and testing data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#test
print("Unique y_train:", y_train.unique())
print("Unique y_test:", y_test.unique())

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

#Sampling 500 rows for quicker result
X_shap = X_test.sample(500, random_state=42)

#Use training data as background for SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

print("RAW SHAP VALUES SHAPE:", shap_values.values.shape)

# anomaly class = index 1
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