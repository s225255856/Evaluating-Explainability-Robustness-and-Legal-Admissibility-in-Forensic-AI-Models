import pandas as pd
import numpy as np
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

#Basic evaluation
acs = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy:", acs)
print("Confusion Matrix:\n", cm)
print("Classification Report: \n", cr)