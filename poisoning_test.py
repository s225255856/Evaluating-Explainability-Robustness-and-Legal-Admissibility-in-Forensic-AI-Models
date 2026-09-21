import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Load clean data
X_train = pd.read_csv("preprocessed2/X_train.csv", index_col=0)
y_train = pd.read_csv("preprocessed2/y_train.csv", index_col=0).squeeze()
X_test = pd.read_csv("preprocessed2/X_test.csv", index_col=0)
y_test = pd.read_csv("preprocessed2/y_test.csv", index_col=0).squeeze()

#Baseline RandomForest
rf_clean = joblib.load("models/baseline_model.pkl")
rf_clean_acc = accuracy_score(y_test, rf_clean.predict(X_test))
print("Baseline RandomForest accuracy:", rf_clean_acc)

#Poisoning the data
for p in [0.01, 0.05, 0.10]:
    y_poisoned = y_train.copy()
    n_flip = int(p * len(y_poisoned))
    flip_idx = np.random.choice(len(y_poisoned), n_flip, replace=False)
    y_poisoned.iloc[flip_idx] = 1 - y_poisoned.iloc[flip_idx]

    rf_poisoned = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_poisoned.fit(X_train, y_poisoned)
    poisoned_acc = accuracy_score(y_test, rf_poisoned.predict(X_test))

    print(f"\nPoisoning rate: {p*100:.0f}%")
    print("Poisoned RandomForest accuracy:", poisoned_acc)
    print("Accuracy drop:", rf_clean_acc - poisoned_acc)
