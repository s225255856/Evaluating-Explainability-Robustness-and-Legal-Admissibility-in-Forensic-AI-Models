import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack, BoundaryAttack, SquareAttack

#Load model
model = joblib.load("models/baseline_model.pkl")

#Load data
X_test = pd.read_csv("preprocessed2/X_test.csv", index_col=0)
y_test = pd.read_csv("preprocessed2/y_test.csv", index_col=0).squeeze()

#Numpy arrays for ART
X_test_np = X_test.values.astype(np.float32)
y_test_np = y_test.values.astype(np.int64)

#Wrap model with ART
classifier = SklearnClassifier(model=model)

#Evaluate clean accuracy
y_pred_clean = model.predict(X_test_np)
clean_acc = accuracy_score(y_test_np, y_pred_clean)

print("Clean accuracy:", clean_acc)


#Square Attack (black-box)
square = SquareAttack(
    estimator=classifier,
    eps=0.1
)
X_test_adv_sq = square.generate(X_test_np)

y_pred_sq = model.predict(X_test_adv_sq)
sq_acc = accuracy_score(y_test_np, y_pred_sq)

print("Square Attack adversarial accuracy:", sq_acc)
print("Square Attack accuracy drop:", clean_acc - sq_acc)

#Summary
print("\n---Robustness Summary---")
print(f"Clean Accuracy: {clean_acc:.4f}")
print(f"Square Attack Accuracy: {sq_acc:.4f} (Drop: {clean_acc - sq_acc:.4f})")
