import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, SquareAttack

#Load model
model = joblib.load("baseline_model.pkl")

#Use data
label_map = {"Normal": 0, "Anomaly": 1}
y_test_int = y_test.map(label_map).astype(int)

#Numpy arrays for ART
X_test_np = X_test.values.astype(np.float32)
y_test_np = y_test_int.values.astype(np.int64)

#Wrap model with ART
classifier = SklearnClassifier(model=model)

#Evaluate clean accuracy
y_pred_clean = model.predict(X_test_np)
clean_acc = accuracy_score(y_test_np, y_pred_clean)

print("Clean accuracy:", clean_acc)

#FGSM Attack
fgsm = FastGradientMethod(
    estimator=classifier, 
    eps=0.1
)
X_test_adv_fgsm = fgsm.generate(X_test_np)

y_pred_fgsm = model.predict(X_test_adv_fgsm)
fgsm_acc = accuracy_score(y_test_np, y_pred_fgsm)

print("FGSM adversarial accuracy:", fgsm_acc)
print("FGSM accuracy drop:", clean_acc - fgsm_acc)

#PGD attack
pgd = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.1,
    eps_step=0.01,
    max_iter=20
)

X_test_adv_pgd = pgd.generate(X_test_np)

y_pred_pgd = model.predict(X_test_adv_pgd)
pgd_acc = accuracy_score(y_test_np, y_pred_pgd)

print("PGD adversarial accuracy:", pgd_acc)
print("PGD accuracy drop:", clean_acc - pgd_acc)

#Square Attack (black-box)
square = SquareAttack(
    estimator=classifier,
    eps=0.1
)
X_test_adv_sq = square.generate(X_test_np)
