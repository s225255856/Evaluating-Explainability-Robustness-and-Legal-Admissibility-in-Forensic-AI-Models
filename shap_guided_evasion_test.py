import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import accuracy_score

#Load model and data
rf = joblib.load("models/baseline_model.pkl")
X_test = pd.read_csv("preprocessed2/X_test.csv", index_col=0)
y_test = pd.read_csv("preprocessed2/y_test.csv", index_col=0).squeeze()
X_train = pd.read_csv("preprocessed2/X_train.csv", index_col=0)
y_train = pd.read_csv("preprocessed2/y_train.csv", index_col=0).squeeze()

#SHAP TreeExplainer
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)[1] #Class 1 anomaly

#Mean absolute SHAP values
shap_importance = np.abs(shap_values).mean(axis=0)

#Top k features to attack
k = 10
top_features = np.argsort(shap_importance)[-k:]
feature_names = X_test.columns[top_features]

print("Top SHAP features:", feature_names.tolist())

#Normal class statistic
normal = X_train[y_train == 0]
normal_median = normal.median()
normal_min = normal.min()
normal_max = normal.max()

# #Extract split thresholds for each feature
# thresholds = {f: [] for f in feature_names}

# for tree in rf.estimators_:
#     tree_ = tree.tree_
#     for node in range(tree_.node_count):
#         feature = tree_.feature[node]
#         if feature != -1:
#             fname = X_test.columns[feature]
#             if fname in feature_names:
#                 thresholds[fname].append(tree_.threshold[node])

# #Median values threshold per feature
# median_thresholds = {f: np.median(thresholds[f]) for f in feature_names}

#Create adversarial copy
X_adv = X_test.copy().astype(float)

#Only attack anomaly samples
anomaly_mask = (y_test == 1)

#Apply SHAP-guided perturbation
for f in feature_names:
    vals = X_adv.loc[anomaly_mask, f]
    shifted = vals * 0.1 + normal_median[f] * 0.9
    shifted = shifted.clip(lower=normal_min[f], upper=normal_max[f])
    X_adv.loc[anomaly_mask, f] = shifted

#Evaluate RandomForest on adversarial samples
rf_clean_acc = accuracy_score(y_test, rf.predict(X_test))
rf_adv_acc = accuracy_score(y_test, rf.predict(X_adv))

print("RandomForest clean accuracy:", rf_clean_acc)
print("RandomForest adversarial accuracy:", rf_adv_acc)
print("Robustness drop:", rf_clean_acc - rf_adv_acc)

#Focused: anomaly detection performance
y_pred_clean = rf.predict(X_test)
y_pred_adv = rf.predict(X_adv)

anomaly_idx = (y_test == 1)

clean_anomaly_acc = accuracy_score(y_test[anomaly_idx], y_pred_clean[anomaly_idx])
adv_anomaly_acc = accuracy_score(y_test[anomaly_idx], y_pred_adv[anomaly_idx])

print("\nClean anomaly detection accuracy:", clean_anomaly_acc)
print("Adversarial anomaly detection accuracy:", adv_anomaly_acc)
print("Anomaly robustness drop:", clean_anomaly_acc - adv_anomaly_acc)