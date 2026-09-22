import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

#Load models 
rf = joblib.load("models/baseline_model.pkl")

#Load test data
X_test = pd.read_csv("preprocessed2/X_test.csv", index_col=0)
y_test = pd.read_csv("preprocessed2/y_test.csv", index_col=0).squeeze()

#Numpy
X_test_np = X_test.values.astype(np.float32)
y_test_np = y_test.values.astype(np.int64)

#Load substitute PyTorch model
class SubstituteMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_test_np.shape[1]
sub_model = SubstituteMLP(input_dim)
sub_model.load_state_dict(torch.load("models/substitute_pytorch.pth"))
sub_model.eval()

#Wrap sub model in PyTorchClassifier for ART
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(sub_model.parameters(), lr=0.001)

sub_clf = PyTorchClassifier(
    model=sub_model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(input_dim,),
    nb_classes=2,
)

#Clean accuracy
rf_clean_acc = accuracy_score(y_test_np, rf.predict(X_test))
sub_clean_acc = accuracy_score(y_test_np, np.argmax(sub_clf.predict(X_test_np), axis=1))
print("RandomForest clean accuracy:", rf_clean_acc)
print("Subtitute clean accuracy:", sub_clean_acc)

#Apply FGSM on subtitute
fgsm = FastGradientMethod(
    estimator=sub_clf,
    eps=0.1
)
X_adv_sub = fgsm.generate(X_test_np)

#Evaluate on substitute after the attack
sub_adv_acc = accuracy_score(y_test_np, np.argmax(sub_clf.predict(X_adv_sub), axis=1))
print("Subtitute adversarial accuracy:", sub_adv_acc)

#Evaluate after transfering to RandomForest
X_adv_df = pd.DataFrame(X_adv_sub, columns=X_test.columns)
rf_adv_acc = accuracy_score(y_test_np, rf.predict(X_adv_df))
print("RandomForest accuracy after the transfer:", rf_adv_acc)
print("RandomForest robustness drop:", rf_clean_acc - rf_adv_acc)