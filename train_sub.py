import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

#Load model
rf_model = joblib.load("models/baseline_model.pkl")

#Load data
X_train = pd.read_csv("preprocessed2/X_train.csv", index_col=0)
y_train = pd.read_csv("preprocessed2/y_train.csv", index_col=0).squeeze()

#Numpy
X_train_np = X_train.values.astype(np.float32)
y_train_np = y_train.values.astype(np.int64)

#Use RandomForest predictions as labels for sub model
y_rf_pred = rf_model.predict(X_train_np).astype(np.int64)

#Create PyTorch multilayer perceptron (mlp) model to sub RandomForest
class SubstituteMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  #binary classification
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train_np.shape[1]
model = SubstituteMLP(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X_train_np)
y_tensor = torch.tensor(y_rf_pred)

#Train sub model
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

#Save the sub model
torch.save(model.state_dict(), "models/substitute_pytorch.pth")
print("Saved substitute model.")