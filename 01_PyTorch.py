import torch
from torch import nn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Preparing and Loading Data

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

# print(X[:10], y[:10], len(X), len(y))

### Splitting data into training and testing data

#Create a train/test split
train_split = int(0.8*len(X))
#print(train_split)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#Visualise data

def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test, 
                     test_labels=y_test,
                     predictions = None)-> None:
    """
    Plots training data, test data, and compares predictions.
    """
    plt.figure(figsize = (10,7))

    plt.scatter(train_data, train_labels, c='b',
                 s=4, label = "Training data")
    plt.scatter(test_data, test_labels, c='g',
                 s=4, label = "Testing data")
    
    if predictions is not None:
        #Plot predictions if they exist
        plt.scatter(test_data, predictions, c = 'r',
        s=4, label = "Predictions")

    plt.legend(prop={"size":14});

    plt.show()

#plot_predictions()

# Build a model

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.linear_layer(x)
    
torch.manual_seed(42)
model_0=LinearRegressionModel()

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr = 0.01)

torch.manual_seed(42)
epochs = 200

for epoch in range (epochs):
    model_0.train()

    #1: Forward Pass
    y_pred = model_0(X_train)

    #2: Calculate the loss
    loss = loss_fn(y_pred, y_train)

    #3: Optimizer zero grad
    optimizer.zero_grad()

    #4: Perform back propogaton
    loss.backward()

    #5: Optimizer step
    optimizer.step()

    ###Testing 
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)

        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")

#Save and Load a model

from pathlib import Path

MODEL_PATH = Path('C:\\Users\\matty\\OneDrive\\Desktop\\DeepLearning\\models')
MODEL_PATH.mkdir(parents = True, exist_ok = True)

MODEL_NAME = 'LinearRegressionModel0.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(model_0.state_dict(), f = MODEL_SAVE_PATH)

loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f = MODEL_SAVE_PATH))



