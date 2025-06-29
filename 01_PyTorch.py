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
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
        #Forward method to define the computation in the model
    def forward(self, x: torch.tensor) -> torch.Tensor:
        return self.weights *x + self.bias

### Checking the contents of our pytorch model

#Create a random seed

torch.manual_seed(42)

#Create an instance of the model
model_0 = LinearRegressionModel()

#Check parameters
#print(model_0.state_dict())

###Making predictions

with torch.inference_mode():
    y_preds = model_0(X_test)
#print(y_preds)
 
###Train Model

#Setup a loss function
loss_fn = nn.L1Loss()

#Setup Optimizer

optimizer = torch.optim.SGD(params = model_0.parameters(),
                            lr = 0.01)

#Building a training loop (and a testing loop)

epochs = 100

#0. Loop through Data

for epoch in range(epochs):
    # Set model to training mode
    model_0.train()

    #1. Forward pass
    y_pred = model_0(X_train)

    #2. Calculate Loss
    loss = loss_fn(y_pred, y_train)

    #3. Optimiser zero grad 
    optimizer.zero_grad()

    #4. Perform back propogation
    loss.backward()

    #5. Perform Grad Descent
    optimizer.step()

    ###Testing
    model_0.eval()
    with torch.inference_mode():
        #1. Forward Pass
        test_pred = model_0(X_test)

        #2. Calc Loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")

print(model_0.state_dict())
with torch.inference_mode():
    y_preds_new = model_0(X_test)
plot_predictions( predictions=y_preds_new)






 




        

    



