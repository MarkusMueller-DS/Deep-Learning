# imports
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt

# 1 Load Data
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# frist 8 columns are the features
# the 9th column is the label
X = dataset[:, :8]
y = dataset[:, -1]

# data needs to be converted to PyTorch tensors
# numpy is 64-bit floating point but PyTorch operates on
# 32-bit floating point
# reshape y to make every label its own tensor
# PyTorch prefers  n x 1 matrix over n-vectors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1,1)

# Split data into training and test sets
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]

# 2 Define Model
# a more verbose way of defining a model through a python class
class PimaClassifier(nn.Module):
    # define the layout of the model in the constructer
    # so that it will be directly created when an
    # instance of the class is created
    def __init__(self):
        # call the parents class constructor to bootstrap model
        super().__init__()
        # define the layers and activation functions after each layer
        # Input: 8, Output: 12
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        # Input: 12 (the output from before), Output: 8 
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        #etc.
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()
    
    # tell PyTorch how to produce the output y Tensor
    # given the input X
    # simple forward pass
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

model= PimaClassifier()
print("Python class:", model)

# 3 Define Loss Function and Optimizers
loss_fn = nn.BCELoss()

# optimizer
# model.parameters() -> tells the optimzer what to optimize
# the weights and bias of each layer
# lr -> leraning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 4 Run Training Loop
# simple training loop
# number of epochs to run
n_epochs = 50
# size of each batch
batch_size = 10
batches_per_epoch = len(Xtrain) // batch_size

# lists to collect statistics
train_loss = []
train_acc = []
test_acc = []

for epoch in range(n_epochs):
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            start = i * batch_size
            # take a batch
            Xbatch = Xtrain[start:start+batch_size]
            ybatch = ytrain[start:start+batch_size]

            # forward pass
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            # calculate accuracy
            acc = (y_pred.round() == ybatch).float().mean()

            # store metrics
            train_loss.append(float(loss))
            train_acc.append(float(acc))

            # backward pass
            # set every gradient to zero to add the new ones based on the new loss
            optimizer.zero_grad()
            # run backpropagation
            loss.backward()

            # update weights in the neural net (based on gradient descent)
            optimizer.step()

            # print progress
            #print(f"epoch: {epoch}, step: {i}, loss: {loss}, accuracy {acc}")
            bar.set_postfix(
                loss=float(loss),
                acc=f"{float(acc)*100:.2f}%"
            )
        
    # evaluate model atend of epoch
    y_pred = model(Xtest)
    acc = (y_pred.round() == ytest).float().mean()
    test_acc.append(float(acc))
    print(f'End of {epoch}, accuracy: {acc}')
