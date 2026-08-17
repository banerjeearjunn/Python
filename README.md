# Logistic Regression using Gradient Descent vs Stochastic Gradient Descent vs Mini-Batch Gradient Descent


################## Classification Using Logistic Regression ##################

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from torch.utils.data import TensorDataset, DataLoader

###################################################################

# Check torch version & device
print(f"Torch version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

###################################################################

################## Labelled Dataset ##################

# Load the dataset
iris_data = sns.load_dataset("iris")
iris_data.head()

# Seperating features and labels
X = iris_data.drop("species", axis = 1)
y = iris_data["species"]

# Convert categorical labels to numerical ones
encoder = LabelEncoder()
y = encoder.fit_transform(y) # y is now a numpy array of integers


###################################################################

################## Setup ##################

p = X.shape[1] # number of features
c = len(np.unique(y)) # number of classes


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features = p, out_features = c)

    def forward(self, x: torch.tensor) -> torch.tensor:
        logit = self.linear(x)
        return logit


###################################################################

################## Train Test Split ##################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 111)

# Note that X_train & X_test are pandas dataframe and y_train & y_test are numpy arrays.
# Convert the sets into tensors
X_train = torch.tensor(np.array(X_train), dtype = torch.float32)
X_test = torch.tensor(np.array(X_test), dtype = torch.float32)
y_train = torch.tensor(np.array(y_train), dtype = torch.long)
y_test = torch.tensor(np.array(y_test), dtype = torch.long)

################## Data Loader ##################

# Create a unified dataset package
train_dataset = TensorDataset(X_train, y_train)


# 1. GD Loader: Batch size equal to the size of the whole dataset
gd_loader = DataLoader(train_dataset, batch_size = len(X_train), shuffle = False)

# 2. SGD Loader: Batch size is exactly 1 (shuffled every epoch)
sgd_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)

# 3. Mini-Batch Loader: Batch size is a small group (e.g. 16 or 32)
batch_size = 16
mini_batch_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


###################################################################

################## 1. Gradient Descent ##################

model_gd = LogisticRegression().to(device)
optimizer_gd = torch.optim.SGD(params = model_gd.parameters(), lr = 0.1, )
loss_function = nn.CrossEntropyLoss()

epochs = 100
epoch_count_gd = []
train_loss_values = []
test_loss_values = []

for epoch in range(epochs):
    epoch_count_gd.append(epoch)

    # Training
    model_gd.train()

    epoch_batch_loss = []

    for batch_X, batch_y in gd_loader:

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer_gd.zero_grad()
        y_train_pred = model_gd(batch_X)
        train_loss = loss_function(y_train_pred, batch_y)
        epoch_batch_loss.append(train_loss.item())
        train_loss.backward()
        optimizer_gd.step()

    train_loss_values.append(np.mean(epoch_batch_loss))

    # Evaluation
    model_gd.eval()
    with torch.inference_mode():
        y_pred_test = model_gd(X_test)
        test_loss = loss_function(y_pred_test, y_test)
        test_loss_values.append(test_loss)

# Plotting
plt.subplot(3, 1, 1)
train_loss_values_list_gd = [train_loss_value for train_loss_value in train_loss_values]
test_loss_values_list_gd = [test_loss_value.detach().cpu().item() for test_loss_value in test_loss_values]

plt.title("Gradient Descent\nTrain & Test Loss")
plt.plot(epoch_count_gd, train_loss_values_list_gd, color = "blue", label = "Train Loss")
plt.plot(epoch_count_gd, test_loss_values_list_gd, color = "red", label = "Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()


###################################################################



################## 2. Stochastic Gradient Descent ##################


model_sgd = LogisticRegression().to(device)
optimizer_sgd = torch.optim.SGD(params = model_sgd.parameters(), lr = 0.01)
loss_function = nn.CrossEntropyLoss()

epochs = 100
epoch_count_sgd = []
train_loss_values = []
test_loss_values = []

for epoch in range(epochs):

    epoch_count_sgd.append(epoch)

    model_sgd.train()

    epoch_batch_loss = []

    for batch_X, batch_y in sgd_loader:

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer_sgd.zero_grad()
        y_pred_train = model_sgd(batch_X)
        train_loss = loss_function(y_pred_train, batch_y)
        epoch_batch_loss.append(train_loss.item())
        train_loss.backward()
        optimizer_sgd.step()

    train_loss_values.append(np.mean(epoch_batch_loss))
    
    # Evaluation
    model_sgd.eval()
    with torch.inference_mode():
        y_pred_test = model_sgd(X_test)
        test_loss = loss_function(y_pred_test, y_test)
        test_loss_values.append(test_loss)

# Plotting
plt.subplot(3, 1, 2)
train_loss_values_list_sgd = [train_loss_value for train_loss_value in train_loss_values]
test_loss_values_list_sgd = [test_loss_value.detach().cpu().item() for test_loss_value in test_loss_values]

plt.title("Stochastic Gradient Descent\nTrain & Test Loss")
plt.plot(epoch_count_sgd, train_loss_values_list_sgd, color = "blue", label = "Train Loss")
plt.plot(epoch_count_sgd, test_loss_values_list_sgd, color = "red", label = "Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()



###################################################################



################## 3. Mini-Batch Gradient Descent ##################


model_mini_batch = LogisticRegression().to(device)
optimizer_mini_batch = torch.optim.SGD(params = model_mini_batch.parameters(), lr = 0.01)
loss_function = nn.CrossEntropyLoss()

epoch_count_mb = []
train_loss_values = []
test_loss_values = []

for epoch in range(epochs):

    epoch_count_mb.append(epoch)

    model_mini_batch.train()

    epoch_batch_loss = []

    for batch_X, batch_y in mini_batch_loader:

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer_mini_batch.zero_grad()
        y_pred_train = model_mini_batch(batch_X)
        train_loss = loss_function(y_pred_train, batch_y)
        epoch_batch_loss.append(train_loss.item())
        train_loss.backward()
        optimizer_mini_batch.step()

    train_loss_values.append(np.mean(epoch_batch_loss))

    # Evaluation
    model_mini_batch.eval()
    with torch.inference_mode():
        y_pred_test = model_mini_batch(X_test)
        test_loss = loss_function(y_pred_test, y_test)
        test_loss_values.append(test_loss)

# Plotting
train_loss_values_list_mb = [train_loss_value for train_loss_value in train_loss_values]
test_loss_values_list_mb = [test_loss_value.detach().cpu().item() for test_loss_value in test_loss_values]

plt.subplot(3, 1, 3)
plt.title("Mini-Batch Gradient Descent\nTrain & Test Loss")
plt.plot(epoch_count_mb, train_loss_values_list_mb, color = "blue", label = "Train Loss")
plt.plot(epoch_count_mb, test_loss_values_list_mb, color = "red", label = "Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()



###################################################################

################## Output ##################

Torch version: 2.11.0+cpu
Device: cpu
<matplotlib.legend.Legend at 0x7d45654ae8a0>
<img width="567" height="476" alt="image" src="https://github.com/user-attachments/assets/4078e035-dcdd-4bdf-8569-c39adf95f020" />

