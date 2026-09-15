# Extrapolation Limitations of Deep Neural Networks on Polynomial Domains
```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
```
### Prepare the training data
```
x_np = np.arange(0, 50, 1)/50.0
error = rng.normal(loc = 0, scale = 1, size = len(x_np))


y_true = [2 + 9*z**2 + 3.7*z**3 for z in x_np]
y_observed = y_true + error

X = torch.tensor(x_np, dtype=torch.float32).unsqueeze(dim=1)
y = torch.tensor(y_observed, dtype=torch.float32).unsqueeze(dim=1)
```


### Prepare the test data
```
X_test = torch.stack([i + torch.tensor(rng.normal(loc = 0, scale = 0.5, size = 1), dtype=torch.float32) for i in X])
```

### True function
```
def f(z: float) -> float:
    return 2 + 9*z**2 + 3.7*z**3
```
### Plot the data
```
plt.figure()
plt.scatter(x_np, y_observed, color = "blue")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Data")
```
<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/8acb1128-a001-46bd-a91a-a003fa76e20e" />



## Let us pretend that we know that the data follows a cubic polynomial, so we estimate weights w1, w2, w3, w4 to model a cubic function.

## The Fixed Analytical Model
```
class PolynomialRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(1, dtype = torch.float32), requires_grad = True)
        self.w2 = nn.Parameter(torch.randn(1, dtype = torch.float32), requires_grad = True)
        self.w3 = nn.Parameter(torch.randn(1, dtype = torch.float32), requires_grad = True)
        self.w4 = nn.Parameter(torch.randn(1, dtype = torch.float32), requires_grad = True)
    def forward(self, x):
        x = self.w1 + self.w2 * x + self.w3 * x**2 + self.w4 * x**3
        return x

model1 = PolynomialRegression()
loss_function = nn.MSELoss()
criterion1 = torch.optim.Adam(params = model1.parameters(), lr = 0.9)
```

### Training on Model 1
```
epochs = 5000
epoch_count = []
train_loss_model1 = []

for epoch in range(epochs):
    epoch_count.append(epoch)
    model1.train()
    y_pred_train = model1(X)
    loss = loss_function(y_pred_train, y)
    train_loss_model1.append(loss)
    criterion1.zero_grad()
    loss.backward()
    criterion1.step()
```

### Inference on Model 1
```
model1.eval()
with torch.inference_mode():
    y_pred_test_model1 = model1(X_test)
    test_loss_model1 = loss_function(y_pred_test_model1, torch.tensor([f(z.item()) for z in X_test]).unsqueeze(dim=1))

```
## Let us compute the training losses in the cases when we approximate the data points (which roughly follow a cubic) with linear/ quadratic/ cubic/ and quartic polynomials.
```
Train_Test_Losses_Model1 = pd.DataFrame(
    {
        'Models': ['Linear', 'Quadratic', 'Cubic', 'Quadratic', 'Quintic'],
        'Train Loss': [2.1739232540130615, 0.7642661333084106, 0.7586764693260193, 0.7547086477279663, 0.742479681968689],
        'Test Loss': [83.98222351074219, 6.129151821136475, 0.7571980953216553, 46.362247467041016, 1208.5894775390625],
        'Comments': ['High Bias, High Variance', 'Low Bias, High Variance', 'Perfect Fit', 'Not Biased but High Variance', 'Not Biased but High Variance']
        }
)
```
<img width="344" height="155" alt="image" src="https://github.com/user-attachments/assets/76a9669c-fde1-41d0-ab1a-1fe6475307b5" />


### Plotting the inference from Model 1
```
plt.figure()
plt.title("Test Errors on Model 1")
plt.scatter([x.detach().numpy() for x in X_test], [pred.numpy() for pred in y_pred_test_model1], color = "Red", label = "Predictions on Model 1")
plt.scatter([x.detach().numpy() for x in X_test], [f((z.detach().numpy())) for z in X_test], color = "Blue", label = "True Values")
plt.vlines([x.detach().numpy() for x in X_test], ymin = [pred.numpy() for pred in y_pred_test_model1], ymax =  [f((z.detach().numpy())) for z in X_test], colors = "Purple")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
```
<img width="577" height="455" alt="image" src="https://github.com/user-attachments/assets/3c9dc46f-1331-4186-a647-5007594e72c9" />

## Let us now use layers to define the neural network

### The Fixed Deep Learning Model
```
class LayeredPolynomialRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features = 1, out_features = 32)
        self.layer2 = nn.Linear(in_features = 32, out_features = 64)
        self.layer3 = nn.Linear(in_features = 64, out_features = 1)
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x


model2 = LayeredPolynomialRegression()
criterion2 = torch.optim.Adam(model2.parameters(), lr = 0.001)
```
### Training on Model 2
```
epochs = 5000
epoch_count = []
train_loss_model2 = []

for epoch in range(epochs):
    epoch_count.append(epoch)
    model2.train()
    y_pred = model2(X)
    loss = loss_function(y_pred, y)
    train_loss_model2.append(loss)
    criterion2.zero_grad()
    loss.backward()
    criterion2.step()

print(f"For model1, after {epoch_count[-1]+1} iterations, the train loss becomes: {train_loss_model2[-1]}")

```
### Inference on Model 2
```
model2.eval()
with torch.inference_mode():
    y_pred_test_model2 = model2(X_test)
    test_loss_model2 = loss_function(y_pred_test_model2, torch.tensor([f(z.item()) for z in X_test]).unsqueeze(dim=1))
```

### Plotting the inference from Model 2
```
plt.figure()
plt.title("Test Errors on Model 2")
plt.scatter([x.detach().numpy() for x in X_test], [pred.numpy() for pred in y_pred_test_model2], color = "Red", label = "Predictions on Model 2")
plt.scatter([x.detach().numpy() for x in X_test], [f((z.detach().numpy())) for z in X_test], color = "Blue", label = "True Values")
plt.vlines([x.detach().numpy() for x in X_test], ymin = [pred.numpy() for pred in y_pred_test_model2], ymax =  [f((z.detach().numpy())) for z in X_test], colors = "Purple")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
```
<img width="577" height="455" alt="image" src="https://github.com/user-attachments/assets/40abb7c9-2f63-4fb8-9b80-0fcf0d7a8b32" />

## Comparison of Model 1 & Model 2
```
plt.figure()
plt.plot(epoch_count, [loss.detach().numpy() for loss in train_loss_model1], c = "Red", label = "Train Loss for Model 1")
plt.plot(epoch_count, [loss.detach().numpy() for loss in train_loss_model2], c = "Blue", label = "Train Loss for Model 2")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Comparison of Test Losses for Model 1 & Model 2")
plt.legend()
```
<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/d4038956-038a-48bc-a43b-966a43c7ab49" />

## **Conclusion:** 
### Model 1 does great, but the problem occurs with the model 2, I see some high spikes in the train loss, although it reaches to zero in the long term, also the its performance on test points is horrible- the loss is very low for the points which are not very low and not very high, which lie in the middle of the test dataset, while for the extreme test data points, the test errors are very high.

### I tried using activation functions like ReLU (which gives best results for points which are within the range of the training points [0, 1]); Tanh (which gives similar results as ReLU); LeakyReLU (for which the test loss is huge for negative points (outside [0, 1]) and relatively relatively smaller (but still huge) errors for points greater than 1 (outside [0, 1]))

### So in both Model 1 & Model 2, the problem is they are weak at extrapolation. Model 1 somehow manages to extrapolate at points nearby the boundary points, whereas Model 2  fails terribly.
### One possible reason maybe Model 2 cannot capture the polynomial trend. So as a solution, I shall add some polynomial components into our model.

```
class LayeredPolynomialRegression(nn.Module):
    def __init__(self, degree = 3):
        super().__init__()

        self.baseline = nn.Linear(in_features = degree, out_features = 1)

        self.degree = degree
        self.layer1 = nn.Linear(in_features = degree, out_features = 32)
        self.layer2 = nn.Linear(in_features = 32, out_features = 1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # Generate polynomial terms [x^1, x^2, x^3]
        polynomial_features = [x**i for i in range(1, self.degree + 1)]
        x_poly = torch.cat(polynomial_features, dim = 1)
        # Calculate the mathematical baseline curve
        base_curve = self.baseline(x_poly)

        # Calculate the deep network correction
        nn_correction = self.activation(self.layer1(x_poly))
        nn_correction = self.layer2(nn_correction)

        return nn_correction + base_curve


modified_model2 = LayeredPolynomialRegression()
criterion3 = torch.optim.Adam(modified_model2.parameters(), lr = 0.001)
epochs = 5000
epoch_count = []
train_loss_modified_model2 = []
```
### Training on Modified Model 2
```
for epoch in range(epochs):
    epoch_count.append(epoch)
    modified_model2.train()
    y_pred = modified_model2(X)
    loss = loss_function(y_pred, y)
    train_loss_modified_model2.append(loss)
    criterion3.zero_grad()
    loss.backward()
    criterion3.step()

print(f"For Modified Model 2, after {epoch_count[-1]+1} iterations, the train loss becomes: {train_loss_modified_model2[-1]}")

```
### Inference on Modified Model 2
```
modified_model2.eval()
with torch.inference_mode():
    y_pred_test_modified_model2 = modified_model2(X_test)
    test_loss_model2 = loss_function(y_pred_test_modified_model2, torch.tensor([f(z.item()) for z in X_test]).unsqueeze(dim=1))
```

### Plotting the inference from Modified Model 2
```
plt.figure()
plt.title("Test Errors on Modified Model 2")
plt.scatter([x.detach().numpy() for x in X_test], [pred.numpy() for pred in y_pred_test_modified_model2], color = "Red", label = "Predictions on Modified Model 2")
plt.scatter([x.detach().numpy() for x in X_test], [f((z.detach().numpy())) for z in X_test], color = "Blue", label = "True Values")
plt.vlines([x.detach().numpy() for x in X_test], ymin = [pred.numpy() for pred in y_pred_test_modified_model2], ymax =  [f((z.detach().numpy())) for z in X_test], colors = "Purple")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
```
<img width="577" height="455" alt="image" src="https://github.com/user-attachments/assets/06db42c5-deb9-43b0-b652-cc95355d2e47" />

### Much better output, the extrapolated values are nearer to the true values.

## **Observations:**
### A bare neural network cannot perfectly approximate or extrapolate a polynomial curve outside our training data range.

### While a standard neural network (using only linear layers and activations like ReLU or LeakyReLU) can match the curve inside our training data perfectly, it is mathematically impossible for it to behave like a polynomial outside that range.

### Bare neural networks built with standard activations (ReLU, LeakyReLU, Tanh, Sigmoid) belong to a class of functions called piecewise linear or bounded functions.

### What Happens Past the boundary of the training dataset:
- ### 1. If using ReLU / LeakyReLU, the network locks into its final state and extrapolates into infinity as a perfectly straight line. A straight line can never track a cubic function, which curves exponentially faster as X grows.
- ### 2. If using Tanh / Sigmoid, the network completely saturates, flattening out into a horizontal line.

### No matter how many hidden layers, neurons, or training epochs you add, a bare network simply does not possess the mathematical operator for multiplication (X*X or X*X*X etc) to generate a true polynomial curve into unknown territory.

### Let us use our custom activation function (square function) and call the resulting model Model 4
```
class ActivationFunction(nn.Module):
    def forward(self, x):
        return x**2


class NewClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features = 1, out_features = 32)
        self.layer2 = nn.Linear(in_features = 32, out_features = 1)
        self.activation = ActivationFunction()
    def forward(self, x_in):
        x = self.activation(self.layer1(x_in))
        x_out = self.activation(self.layer2(x))
        return x_out
```
### The activation function will make the training dataset to learn quartic polynomial patterns, so it is obvious that this model will make huge errors for predicting beyond 1.
```
model4 = NewClassifier()
loss_function = nn.MSELoss()
criterion4 = torch.optim.Adam(model4.parameters(), lr = 0.01)
epochs = 5000
epoch_count = []
train_loss_model4 = []

for epoch in range(epochs):
    epoch_count.append(epoch)
    model4.train()
    y_pred = model4(X)
    loss = loss_function(y_pred, y)
    train_loss_model4.append(loss)
    criterion4.zero_grad()
    loss.backward()
    criterion4.step()

print(f"For Model 4, after {epoch_count[-1]+1} iterations, the train loss becomes: {train_loss_model4[-1]}") # Output: 0.7487860918045044

```
### Inference on Model 4
```
model4.eval()
with torch.inference_mode():
    y_pred_test_model4 = model4(X_test)
    test_loss_model2 = loss_function(y_pred_test_model4, torch.tensor([f(z.item()) for z in X_test]).unsqueeze(dim=1))
```

### Plotting the inference from Model 4
```
plt.figure()
plt.title("Test Errors on Model 4")
plt.scatter([x.detach().numpy() for x in X_test], [pred.numpy() for pred in y_pred_test_model4], color = "Red", label = "Predictions on Model 4")
plt.scatter([x.detach().numpy() for x in X_test], [f((z.detach().numpy())) for z in X_test], color = "Blue", label = "True Values")
plt.vlines([x.detach().numpy() for x in X_test], ymin = [pred.numpy() for pred in y_pred_test_model4], ymax =  [f((z.detach().numpy())) for z in X_test], colors = "Purple")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
```
<img width="577" height="455" alt="image" src="https://github.com/user-attachments/assets/96c70463-705d-4627-851b-e9fdc502b803" />

## **Observations:** 
### Test errors of Model 4 are huge beyond 1, but smaller compared to Modified Model 2; and beyond 0, just the opposite thing happens.


## **Conclusions:**

### The visual results perfectly confirm the structural limitations of each neural network architecture when tasked with curve extrapolation. The plots clearly illustrate how Model 1 tracks the cubic baseline flawlessly, Model 2 fails due to activation saturation, Modified Model 2 improves via its hybrid design but gets held back on the negative axis by LeakyReLU, and Model 4 matches the negative axis well but overshoots on the positive side due to its forced quartic (X^4) behavior.
```
conclusions = pd.DataFrame({
    'Network Architecture': [
        'Model 1 (Pure Polynomial)',
        'Model 2 (Bare Tanh MLP)',
        'Modified Model 2 (Hybrid LeakyReLU)',
        'Model 4 (Custom X^2 MLP)'
    ],
    'Inside Training Range ([0, 1])': [
        'Good', 'Good', 'Good', 'Good'
    ],
    'Extrapolation (X < 0)': [
        'Perfect', 'Fails (Flattens)', 'Poor (Stiff line)', 'Good (Forced Positive)'
    ],
        'Reason': [
        'Mathematical form matches the true cubic data-generating process exactly.',
        'Bounded activation saturates and locks hidden states between -1 and 1.',
        'Asymmetric LeakyReLU leaves piecewise linear trends that cannot curve.',
        'Two layers of squaring activations force a strict quartic (X^4) shape.'
    ]
}
)
```


<img width="610" height="197" alt="image" src="https://github.com/user-attachments/assets/0c6e7dbe-75d5-4961-b1b2-7ddefeca2429" />


    
  
