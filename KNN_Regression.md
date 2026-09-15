# K-nearest neighbours regression (for continuous target values)
```
import numpy as np
import matplotlib.pyplot as plt
```
## Sample data for illustration
```
X_train = np.array([[1, 0.5], [1.4, 3], [4.2, 4.6], [3, 5], [2.7, 6]])
y_train = np.array([2, 3, 4, 5, 6])
X_test = np.array([[2.5, 3.5], [4.5, 5.5]])
```
### KNN regression
`k = 3`

# Make prediction
```
predictions = []
for x_test in X_test:
```
### Calculate distance between the x_test point and all data points in X_train
    distances = [np.linalg.norm(x_test - x_train) for x_train in X_train]
### Sort the data points by distances of the K nearest neighbours
    k_indices = np.argsort(distances)[:k]
### Get the target values of the K nearest neighbours
    k_nearest_neighbours = [y_train[i] for i in k_indices]
### Calculate the regression prediction as the mean of the target values of the K neighbours
### Give more weights on the nearer points, while the ferther points get less weights
```
    weights = distances[:k]
    prediction = np.average(k_nearest_neighbours, weights = weights)
    predictions.append(prediction)
```

# Plot the training data
`plt.scatter(X_train[:,0], y_train, label = "Training Data", color = "blue")`

# Plot the test data and their predicted labels
```
plt.scatter(X_test[:,0], predictions, label = "Test Predictions", color = "red", marker = "x")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.legend()
```

## Step by step procedure:

### Pick a single test point
`test_point = X_test[0]`

### Plot the test point
```
coord_text = f"({test_point[0].item(), test_point[1].item()})"
plt.scatter(test_point[0], test_point[1], color = "blue")
plt.text(test_point[0], test_point[1], coord_text, color = "blue", weight = "bold", horizontalalignment="left", verticalalignment="center")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.legend()
```
### Calculate distance between the point x and all data points in X_train
```
x_coord = X_train[:, 0]
y_coord = X_train[:, 1]

plt.title("Calculate the distances of each of the training points from the test point")
distances = [np.linalg.norm(test_point - x_train) for x_train in X_train]
for x, y in zip(x_coord, y_coord):
    coord_text = f"({x}, {y})"
    plt.scatter(x, y, color = "purple")
    plt.scatter(test_point[0], test_point[1], color = "blue")
    plt.plot([test_point[0], x], [test_point[1], y], color = "red")
    plt.text(x+x_offset, y+y_offset, coord_text, color = "blue", weight = "bold")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.xlim(-1, 8)
    plt.ylim(0, 8)
```
### Determine k = 3 nearest training points
```
plt.title("Determine k = 3 nearest training points")
sorted_distances_idx = np.argsort(distances)[:k]
sorted_training_points = np.array([X_train[i] for i in sorted_distances_idx])

x_coord_sorted = sorted_training_points[:, 0]
y_coord_sorted = sorted_training_points[:, 1]

for x, y in zip(x_coord_sorted, y_coord_sorted):
    coord_text = f"({x}, {y})"
    plt.scatter(x, y, color = "purple")
    plt.scatter(test_point[0], test_point[1], color = "blue")
    plt.plot([test_point[0], x], [test_point[1], y], color = "red")
    plt.text(x+x_offset, y+y_offset, coord_text, color = "blue", weight = "bold")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.xlim(-1, 8)
    plt.ylim(0, 8)
plt.legend()
```

### Get the target values of the K nearest neighbours
```
k_nearest_neighbours = [y_train[i] for i in k_indices]

for i, (x, y) in enumerate(zip(x_coord_sorted, y_coord_sorted)):
    coord_text = f"({x}, {y})"
    plt.scatter(x, y, color = "purple")
    plt.scatter(test_point[0], test_point[1], color = "blue")
    plt.plot([test_point[0], x], [test_point[1], y], color = "red")
    plt.text(x+x_offset, y+y_offset, coord_text, color = "blue", weight = "bold", horizontalalignment = "center", verticalalignment = "bottom")
    target_value = y_train[i]
    label_text = f"y = {target_value}"
    plt.text(x+x_offset, y+y_offset, label_text, color = "blue", weight = "bold", horizontalalignment = "center", verticalalignment = "top")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.xlim(-1, 8)
    plt.ylim(0, 8)
plt.legend()
```
### Take the average of the k nearest target values:
```
weights = distances[:k]
prediction = np.array(np.average(k_nearest_neighbours, weights = weights)).round(2)

plt.title("Regress the target by weighted-averaging the targets of the k-nearest training points")
coord_text = f"({test_point[0]}, {test_point[1]})"
label_text = f"y = {prediction}"
plt.scatter(test_point[0], test_point[1], color = "blue")
plt.text(test_point[0], test_point[1], coord_text, color = "black", horizontalalignment = "center", verticalalignment = "bottom")
plt.text(test_point[0], test_point[1], label_text, color = "black", horizontalalignment = "center", verticalalignment = "top")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.xlim(-1, 8)
plt.ylim(0, 8)
```


