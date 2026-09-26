### Import necessary modules
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
```
### The Rain in Australia dataset can be found [here](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

### Read the data
```
df = pd.read_csv("/content/weatherAUS.csv")
```
```
df.describe()
```
### Check missing values
### If the number of missing values for a feature is very high, remove the entire column.

```
missing_count = df.isnull().sum()
value_count = len(df)
missing_percentage = (missing_count/value_count*100).round(2)

missing_df = pd.DataFrame({
    'Count': missing_count,
    'Percentage': missing_percentage
})
```
<img width="188" height="533" alt="Screenshot 2026-09-26 193429" src="https://github.com/user-attachments/assets/579c9557-8d9a-43aa-9f7e-b661e401088c" />


### Evaporation: 43.17%, Sunshine: 48.01, Cloud9am: 38.42, Cloud3pm: 40.81
### Remove these features
```
df = df.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis = 1)
```
### Let us see whether there are any missing target values
```
print(f"Number of missing target values: {df['RainTomorrow'].isnull().sum()}")
output:  3267
```

### Treat the missing values of the label RainTomorrow by removing the entire rows.
```
df = df.dropna(subset = ['RainTomorrow'], axis = 0)
```
### Before removing the missing target values: 145460
### After removing the missing target values: 142193

### For remaining columns, impute the missing categorical values with na and the numerical values with their means.
```
df_num_cols = []
df_char_cols = []

for col in df.columns:
    if col != 'RainTomorrow':
        if is_numeric_dtype(df[col]):
            df_num_cols.append(col)
        elif is_string_dtype(df[col]):
            df_char_cols.append(col)


df[df_num_cols] = df[df_num_cols].fillna(df[df_num_cols].mean(numeric_only = True, axis = 0))

for i in df_char_cols:
    if df[i].isnull().any():
        df[i].fillna("Unknown", inplace = True)
```

### Address outliers: if a point is > 0.9 quantile, remove it.
```
maximum = df['Rainfall'].quantile(0.9)
df = df[df['Rainfall'] < maximum]
plt.figure()
sns.displot(data = df, x = 'Rainfall')
```
<img width="489" height="489" alt="image" src="https://github.com/user-attachments/assets/e64cbc89-ca3a-4a75-9d61-a383d63c957a" />

### Feature Transformation: transform the 'month' variable as categorical variable
```
df['Month'] = pd.to_datetime(df['Date']).dt.month.astype('category')
```
### Drop the Date variable
```
df = df.drop('Date', axis = 1)
```

### Encoding Categorical Data
```
encoder = LabelEncoder()
```
### List the categorical features
```
categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainTomorrow', 'RainToday', 'Month']

```
### Encode the labels
```
for i in categorical_features:
    df[i] = encoder.fit_transform(df[i])
```
### Get all the information about the dataset
```
df.info()
```
### Correlation Analysis
```
correlation = df.corr()
plt.figure(figsize = (15, 15))
sns.heatmap(correlation, cmap = "GnBu", annot = True)
```
<img width="1229" height="1300" alt="image" src="https://github.com/user-attachments/assets/638ff684-2a8d-4036-8bc4-979a46554685" />

### Logistic regression requires there to be little multicollinearity among predictors.
### Keep only one variable in each group of highly correlated variables.
```
df = df[['Location', 'MinTemp', 'Rainfall', 'WindGustDir', 'WindGustSpeed', 'WindDir9am',  'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Pressure9am', 'Month', 'RainTomorrow']]
```
### Model Building
```
X = df.iloc[:, :-1]
y = df['RainTomorrow']
```
### Train Test Split
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, stratify = y, random_state = 1)
```

# Logistic Regression
```
clf = LogisticRegression(max_iter = 1000)
clf.fit(X_train, y_train)
y_pred_lr = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)
accuracy_clf = clf.score(X_test, y_test)
print(f"Using Logistic Regression, Accuracy Score: {(accuracy_clf*100):.2f}% ")
```
### Output: Using Logistic Regression, Accuracy Score: 83.12% 

### Finding test loss
```
test_loss = 0.0
for s in range(len(y_test)):
    test_loss -= np.log(y_pred_prob[s][1])*(np.array(y_test)[s]) + np.log(y_pred_prob[s][0])*(1 - np.array(y_test)[s])
test_loss = 1/(len(y_test))*test_loss
```

# Decision Tree
```
dtc = DecisionTreeClassifier(criterion = "gini", splitter = "best", max_leaf_nodes = None, max_depth = None, random_state = 1, max_features = None, min_samples_split = 2)

dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)
accuracy_dtc = accuracy_score(y_test, y_pred_dtc)
print(f"Using Decision Tree Regression, Accuracy Score: {(accuracy_dtc*100):.2f}% ")
```
### Output: Using Decision Tree Regression, Accuracy Score: 75.59% 

### Plotting Decision Tree: max_depth = 5
```
plt.figure()
plot_tree(dtc, feature_names = X.columns, class_names = ["Rain Tomorrow", "Not Rain Tomorrow"], label = 'all', max_depth = 5)
```
<img width="518" height="389" alt="image" src="https://github.com/user-attachments/assets/d0e46ef5-79b6-4b79-9ef8-4ad47fd1bcec" />

### Let us prune the decision tree and see whether the accuracy improves.
### Cost Complexity Pruning
```
path = dtc.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
```
### The maximum effective alpha value is removed, because it is the trivial tree with only one node.
```
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker = "o", drawstyle = "steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total impurity vs eccective alpha for training set")
```

<img width="594" height="455" alt="image" src="https://github.com/user-attachments/assets/4bf5c094-6752-4b5a-8981-4f9806ab3e5f" />

### Next, we train a decision tree using the effective alphas. The last value in ccp_alphas is the alpha value that prunes the whole tree, leaving the tree, clfs[-1], with one node.
```
dtcs = []
for ccp_alpha in ccp_alphas:
    dtc = DecisionTreeClassifier(ccp_alpha = ccp_alpha, random_state = 0)
    dtc.fit(X_train, y_train)
    dtcs.append(dtc)
print(f"Number of nodes in the last tree is : {dtcs[-1].tree_.node_count} with ccp_alpha = {ccp_alphas[-1]}")
```
### Here we show that the number of nodes and tree depth decreases as alpha increases.
```
dtcs = dtcs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [dtc.tree_.node_count for dtc in dtcs]
depth = [dtc.tree_.max_depth for dtc in dtcs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_aplhas, node_counts, marker = "o", drawstyle = "steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of Nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker = "o", drawstyle = "steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].title("Depth vs alpha")
fig.tight_layout()
```
### Accuracy vs alpha for Training & Testing sets
```
train_scores = [dtc.score(X_train, y_train) for dtc in dtcs]
test_scores = [dtc.score(X_test, y_test) for dtc in dtcs]


fig, ax = plt.subplots()
ax.plot(ccp_alphas, train_scores, marker = "o", label = "train", drawstyle = "steps-post")
ax.plot(ccp_alphas, test_scores, marker = "o", label = "test", drawstyle = "steps-post")
ax.set_xlabel("alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs alpha for Training & Testing sets")
```

## XGBoost
```
xgboost = xgb.XGBClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 4, enable_categorical=True)
xgboost.fit(X_train, y_train)
y_pred_xgboost = xgboost.predict(X_test)
accuracy_xgboost = xgboost.score(X_test, y_test)
print(f"Using XGBoost, Accuracy Score: {(accuracy_xgboost*100):.2f}% ")
```
### Output: Using XGBoost, Accuracy Score: 84.08% 


## Random Forest
```
rf = RandomForestClassifier(n_estimators = 100, criterion = "gini", min_samples_split = 2, max_features = 3, n_jobs = -1, random_state = 0)
rf.fit(X_train, y_train) ### MAX_FEATURES = np.sqrt(len(X.columns)).round(0) = 3
y_pred_rf = rf.predict_proba(X_test)
accuracy_rf = rf.score(X_test, y_test)
print(f"Using Random Forest, Accuracy Score: {(accuracy_rf*100):.2f}% ")
```
### Output: Using Random Forest, Accuracy Score: 84.05% 

```
accuracy = pd.DataFrame({
    'Logistic Regression': [accuracy_clf],
    'Decision Tree': [accuracy_dtc],
    'XGBoost': [accuracy_xgboost],
    'Random Forest': [accuracy_rf]
})

plt.figure()
plt.bar(accuracy.columns, accuracy.iloc[0])
```
<img width="547" height="413" alt="image" src="https://github.com/user-attachments/assets/bb514399-4284-4a1c-9998-0bb3ed635cb9" />



## Feature importance based on feature permutation

### Calculate the permutation importance
```
importances = permutation_importance(rf, X_test, y_test, n_repeats = 10, random_state = 0, n_jobs = -1)
```
### Extract the average drop in accuracy for each feature across the 10 shuffles, it wraps these numbers into a pandas Series and maps them to their original column names (index=X.columns)
```
forest_importances = pd.Series(importances.importances_mean, index=feature_names)
```
<img width="137" height="284" alt="image" src="https://github.com/user-attachments/assets/4ae7c900-f16a-4a11-ae09-3db006b3b1f8" />

### Sort the features

```
forest_importances = forest_importances.sort_values(ascending=True)
```
### Extract the standard deviation of the accuracy drops across the 10 repeats, showing how much the importance score varied
```
std_series = pd.Series(importances.importances_std, index=X.columns)
importances_std = std_series.reindex(forest_importances.index)
```
<img width="137" height="290" alt="image" src="https://github.com/user-attachments/assets/8ca3d50e-0f4c-44ec-a092-439dd950ebed" />

### Plotting the results
```
fig, ax = plt.subplots(figsize=(10, 6))
forest_importances.plot.barh(xerr=importances_std, ax=ax, color='skyblue', edgecolor='black')

ax.set_title("Permutation Feature Importance (Test Set)")
ax.set_xlabel("Decrease in Mean Accuracy")
ax.set_ylabel("Features")
fig.tight_layout()
```
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/246d686a-11dc-48db-9e72-95a262cda60d" />

### Conclusion: WindGustSpeed, Humidity9am and Pressure9am are most important variables for prediction.
