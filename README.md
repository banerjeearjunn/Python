# **Analyzing Cardiotocography Dataset Using SHAP Values**
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
```
### fetch dataset
```
cardiotocography = fetch_ucirepo(id=193)
```
### data (as pandas dataframes)
```
X = cardiotocography.data.features
y = cardiotocography.data.targets['NSP']-1
```
### train-test split
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 0)
```
### initialize the xgboost model
```
model = xgb.XGBClassifier(n_estimators = 100, max_depth = 5, learning_rate = 0.1, random_state = 0)
```
### train model
```
model.fit(X_train, y_train)
```
### prediction
```
y_pred_test = model.predict(X_test)
```
### accuracy
```
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Classification Accuracy: {accuracy*100:.2f}")
```
## SHAP: SHapley Additive exPlanations
```
import shap
```
### initialize the SHAP explainer
```
explainer = shap.TreeExplainer(model)
```
### compute SHAP values from the test set
```
shap_values = explainer(X_test)

shap_values.shape # (426, 21, 3) = (number of samples in X_test, number of features, number of classes)
```
### The shap_values object contains 3 arrays (shap_values.values, shap_values.base_values, shap_values.data)
```
print(f"Shape of shap_values.values{shap_values.values.shape} (samples, features, classes)\nShape of shap_values.bese_values: {shap_values.base_values.shape} (samples, classes)\nShape of shap_values.data: {shap_values.data.shape} (samples, features)")
```
### shap_values.data: A direct copy of the input test set X_test
### shap_values.base_values: Before looking at any features, the XGBoost model establishes the initial starting line using log-odds. This becomes the fixed base values for every single test patient It represents the "prior belief" before reading any medical metrics.
### shap_values.values: These are the shap values for each patient.

## Putting Everything Together
### We first build the ensemble of n_estimators = 100 decision trees
### Next, SHAP defines the baseline value (as the expected value of each class disregarding all features) of the model's output over the training distribution.
### Next, for each patient in our test set X_test, we calculate its SHAP values.
### Finally, we calculate the base value + SHAP value = final model sore, and convert it into a probability using logistic function (1/(1+e^-x))
### We compare the individual feature impacts against the baseline value. If a feature's SHAP value is positive, it actively increases the log-odds score of the patient belonging to that class. If it is negative, it actively decreases the score, ruling that class out. A SHAP value near 0 means the feature had no influence on that specific patient's prediction.



# Plottings
## Figure 1: Multi-Class Bar Plot (Global Feature Importance)
```
plt.figure(figsize = (10, 6))
shap.summary_plot(shap_values.values, X_test, plot_type = "bar", show = False)
plt.title("Global Feature Importance Across All 3 Classes", fontsize = 14)
plt.tight_layout()
plt.show()
```
<img width="789" height="940" alt="image" src="https://github.com/user-attachments/assets/116808c4-4866-4182-a4e3-9581d5c7d813" />

```
classes = ["Normal", "Suspect","Pathologic"]
```
## *Observations:*
### This plot looks across the entire test set (X_test) and ranks features based on their overall absolute impact on the model's predictions.
### The x-axis represents the average absolute SHAP value (mean(|SHAP value|)). higher values mean a feature has a greater influence on the model. It measures magnitude, not direction. SHAP takes the absolute value of every individual patient's score before averaging them. It does not matter if a feature pushed a patient's risk up by +0.5 or down by -0.5. Both changes represent a substantial alteration to the model's decision, so both are treated as an absolute impact of 0.5.


### The total length of a bar: represents the average size of the push (or impact) that feature had on the model's final prediction scores across all 426 patients in the entire test dataset; it answers the question: "how much did this medical measurement shift the final decision away from the default baseline guess?"
### A long bar means he feature is powerful signal that the model relies on constantly to make its decisions.
### A short bar (like DL or Nzeros) means t• he feature has almost no influence, even if we drastically change a patient's Nzeros value, it barely nudges the model's final prediction scores.

### The bars are divided into three distinct segments or colors, one for each class: Class 0, Class 1, Class 2.
### A bar that is mostly one color tells us that feature is highly specialised.

### Feature importance ranking isolated for each individual class:
### Class 2 (Pathologic): Median, ASTV, Mean, ALTV, LB, DP, UC, Mode
### Class 0 (Normal): AC, ASTV, UC, DP, ALTV, Mode, Mean, Median, Variance
### Class 1 (Suspect): ALTV, AC, MSTV, ASTV, Mean, Min, Max, Variance

## Figure 2: Beeswarm Plot (For a Specific Class)
```
for c in range(3):
    plt.figure(figsize = (10, 6))
    shap.plots.beeswarm(shap_values[:, :, c], show = False)
    plt.title(f"SHAP Beeswarm Plot for Class {c} ({classes[c]})", fontsize = 14)
    plt.tight_layout()
    plt.show()
```
<img width="776" height="540" alt="image" src="https://github.com/user-attachments/assets/e44e8c8d-a93e-43c5-bfcc-4f8054f433ce" />
<img width="776" height="540" alt="image" src="https://github.com/user-attachments/assets/a390578c-2387-46ea-8cf3-0784c770806d" />
<img width="776" height="540" alt="image" src="https://github.com/user-attachments/assets/60ede712-2459-46a2-bf38-6daf6fd16f1c" />


## *Observations:*
### There are 3 components of these plots: vertical ranking, dot color and horizontal position.

### Vertical ranking is based on the importance/predictive power of the features for the specific class. We note that this rankings agree with our rankings from figure 1, which shows the importance of the features for predicting each class.

### Let us discuss only the case of Class 0.

### For Class 0 (Normal), AC is at the very top, this means it is the absolute most critical feature for determining whether a patient is normal; ASTV and UC follow closely behind as heavy drivers.
### There is a large cluster of red/pink dots on the right side (positive SHAP) and blue dots on the left side (negative SHAP).
### High AC (red) strongly push the model to classify the patient as Normal (Class 0), conversely, low or zero accelerations (blue) significantly decrease the probability of being normal.

### ASTV is the exact opposite of AC, there is a long spread of red/pink dots on the left side (negative SHAP) and a dense cluster of blue dots on the right side (positive SHAP).
### High ASTV (red) heavily penalizes the "Normal" classification - driving the risk down into danger zones, low ASTV keeps the patient firmly on the "Normal" side.

### For the feature UC, the red dots extend to the right, while the blue dots lean to the left. higher numbers of UC (red) generally correlate with a "Normal" patient.

### For the features DP & ALTV, we notice a long tail of red dots stretches far out to the left (reaching down to -2 or lower).
### This means that even a tiny increase in DP or ALTV (red) acts as a red flag, it instantly and forcefully subtracts from the "Normal" score, telling the model that this patient is very likely in a Suspect (Class 1) or Pathologic (Class 2) state.

### Summary: we observe where the red dots cluster vs. the blue dots.
### If red is on the right: high values of this metric = good indicator for a "Normal" (Class 0) diagnosis.
### If red is on the left: high values of this metric = dangerous indicator that rules out a "Normal" (Class 0) diagnosis.



## Figure 3: Waterfall Plot (Individual Prediction Breakdown)
### Pick a single patient to see exactly why the model made its specific prediction for, say class 2
```
instance = 1
for c in range(3):
    plt.figure(figsize = (10, 6))
    shap.plots.waterfall(shap_values[instance, :, c], show = False)
    plt.title(f"SHAP Waterfall Plot for Test Instance {instance}\n(Class {c} Prediction)")
    plt.tight_layout()
    plt.show()
```
<img width="793" height="640" alt="image" src="https://github.com/user-attachments/assets/88a870df-b265-4978-ac3d-26d1b59f402e" />
<img width="796" height="640" alt="image" src="https://github.com/user-attachments/assets/9abc500a-e173-4d84-be35-e2c9115c4567" />
<img width="791" height="640" alt="image" src="https://github.com/user-attachments/assets/91bc361c-cb93-4512-b4cd-3a9f2ef25d32" />

## *Observations:*
### The average log-odds score for Class 0, Class 1 & Class 2 are 1.35, -0.351 & -0.992

### If we look at the log-odds output f(x) at the top of each plot:
### Class 0 (Normal): f(x) = 4.886, extremely high, strong positive push
### Class 1 (Suspect): f(x) = -3.157, very low, strong negative push
### Class 2 (Pathologic): f(x) = -4.963, very low, strong negative push
### Since Class 0 has the highest positive score, the model firmly classifies this patient as Normal.

### Plot 1: Breakdown for Class 0 (Normal)
### Starting Point E[f(X)] = 1.35: The base log-odds probability of a random fetus being healthy.
### The single biggest driver is AC = 0.008, which single-handedly adds +2.09 to the score.
### Other healthy/"Normal" signs like ALTV = 0 (+0.27) and DP = 0 (+0.23) also confirm the absence of dangerous variations, pushing the final score up to a confident 4.886.

### Plot 2: Breakdown for Class 1 (Suspect)

### Starting Point E[f(X)] = -0.351: The default baseline log-odds.
### Because the fetus has very high AC (0.008), the model heavily penalizes the "Suspect" score by -1.71.
### Having ALTV = 0 drops the score by another -0.75, dragging the final result deep into the negatives (-3.157).


### Plot 3: Breakdown for Class 2 (Pathologic)

### Starting Point E[f(X)] = -0.992: The default baseline log-odds.
### This plot shows what safely rules out a dangerous pathologic condition, the primary lifesavers (features) here are good Median = 150 (which drops pathologic risk by -1.29) and a healthy Mean = 146 (which drops risk by -0.57).
### Even though a slightly elevated LB (134) gives a small nudge of +0.26, the overwhelming collection of healthy markers crushes any chance of a pathologic diagnosis, ending at a massive negative score of -4.963.




## Figure 4: SHAP Dependence Plot (Feature Interaction Analysis)
### Extract SHAP values specifically for class 2 (Pathologic)
```
shap_values_class2 = shap_values[:, :, 2]
```
### Automatic interaction selection: SHAP will automatically pick the feature that has the strongest interaction with ASTV.
```
plt.figure(figsize = (10, 6))
shap.plots.scatter(shap_values_class2[:, "ASTV"])
plt.title("SHAP Dependence Plot for ASTV (Class 2: Pathologic)", fontsize = 14)
plt.tight_layout()
plt.show()
```
<img width="567" height="453" alt="image" src="https://github.com/user-attachments/assets/b776d9c5-1e9c-4cc5-b44b-89be254e7474" />

## *Observations:*
### In this example, the X-axis represents the ASTV feature.
###  The blue dots stay in a flat, tight band between 15 and 70, a vertical surge begins right after 70.
### For most patients (ASTV between 15 and 70), the blue dots sit below the 0 line, hovering around -1.0 and -0.5. The moment ASTV crosses 70, the dots rocket straight up, with the highest point landing just under +5.0 on the Y-axis.
### The gray shapes at the very bottom are a histogram of the test dataset X_test for the feature ASTV.
### The gray bars are tall between 20 and 65, it means there are a lot of patients with those values. This is why the blue dots are clustered so densely there.
### The gray bars are tiny or flat above 70, it means patients with such high abnormal variability are rare in our dataset.

### The region where ASTV<25 the blue dots dip downwards, hitting -1.5 to -2.0. This means for a very low abnormal short-term variability of ASTV, the pathologic state (Class 2) is highly unlikely.
### Once ASTV crosses 70, the dots are scattered wildly between +1.0 and +5.0, this vertical spread means that an ASTV above 70 is highly dangerous, but its exact danger level depends heavily on other features, which is exactly what the color-coded interaction plot will reveal.


## Manual Interaction Selection (ALTV & ASTV for Class 2)
```
plt.figure(figsize=(10, 6))
shap.plots.scatter(shap_values_class2[:, "ASTV"], color=shap_values_class2[:, "ALTV"], show=False)
plt.title("SHAP Dependence Plot: ASTV colored by ALTV for Class 2", fontsize=14)
plt.tight_layout()
plt.show()
```
<img width="701" height="490" alt="image" src="https://github.com/user-attachments/assets/18a360d1-902e-4279-b410-38b8408d7b5d" />

## *Observations:*
### The dots are a mix of light blue (ALTV = 10) and bright pink (ALTV = 50) scattered randomly together. This means ALTV isn’t doing a great job explaining why some patients at ASTV = 80 have a SHAP value of 2.5 while others jump to 4.5 to 5.


## Automatic Interaction Selection (with ASTV for Class 2)
```
plt.figure(figsize=(10, 6))
shap.plots.scatter(shap_values_class2[:, "ASTV"], color=shap_values[:, :, 2], show=False)
plt.title("SHAP Dependence Plot for ASTV for Class 2", fontsize=14)
plt.tight_layout()
plt.show()
```
<img width="710" height="490" alt="image" src="https://github.com/user-attachments/assets/50336adc-c772-4b9f-8868-bb8b781775ea" />

## *Observations:*
### In this interaction plot, SHAP searches all 21 variables and automatically picks Median, because Median explains the wild vertical spreading of the risk score much more cleanly.
### We can divide the plot into 3 regions: Low ASTV Zone (15 to 25), The Middle Zone (30 to 70), The High-Risk Zone (> 78)

### **Low ASTV Zone (15 to 25):** The pink dots in this region stay higher up near -1.0 and bright blue dots dip down to -2.0. This means that if a patient has low ASTV (15-25) and has low Median (115-120), the pathologic score gets lower. And if a patient has an ASTV value of 15-25, and a Median value between around 130-140, the pathologic risk goes a little bit higher (-2 to -1).
### **The Middle Zone (30 to 70):** The pink dots compress into a perfectly tight, horizontal line around -0.5. This means, a patient with ASTV value between 30 to 40 and Median value around 150-160 is not generally classified as Pathologic (Class 2). 
### **The High-Risk Zone (> 78):** The colors now group beautifully. The bright blue dots (low Median value) are clustered around +2.5 to +4.0. The dark purple/pink dots (High Median value) rise even higher, reaching +4.0 to +5.0. We can interpret this as: an ASTV over 78 is highly pathologic on its own, but if the patient also has a high Median value, the two risk factors combine inside the XGBoost trees to trigger the maximum possible risk penalty.

