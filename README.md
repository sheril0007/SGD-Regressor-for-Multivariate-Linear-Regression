# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset containing house features such as area, number of bedrooms, and age of the house along with output values (price and occupants).
2. Split the dataset into training and testing sets and standardize the input features using StandardScaler.
3. Train the SGD Regressor model using MultiOutputRegressor for predicting multiple output variables.
4. Predict the house price and number of occupants for test data and evaluate the model using Mean Squared Error and R² score.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SHERIL P
RegisterNumber:  212225230262
*/
```
```
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# -----------------------------
# Step 1: Create Dataset
# -----------------------------

# Input Features: [House Size (sq ft), Number of Rooms]
X = np.array([
    [800, 2],
    [1000, 3],
    [1200, 3],
    [1500, 4],
    [1800, 4],
    [2000, 5],
    [2200, 5],
    [2500, 6]
])

# Output Targets: [House Price (in lakhs), Number of Occupants]
y = np.array([
    [40, 2],
    [55, 3],
    [65, 3],
    [85, 4],
    [95, 4],
    [110, 5],
    [125, 5],
    [145, 6]
])

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# Step 3: Feature Scaling
# -----------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Step 4: Create SGD Regressor Model
# -----------------------------

sgd = SGDRegressor(
    max_iter=2000,
    eta0=0.01,
    learning_rate='constant',
    random_state=42
)

model = MultiOutputRegressor(sgd)

# -----------------------------
# Step 5: Train the Model
# -----------------------------

model.fit(X_train_scaled, y_train)

# -----------------------------
# Step 6: Test the Model
# -----------------------------

y_pred = model.predict(X_test_scaled)

print("Predicted [Price, Occupants]:")
print(y_pred)

print("\nActual [Price, Occupants]:")
print(y_test)

# -----------------------------
# Step 7: Evaluate the Model
# -----------------------------

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# -----------------------------
# Step 8: Predict for New House
# -----------------------------

# New house: 1600 sq ft, 4 rooms
new_house = np.array([[1600, 4]])
new_house_scaled = scaler.transform(new_house)

new_prediction = model.predict(new_house_scaled)

print("\nFor New House [1600 sq ft, 4 rooms]:")
print("Predicted House Price (lakhs):", round(new_prediction[0][0], 2))
print("Predicted Number of Occupants:", round(new_prediction[0][1]))
import matplotlib.pyplot as plt

# Plot: House Size vs Price (Actual vs Predicted)
plt.figure()
plt.scatter(X_test[:, 0], y_test[:, 0], label="Actual Price")
plt.scatter(X_test[:, 0], y_pred[:, 0], label="Predicted Price")
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price (lakhs)")
plt.title("House Size vs House Price (Actual vs Predicted)")
plt.legend()
plt.show()
```
## Output:
<img width="831" height="690" alt="Screenshot 2026-01-31 131007" src="https://github.com/user-attachments/assets/1be90aae-e800-48b4-85c8-2f5b291b514f" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
