
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Sample Data (study hours and scores)
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Scores': [20, 40, 60, 80, 85, 95, 90, 97, 98, 100]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[['Hours']]  # Independent variable (study hours)
y = df['Scores']   # Dependent variable (scores)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Coefficients: ", model.coef_)
print("Model Intercept: ", model.intercept_)
print("Mean Absolute Error (MAE): ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE): ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE): ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Plotting the data
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Study Hours vs Score Prediction')
plt.xlabel('Study Hours')
plt.ylabel('Scores')
plt.show()
