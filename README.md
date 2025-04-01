# Student_score_prediction
This project predicts student scores based on study hours using a linear regression model.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset: Study hours vs Scores
data = {
    'Study Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [15, 30, 35, 40, 50, 60, 70, 75, 85, 95]
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Plot the data
plt.scatter(df['Study Hours'], df['Score'], color='blue')
plt.title('Study Hours vs Score')
plt.xlabel('Study Hours')
plt.ylabel('Score')
plt.show()

# Splitting the dataset into training and testing sets
X = df[['Study Hours']]  # Features
y = df['Score']          # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Linear Regression: Study Hours vs Score')
plt.xlabel('Study Hours')
plt.ylabel('Score')
plt.show()
