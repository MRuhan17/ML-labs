"""
Project 1: Predict Student Scores using Linear Regression

📌 Objective:
Predict exam scores of students based on the number of hours studied using a simple linear regression model.

Author: Ruhan
Date: August 2025
"""

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Create Dataset
data = {'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Scores': [35, 40, 50, 65, 75, 85, 88, 95, 99]}
df = pd.DataFrame(data)

# Step 3: Features (X) and Labels (y)
X = df[['Hours']]   # independent variable (2D)
y = df['Scores']    # dependent variable

# Step 4: Train Model
model = LinearRegression()
model.fit(X, y)

# Step 5: Show model parameters
print("Equation of line: y = {:.2f}x + {:.2f}".format(model.coef_[0], model.intercept_))

# Step 6: Make Predictions
pred_7_5 = model.predict([[7.5]])
pred_10 = model.predict([[10]])

print("Predicted score for 7.5 hours of study: {:.2f}".format(pred_7_5[0]))
print("Predicted score for 10 hours of study: {:.2f}".format(pred_10[0]))

# Step 7: Visualization
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, model.predict(X), color="red", label="Best fit line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.title("Student Scores Prediction (Linear Regression)")
plt.show()
