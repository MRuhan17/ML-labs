import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 1. Create Dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Pass":  [0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# 2. Features & Labels
X = df[["Hours"]]   # input: study hours
y = df["Pass"]      # output: pass/fail

# 3. Train Logistic Regression Model
model = LogisticRegression()
model.fit(X, y)

# 4. Predictions
hours = [[2], [4], [6], [8]]
predictions = model.predict(hours)
probabilities = model.predict_proba(hours)

print("Predictions (Pass/Fail):", predictions)
print("Probabilities:\n", probabilities)

# 5. Visualization
plt.scatter(df["Hours"], df["Pass"], color="blue", label="Data Points")
plt.plot(df["Hours"], model.predict_proba(df[["Hours"]])[:,1], color="red", label="Logistic Curve")
plt.xlabel("Hours Studied")
plt.ylabel("Pass Probability")
plt.legend()
plt.show()
