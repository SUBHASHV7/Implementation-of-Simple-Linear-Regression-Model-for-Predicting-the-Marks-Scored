# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset into a DataFrame and explore its contents to understand the data structure.
2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.

3.Create a linear regression model and fit it using the training data.

4.Predict the results for the testing set and plot the training and testing sets with fitted lines.

5.Calculate error metrics (MSE, MAE, RMSE) to evaluate the model’s performance.

## Program and Output:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Subhash V
RegisterNumber: 212224240163
```

~~~py
# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('exp_2_dataset_student_scores.csv')   # CSV should have two columns, e.g. "Hours","Scores"
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")

X = df.iloc[:, :-1].values   # all rows, all columns except last -> shape (n_samples, 1)
Y = df.iloc[:, -1].values    # all rows, last column -> shape (n_samples,)
print("X (features):", X.flatten())
print("Y (targets):", Y) 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))

regressor = LinearRegression()
regressor.fit(X_train, Y_train)   # fit on training data

Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)

plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

new_hours = np.array([[2.5], [8.0]])   # shape must be (n_samples, 1)
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
~~~

## Output: 

<img width="303" height="295" alt="image" src="https://github.com/user-attachments/assets/a2347076-12f6-4fcd-aeaf-af1fcfde6df5" />


<img width="792" height="84" alt="image" src="https://github.com/user-attachments/assets/de73e5bb-29c6-4a12-acd6-017373c9cfbd" />


<img width="436" height="45" alt="image" src="https://github.com/user-attachments/assets/9785ca7a-d9ec-4753-a32a-35c5d5e312d4" />


<img width="320" height="86" alt="image" src="https://github.com/user-attachments/assets/77b81f10-f952-4e41-a232-7c39e5173e1a" />


<img width="718" height="69" alt="image" src="https://github.com/user-attachments/assets/c48b88e1-4ec2-46b6-a56a-7c24c35661ef" />


<img width="651" height="413" alt="image" src="https://github.com/user-attachments/assets/06baa508-2a80-454f-8837-b5fb4de1058f" />


<img width="535" height="399" alt="image" src="https://github.com/user-attachments/assets/3cf7082e-a4b3-4df2-a148-1ec8ce946848" />


<img width="458" height="91" alt="image" src="https://github.com/user-attachments/assets/1e859940-b1d3-4ae7-b0f5-f8e7c7d04116" />


<img width="515" height="51" alt="image" src="https://github.com/user-attachments/assets/69044afe-c94b-449c-8b42-e993aaa0510b" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
