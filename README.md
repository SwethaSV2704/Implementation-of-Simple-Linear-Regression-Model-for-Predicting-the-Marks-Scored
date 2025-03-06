# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import required libraries and read the data frame

2. Assign hours to X and scores to Y 

3. Implement the training set and test set the dataframe

4. plot the required graph for both test data and training data

5. Find the values of MSE, MAE and RMSE

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SWETHA S V
RegisterNumber: 212224230285
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:\\Users\\admin\\Downloads\\student_scores.csv")
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.xlabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_test,regressor.predict(x_test),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.xlabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![Screenshot 2025-03-06 093917](https://github.com/user-attachments/assets/aa18caf4-2480-4997-bdd9-422cd6a4c93d)

![Screenshot 2025-03-06 094020](https://github.com/user-attachments/assets/ce072dc2-b023-4ebe-8ec2-68eb5ae38a92)

![Screenshot 2025-03-06 094103](https://github.com/user-attachments/assets/a2897e96-efbb-4f1e-a9ed-e80d1848cb22)

![Screenshot 2025-03-06 094144](https://github.com/user-attachments/assets/7b72a1b7-2bac-4a5c-9065-6bf3e73b028d)

![Screenshot 2025-03-06 094230](https://github.com/user-attachments/assets/395a3527-fc52-4ef3-88fd-c1efd729be30)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
