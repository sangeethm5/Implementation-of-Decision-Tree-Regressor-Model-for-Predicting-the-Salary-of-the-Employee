# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

data = pd.read_csv("Salary.csv")
print(data.head(), "\n")
print(data.info(), "\n")
print(data.isnull().sum(), "\n")

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head(), "\n")

x = data[["Position", "Level"]]
y = data["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

dt = DecisionTreeRegressor(random_state=2)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(mse)
print(r2)

print(dt.predict([[5, 6]])[0])
```
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:Sangeeth M
RegisterNumber:25004402
```

## Output:
### data.head():

![Screenshot 2025-05-17 084149](https://github.com/user-attachments/assets/a00584bf-559e-4c69-a495-72a01e455167)

### data.isnull().sum():

![Screenshot 2025-05-17 084200](https://github.com/user-attachments/assets/75a0b91d-624e-4d03-ab26-cdebe88a502b)

### data.head() for salary

![Screenshot 2025-05-17 084224](https://github.com/user-attachments/assets/dbccc21f-c851-4cff-9326-5ed0ab0c61f4)

### MSE and r2 value:

![Screenshot 2025-05-17 084235](https://github.com/user-attachments/assets/3bf2bda8-6d5e-4923-8b18-99afa2488a0b)

### data prediction:

![Screenshot 2025-05-17 084249](https://github.com/user-attachments/assets/624df11b-ff59-42ff-a309-af9da8ccd00b)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
