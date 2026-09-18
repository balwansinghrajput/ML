import sklearn
import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# print(sklearn.__version__)

# X = np.array([[10] , [20] , [30]])
# print('np array' ,X)

# scaler = StandardScaler()

# scaler.fit(X)

# print("scaler mean" ,scaler.mean_)
# print("scaler scale" ,scaler.scale_)
# # print("fit model" , fitModel)

# X_transform = scaler.transform(X)

# print("transform x" ,X_transform)


# from sklearn.linear_model import LinearRegression

# x = np.array([[1], [2], [3], [4]])
# y = np.array([10, 20, 30, 40])

# model = LinearRegression()

# model.fit(x , y)

# prediction = model.predict([[9]])

# print(prediction)


# dataset
# Age      Salary      Purchased
# 22       20000       0
# 30       45000       0
# 40       70000       1
# 50       90000       1


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

data = {
    "Age":[22,30,40,50],
    "Salary":[2000,4500,70000,90000],
    "Purchased":[0,0,1,1],
}

dataframe = pd.DataFrame(data)
# print(dataframe)


x = dataframe.iloc[:,0:2]
print("x" ,x)
y = dataframe.iloc[:,-1]
print("y" ,y)



X_train , x_test , Y_train , y_test= train_test_split(x, y, test_size=0.2)
print("X_train" ,X_train)
print("x_test" ,x_test)
print("Y_train" ,Y_train)
print("y_test" ,y_test)

scaler = StandardScaler()

scaler.fit(X_train)

x_train_scaled = scaler.transform(X_train)

x_test_scaled = scaler.transform(x_test)


print("x_train_scaled" ,x_train_scaled)
print("x_test_scaled" ,x_test_scaled)   


model = LogisticRegression()

model.fit(x_train_scaled , Y_train)

prediction = model.predict(x_test_scaled)

print("prediction" ,prediction)

print("y_test" ,y_test)


