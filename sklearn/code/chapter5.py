# from sklearn.metrics._classification import accuracy_score
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.linear_model import LinearRegression , LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# import numpy as np
# import pandas as pd

# X = np.array([
#     [1],
#     [2],
#     [3],
#     [4]
# ])

# y = np.array([
#     10,
#     20,
#     30,
#     40
# ])

# model = LinearRegression()

# model.fit(X, y)

# print(model.predict([[999]]))

# the dataset have 100 rows and 8 features they are not good data set only for prectice

# df = pd.read_csv("../datasets/house.csv")

# head = df.head()

# x = df.drop(["id", "price"] , axis=1)
# y = df["price"]

# x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2 , random_state=42)

# scaler = StandardScaler()

# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)

# model = LogisticRegression()
# model.fit(x_train_scaled , y_train)

# pred = model.predict(x_test_scaled)
# accuracy = accuracy_score(y_test , pred)

# print("Accuracy :" , accuracy )

# # print("x_train :" , x_train.shape)
# # print("x_test :" , x_test.shape)
# # print("y_train :" , y_train.shape)
# # print("y_test :" , y_test.shape)