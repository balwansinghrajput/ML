# from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np


# x = [[1], [2], [3], [4], [5]]
# y = [0, 1, 0, 1, 0]

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Initialize and train KNN classifier with 3 neighbors
# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(X_train_scaled, y_train)

# # Make prediction
# prediction = model.predict(X_test_scaled)

# print("Prediction:", prediction)
# print("Actual:", y_test)


# x = [["Indore"],["Dewas"],["Bhopal"],["Indore"]]


# encoder = OneHotEncoder()


# encoder.fit(x)
# # encoder.transform(x)

# print(encoder.categories_)
# print(encoder.transform(x))


# X = [
#     [10],
#     [20],
#     [np.nan],
#     [40]
# ]

# print(X)

# impute = SimpleImputer(strategy="mean")

# impute.fit(X)

# print(impute.statistics_)

# print(impute.transform(X))


X = np.array([
    [20, 20000],
    [25, 30000],
    [30, 40000],
    [35, 50000],
    [40, 60000],
    [45, 70000],
    [50, 80000],
    [55, 90000]
])


y = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1
])


X_train , x_test , Y_train , y_test = train_test_split(X, y, test_size=0.2 , random_state=42)

print("X_train" ,X_train)
print("x_test" ,x_test)
print("Y_train" ,Y_train)
print("y_test" ,y_test)