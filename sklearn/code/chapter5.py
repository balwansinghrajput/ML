from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression , LogisticRegression

import numpy as np

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


X = np.array([
    ["kuldeep"],
    ["kuldeep"],
    ["parth"],
    ["parth"],
    ["parth"],
    ["parth"],
    ["parth"],
    ["parth"],
    ["parth"]
])

y = np.array([
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0
])

encoder = OneHotEncoder()

X = encoder.fit_transform(X)

model = LogisticRegression()


model.fit(X, y)

new_data = encoder.transform([["parth"]])
print(model.predict(new_data))
