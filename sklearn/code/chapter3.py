from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np


# x = np.array([10, 20, 30, 23]).reshape(-1, 1)

# x_train, x_test = train_test_split(x, test_size=0.2, random_state=42)

# print("x_train:", x_train)
# print("x_test:", x_test)

# scaler = StandardScaler()

# scaler.fit(x_train)
# X_train_scaled = scaler.transform(x_train)

# scaler.fit(x_test)
# X_test_scaled = scaler.transform(x_test)


# print("x_train_scaled:", X_train_scaled)
# print("x_test_scaled:", X_test_scaled)


# X = [
#     [10],
#     [20],
#     [np.nan],
#     [40]
# ]


# imputer = SimpleImputer(strategy="mean")

# imputer.fit(X)

# X_imputed = imputer.transform(X)

# print(X_imputed)


X = np.array([["Indore"],
                ["Dewas"],  
                ["Bhopal"],
                ["Indore"]])


print(X)

encode = OneHotEncoder(handle_unknown="ignore")

encode.fit(X)

print(encode.categories_)
print(encode.transform(X))