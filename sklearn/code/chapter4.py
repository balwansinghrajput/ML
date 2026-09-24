from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np


# X = np.array([

#     [1,2,3],
#     [4,5,6],
#     [7,8,9],
#     [10,11,12]
# ])

# y = np.array([
#     1,
#     2,
#     3,
#     4
# ])

# scaler = StandardScaler()

# X_scaled = scaler.fit_transform(X)

# print(X_scaled)


# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )

# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train)

# X_test_scaled = scaler.transform(X_test)

# print(X_train_scaled)
# print(X_test_scaled)



# X = np.array([

#     [1,2,3],
#     [4,5,np.nan],
#     [7,np.nan,9],
#     [np.nan,11,12]
# ])

# y = np.array([
#     1,
#     2,
#     3,
#     4
# ])

# imputer = SimpleImputer(
#     strategy="mean"
# )

# X_clean = imputer.fit_transform(X)

# print(X_clean)  


# X = np.array([

#     ["Indore"],
#     ["Dewas"],
#     ["Bhopal"],
#     ["Indore"],
#     ["Bhopal"]
# ])

# y = np.array([
#     1,
#     2,
#     3,
#     4,
#     5
# ])

# encoder = OneHotEncoder(
#     handle_unknown="ignore"
# )

# X_encoded = encoder.fit_transform(X)

# print(X_encoded)


X = np.array([
    ["Father"],
    ["Mother"],
    ["Father"],
    ["Mother"],
    ["Father"],
    ["Mother"],
    ["Father"],
    ["Mother"],
])

y = np.array([1, 0, 1, 0, 1, 0, 1, 0])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    LogisticRegression()
)

model.fit(X_train, y_train)

print(model.predict(X_test))