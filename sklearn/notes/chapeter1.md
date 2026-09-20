# Chapter 1 — Scikit-learn Introduction

Aaj hum Scikit-learn ko bilkul foundation se samjhenge. Is chapter ke end tak tumhe clear ho jana chahiye ki **Scikit-learn kya hai, machine learning workflow me iska role kya hai, aur Estimator, Transformer aur Predictor kya hote hain.**

---

## 1. Scikit-learn kya hai?

**Scikit-learn**, jise commonly `sklearn` bolte hain, Python ki ek Machine Learning library hai.

Iska use hum ML ke almost poore workflow me kar sakte hain:

```text
Raw Data
   ↓
Data Preprocessing
   ↓
Feature Transformation
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Hyperparameter Tuning
```

Example ke liye maan lo hamare paas students ka data hai:

```text
Hours_Studied    Attendance    Result
5                80            Pass
2                45            Fail
8                95            Pass
3                50            Fail
```

Hum ek model banana chahte hain jo predict kare:

```text
New Student
Hours = 6
Attendance = 85

          ↓

Model

          ↓

Prediction = Pass
```

Is poore process ke liye Scikit-learn bahut saare ready-made tools deta hai.

---

# 2. Scikit-learn import kaise karte hain?

Library install:

```bash
pip install scikit-learn
```

Version check:

```python
import sklearn

print(sklearn.__version__)
```

Lekin normally hum poora `sklearn` import nahi karte.

Hum jis functionality ki zarurat hoti hai sirf wahi import karte hain.

For example:

```python
from sklearn.linear_model import LinearRegression
```

Ya:

```python
from sklearn.preprocessing import StandardScaler
```

Ya:

```python
from sklearn.model_selection import train_test_split
```

Iska matlab sklearn ke andar alag-alag **modules** hote hain.

---

# 3. Scikit-learn ka basic structure

Example:

```python
from sklearn.preprocessing import StandardScaler
```

Isko breakdown karo:

```text
sklearn
   ↓
preprocessing
   ↓
StandardScaler
```

`sklearn` main library hai.

`preprocessing` uske andar module hai.

`StandardScaler` us module ke andar ek class hai.

Same:

```python
from sklearn.linear_model import LinearRegression
```

Structure:

```text
sklearn
   ↓
linear_model
   ↓
LinearRegression
```

---

# 4. Important sklearn modules

Abhi memorize mat karna. Course ke saath naturally yaad ho jayenge.

```text
sklearn.preprocessing
→ Scaling, Encoding etc.

sklearn.impute
→ Missing values

sklearn.model_selection
→ Train-test split, Cross-validation

sklearn.linear_model
→ Linear Regression, Logistic Regression

sklearn.tree
→ Decision Tree

sklearn.ensemble
→ Random Forest, Gradient Boosting etc.

sklearn.neighbors
→ KNN

sklearn.svm
→ Support Vector Machine

sklearn.cluster
→ Clustering

sklearn.metrics
→ Model evaluation

sklearn.pipeline
→ Pipeline

sklearn.compose
→ ColumnTransformer
```

Ab sabse important concept par aate hain.

---

# 5. Scikit-learn ka common design

Sklearn ki almost sari classes ek similar pattern follow karti hain.

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Aur:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

Dono alag kaam karte hain, lekin syntax similar hai.

General workflow:

```python
object = SomeClass()

object.fit(data)
```

Uske baad depending on object:

```python
object.transform(data)
```

ya:

```python
object.predict(data)
```

Isi design ko samajhne ke liye hume teen concepts samajhne hain:

**Estimator → Transformer → Predictor**

---

# 6. Estimator kya hota hai?

Ye Scikit-learn ka bahut important concept hai.

Simple definition:

> **Estimator ek aisa object hai jo data se kuch learn karta hai using `fit()`.**

Sabse important word:

```text
fit()
```

Agar koi sklearn object data se parameters learn karta hai, to generally wo **Estimator** hai.

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Abhi:

```python
scaler
```

ne kuch learn nahi kiya.

Jab hum karenge:

```python
scaler.fit(X)
```

tab scaler data ko dekhega aur kuch information learn karega.

Example:

```text
Age
20
30
40
```

StandardScaler calculate karega:

```text
Mean = 30
Standard deviation = ...
```

Ye information usne data se **learn** ki.

Isliye `StandardScaler` ek estimator hai.

---

# 7. Machine Learning model bhi Estimator hai

Example:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

Then:

```python
model.fit(X, y)
```

Model training data ko dekhega aur relationship learn karega.

Example:

```text
Hours → Marks

1 → 20
2 → 30
3 → 40
4 → 50
```

Model kuch aisa relationship learn kar sakta hai:

```text
Marks ≈ 10 × Hours + 10
```

Ye relationship data se learn hua.

Isliye `LinearRegression` bhi **Estimator** hai.

---

# 8. To kya har ML model Estimator hai?

Scikit-learn me generally haan.

Example:

```text
LinearRegression
LogisticRegression
DecisionTreeClassifier
RandomForestClassifier
KNeighborsClassifier
SVC
KMeans
StandardScaler
OneHotEncoder
SimpleImputer
```

Ye sab data se kuch na kuch learn karte hain.

Isliye broadly ye sab sklearn **Estimators** hain.

Lekin inme se kuch Transformer hain aur kuch Predictor.

---

# 9. Estimator ka basic pattern

```python
estimator = SomeEstimator()
```

Then:

```python
estimator.fit(X)
```

ya supervised ML me:

```python
estimator.fit(X, y)
```

General:

```text
Estimator
   ↓
fit()
   ↓
Data se parameters learn
```

---

# 10. Hyperparameters kya hote hain?

Jab estimator create karte hain:

```python
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5
)
```

Yahan:

```text
n_estimators = 100
max_depth = 5
```

ye **hyperparameters** hain.

Ye model data se learn nahi karta.

Ye hum model ko training se pehle dete hain.

Compare:

```text
Hyperparameter
→ Hum set karte hain

Learned Parameter
→ Model data se seekhta hai
```

Ye distinction bahut important hai.

---

# 11. Learned parameters ko sklearn me kaise identify karein?

Scikit-learn me training ke baad learn hone wale attributes ke end me usually underscore `_` hota hai.

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)
```

Training ke baad:

```python
scaler.mean_
```

Aur:

```python
scaler.scale_
```

Notice:

```text
mean_
scale_
```

end me `_` hai.

Matlab ye values training ke baad learn hui hain.

---

# 12. Linear Regression me learned parameters

Example:

```python
model.fit(X, y)
```

Uske baad:

```python
model.coef_
```

Aur:

```python
model.intercept_
```

Ye model ne data se learn kiya.

Difference:

```python
LinearRegression()
```

Parameters jo hum constructor me dete hain:

```text
Hyperparameters
```

Aur training ke baad:

```python
model.coef_
model.intercept_
```

ye:

```text
Learned parameters
```

hain.

---

# 13. Transformer kya hota hai?

Simple definition:

> **Transformer ek estimator hota hai jo data ko learn karne ke baad usko transform/change karta hai.**

Transformer commonly do important methods use karta hai:

```python
fit()
transform()
```

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)
```

Flow:

```text
Original Data

      ↓

fit()

      ↓

Learn mean/std

      ↓

transform()

      ↓

Scaled Data
```

---

# 14. Real Transformer example

Suppose:

```python
import numpy as np

X = np.array([
    [20],
    [30],
    [40]
])
```

Scaler:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Fit:

```python
scaler.fit(X)
```

Check learned mean:

```python
print(scaler.mean_)
```

Approximately:

```text
[30.]
```

Now:

```python
X_scaled = scaler.transform(X)

print(X_scaled)
```

Output approximately:

```text
[[-1.2247]
 [ 0.    ]
 [ 1.2247]]
```

Original:

```text
20
30
40
```

transform hokar:

```text
-1.22
 0
 1.22
```

ho gaya.

Isliye StandardScaler:

```text
Estimator + Transformer
```

dono hai.

---

# 15. Transformer aur Estimator ka relation

Ye bahut important hai.

```text
Estimator
    |
    └── Transformer
```

Transformer bhi estimator hota hai kyunki transformer bhi `fit()` karta hai.

Lekin transformer ke paas extra ability hoti hai:

```python
transform()
```

Isliye:

```text
StandardScaler
```

Estimator bhi hai aur Transformer bhi.

---

# 16. Aur examples of Transformers

Examples:

```text
StandardScaler
MinMaxScaler
RobustScaler

SimpleImputer

OneHotEncoder
OrdinalEncoder

PCA

PolynomialFeatures
```

Inka main kaam hota hai data ko transform karna.

Example:

```text
Raw Data
   ↓
OneHotEncoder
   ↓
Encoded Data
```

ya:

```text
Raw Data
   ↓
SimpleImputer
   ↓
Missing Values Filled
```

---

# 17. Predictor kya hota hai?

Simple definition:

> **Predictor ek estimator hota hai jo data learn karne ke baad new data ke liye prediction karta hai.**

Common methods:

```python
fit()
predict()
```

Example:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

Then:

```python
prediction = model.predict(X_test)
```

Flow:

```text
Training Data
      ↓
    fit()
      ↓
Model learns pattern
      ↓
 New Data
      ↓
  predict()
      ↓
Prediction
```

---

# 18. Simple Predictor example

Suppose:

```python
X = [[1], [2], [3], [4]]
y = [10, 20, 30, 40]
```

Train:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X, y)
```

Now:

```python
prediction = model.predict([[5]])

print(prediction)
```

Approximately:

```text
[50.]
```

Model ne relationship learn ki:

```text
1 → 10
2 → 20
3 → 30
4 → 40
```

Isliye:

```text
5 → 50
```

predict kiya.

---

# 19. Predictor ke examples

Regression:

```text
LinearRegression
Ridge
Lasso
RandomForestRegressor
DecisionTreeRegressor
```

Classification:

```text
LogisticRegression
KNeighborsClassifier
DecisionTreeClassifier
RandomForestClassifier
SVC
```

In sab me usually:

```python
fit()
predict()
```

hota hai.

---

# 20. Estimator vs Transformer vs Predictor

Ye chapter ka sabse important table hai:

| Type        | `fit()` | `transform()` | `predict()` | Main kaam            |
| ----------- | ------: | ------------: | ----------: | -------------------- |
| Estimator   |       ✅ |         Maybe |       Maybe | Data se learn karna  |
| Transformer |       ✅ |             ✅ | ❌ generally | Data ko change karna |
| Predictor   |       ✅ |   ❌ generally |           ✅ | Prediction karna     |

Example:

```text
StandardScaler

fit()       ✅
transform() ✅
predict()   ❌

Therefore:
Estimator + Transformer
```

While:

```text
LinearRegression

fit()       ✅
transform() ❌
predict()   ✅

Therefore:
Estimator + Predictor
```

---

# 21. Ek easy rule yaad rakho

```text
fit()
↓
LEARN
```

```text
transform()
↓
CHANGE DATA
```

```text
predict()
↓
PREDICT OUTPUT
```

Ye teen lines sklearn ke bahut bade portion ko samajhne me help karengi.

---

# 22. Ek real ML workflow

Suppose hamare paas:

```text
Age      Salary      Purchased
22       20000       0
30       45000       0
40       70000       1
50       90000       1
```

Hum Logistic Regression se prediction banana chahte hain.

Sabse pehle scaler:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Train data se learn:

```python
scaler.fit(X_train)
```

Transform:

```python
X_train_scaled = scaler.transform(X_train)
```

Then model:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
```

Train:

```python
model.fit(X_train_scaled, y_train)
```

Predict:

```python
prediction = model.predict(X_test_scaled)
```

Complete flow:

```text
X_train
   ↓
StandardScaler
   ↓
fit()
   ↓
mean/std learned
   ↓
transform()
   ↓
X_train_scaled
   ↓
LogisticRegression
   ↓
fit()
   ↓
pattern learned
   ↓
predict()
   ↓
0 / 1
```

Ab observe karo:

```text
StandardScaler
= Transformer

LogisticRegression
= Predictor

Dono
= Estimators
```

---

# 23. Class aur Object ko bhi clear rakho

Jab hum likhte hain:

```python
StandardScaler
```

ye **class** hai.

Jab:

```python
scaler = StandardScaler()
```

to `scaler` us class ka **object / instance** hai.

Similarly:

```python
LinearRegression
```

class.

```python
model = LinearRegression()
```

object.

Then methods object par use hote hain:

```python
model.fit()
model.predict()
```

---

# 24. Sklearn ka standard pattern

Tum bahut baar ye same pattern dekhoge:

```python
from sklearn.some_module import SomeClass

obj = SomeClass()

obj.fit(X)

result = obj.transform(X)
```

Ya:

```python
from sklearn.some_module import SomeModel

model = SomeModel()

model.fit(X, y)

prediction = model.predict(X_new)
```

Isi consistency ki wajah se sklearn learn karna relatively easy hota hai.

Ek model samajhne ke baad doosre models ka API familiar lagta hai.

---

# 25. Supervised vs Unsupervised fit

Is difference ko abhi basic level par samjho.

Supervised model ko input ke saath target bhi chahiye:

```python
model.fit(X, y)
```

Example:

```python
LinearRegression
LogisticRegression
RandomForestClassifier
```

Yahan:

```text
X = Features
y = Target
```

Lekin preprocessing transformer ko usually target nahi chahiye:

```python
scaler.fit(X)
```

Example:

```text
StandardScaler
MinMaxScaler
PCA
```

---

# 26. `X` capital aur `y` small kyon?

Machine learning convention hai.

Example dataset:

```text
Age   Salary   Purchased

20    20000    0
30    40000    0
40    80000    1
```

Features:

```python
X = [
    [20, 20000],
    [30, 40000],
    [40, 80000]
]
```

X 2-dimensional hai:

```text
rows × features
```

Target:

```python
y = [0, 0, 1]
```

Usually `y` one-dimensional hota hai.

Isliye convention:

```text
X → Matrix
y → Vector
```

Ye topic hum next chapters me aur detail me karenge.

---

# 27. Important misconception

Bahut students sochte hain:

> `fit()` ka matlab model train karna hota hai.

Ye incomplete definition hai.

Better definition:

> **`fit()` ka matlab estimator ko data se required parameters learn karwana hai.**

Model ke case me:

```python
model.fit()
```

= model training.

Scaler ke case me:

```python
scaler.fit()
```

= scaling parameters learn karna.

Encoder ke case me:

```python
encoder.fit()
```

= available categories learn karna.

Imputer:

```python
imputer.fit()
```

= mean/median etc. learn karna.

Ye concept bahut important hai.

---

# 28. Ek aur example — OneHotEncoder

Suppose data:

```text
City
Indore
Dewas
Indore
Bhopal
```

Encoder:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
```

Fit:

```python
encoder.fit(X)
```

Ye categories learn karega:

```text
Bhopal
Dewas
Indore
```

Then:

```python
encoder.transform(X)
```

data convert karega.

Yahan bhi:

```text
fit()
= categories learn

transform()
= categories ko numbers me convert
```

Therefore:

```text
OneHotEncoder
= Estimator + Transformer
```

---

# 29. SimpleImputer example

Suppose:

```text
Age
20
NaN
40
```

Imputer:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
```

Fit:

```python
imputer.fit(X)
```

Learn karega:

```text
mean = 30
```

Transform:

```python
imputer.transform(X)
```

Result:

```text
20
30
40
```

Again:

```text
fit()
→ learn mean

transform()
→ missing value replace
```

---

# 30. Chapter 1 ka mental model

Sklearn ko abhi is tarah imagine karo:

```text
                  Estimator
                     |
          ┌──────────┴──────────┐
          ↓                     ↓
     Transformer            Predictor
          ↓                     ↓
   Data ko change         Output predict
          ↓                     ↓
     transform()            predict()
```

Dono generally:

```text
fit()
```

use karte hain.

---

# Chapter 1 Summary

Bas ye concepts strong hone chahiye:

```text
Scikit-learn
= Python Machine Learning library

Estimator
= Data se learn karta hai
= fit()

Transformer
= Data se learn + data ko change
= fit() + transform()

Predictor
= Data se learn + prediction
= fit() + predict()

Hyperparameters
= Hum set karte hain

Learned parameters
= Model fit() ke through learn karta hai

Learned sklearn attributes
= often underscore "_" par end hote hain
```

Example:

```python
scaler = StandardScaler()

scaler.fit(X)
scaler.transform(X)
```

So:

```text
StandardScaler
→ Estimator + Transformer
```

Example:

```python
model = LinearRegression()

model.fit(X, y)
model.predict(X_new)
```

So:

```text
LinearRegression
→ Estimator + Predictor
```

### Tumhare liye ek chhota test

Bina code chalaye answer socho:

```python
scaler = StandardScaler()
scaler.fit(X)
```

`scaler` ne kya learn kiya?

```python
model = LinearRegression()
model.fit(X, y)
```

`model` ka type kya hua — Transformer ya Predictor?

Aur:

```python
encoder = OneHotEncoder()
encoder.fit(X)
encoder.transform(X)
```

`fit()` aur `transform()` dono alag-alag kya kaam kar rahe hain?
