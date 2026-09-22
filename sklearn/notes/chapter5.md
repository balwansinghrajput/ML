# Chapter 5 — `predict()` Method Deep Dive

Ab hum Scikit-learn ke ek aur core method **`predict()`** ko detail me samjhenge.

Pichhle chapters me:

```text
fit()
= learn

transform()
= learned preprocessing apply

fit_transform()
= learn + transform
```

Ab:

```text
predict()
= trained model se output nikalna
```

---

## 1. `predict()` kya hota hai?

Simple definition:

> **`predict()` trained/fitted model ko new ya unseen input deta hai aur model us input ke liye predicted output return karta hai.**

Basic flow:

```text
Training Data
   ↓
model.fit(X_train, y_train)
   ↓
model pattern learn karta hai
   ↓
X_test / new data
   ↓
model.predict(...)
   ↓
Prediction
```

Example:

```python
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Yahan:

```text
fit()
→ model ko sikhaya

predict()
→ sikhe hue model se answer liya
```

---

# 2. `predict()` se pehle `fit()` kyon zaroori hai?

Suppose:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

Ab model create ho gaya hai, lekin usne kuch seekha nahi.

Agar directly:

```python
model.predict([[5]])
```

karoge to error milega, usually:

```text
NotFittedError
```

Correct:

```python
model.fit(X, y)

model.predict([[5]])
```

Rule:

```text
Create Model
    ↓
fit()
    ↓
Learn
    ↓
predict()
```

---

# 3. Simple Regression example

Suppose:

```python
X = [
    [1],
    [2],
    [3],
    [4]
]

y = [10, 20, 30, 40]
```

Train:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X, y)
```

Model roughly relationship seekhega:

```text
y = 10 × x
```

Now:

```python
prediction = model.predict([[5]])

print(prediction)
```

Output approximately:

```text
[50.]
```

Meaning:

```text
Input = 5
Prediction = 50
```

---

# 4. `predict()` actually kya use karta hai?

`predict()` khud training nahi karta.

Wo `fit()` ke time learned information use karta hai.

Example Linear Regression:

```python
model.fit(X, y)
```

Model learns:

```text
coef_
intercept_
```

Suppose:

```text
coef_ = 10
intercept_ = 0
```

Then:

```python
model.predict([[5]])
```

internally idea kuch aisa hai:

$$
y = mx + c
$$

$$
y = 10(5)+0 = 50
$$

So:

```text
fit()
→ m aur c learn

predict()
→ m aur c ko new data par use
```

---

# 5. Regression me `predict()` kya return karta hai?

Regression me target continuous numerical value hota hai.

Examples:

```text
House Price
Salary
Temperature
Sales
Weight
```

Suppose:

```python
y_pred = model.predict(X_test)
```

Output ho sakta hai:

```text
[250000.5, 310000.2, 180500.7]
```

Ye continuous predictions hain.

---

# 6. Classification me `predict()` kya return karta hai?

Classification me classes/categories predict hoti hain.

Example:

```text
Pass / Fail
Spam / Not Spam
Fraud / Not Fraud
Disease / No Disease
```

Code:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Output:

```text
[0, 1, 1, 0, 1]
```

For example:

```text
0 = Fail
1 = Pass
```

So classification me `predict()` usually final class label deta hai.

---

# 7. Regression vs Classification prediction

| Problem        | `predict()` output |
| -------------- | ------------------ |
| Regression     | Continuous number  |
| Classification | Class label        |

Example regression:

```python
model.predict([[1200]])
```

Could return:

```text
[4500000.]
```

House price.

Classification:

```python
model.predict([[35, 70000]])
```

Could return:

```text
[1]
```

Purchased = Yes.

---

# 8. `predict()` aur `transform()` me difference

Ye confuse nahi karna.

### Transformer:

```python
scaler.transform(X_test)
```

Output:

```text
transformed features
```

Example:

```text
Age = 40
Salary = 60000
```

becomes:

```text
0.5
0.8
```

### Predictor:

```python
model.predict(X_test)
```

Output:

```text
final target prediction
```

Example:

```text
Purchased = 1
```

So:

```text
transform()
→ X ko change karta hai

predict()
→ y ka estimate deta hai
```

---

# 9. Complete preprocessing + prediction example

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
```

Split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Scaler:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Model:

```python
model = LogisticRegression()

model.fit(X_train_scaled, y_train)
```

Prediction:

```python
y_pred = model.predict(X_test_scaled)
```

Flow:

```text
X_train
   ↓
scaler.fit_transform()
   ↓
X_train_scaled
   ↓
model.fit()
   ↓
trained model


X_test
   ↓
scaler.transform()
   ↓
X_test_scaled
   ↓
model.predict()
   ↓
y_pred
```

---

# 10. New unseen data par prediction

Suppose model already trained hai.

New customer:

```python
new_customer = [[32, 55000]]
```

Agar scaling use hui thi:

```python
new_customer_scaled = scaler.transform(new_customer)
```

Then:

```python
prediction = model.predict(new_customer_scaled)

print(prediction)
```

Could return:

```text
[1]
```

Meaning:

```text
Purchased = Yes
```

Important:

```text
New data
→ fit nahi

New data
→ transform
→ predict
```

---

# 11. New data par `fit()` kyon nahi?

Wrong:

```python
model.fit(new_customer)
```

Model ko training dataset par train kiya ja chuka hai.

Production me normally:

```python
model.predict(new_customer)
```

use hota hai.

Mental model:

```text
Training Phase
→ fit()

Inference / Prediction Phase
→ predict()
```

`Inference` word ML me prediction phase ke liye commonly use hota hai.

---

# 12. `predict()` ka return usually array kyon hota hai?

Suppose:

```python
prediction = model.predict([[5]])
```

Even ek sample hai, output:

```text
[50.]
```

aata hai.

Why?

Because sklearn generally multiple samples predict karne ke liye designed hai.

Example:

```python
model.predict([
    [5],
    [6],
    [7]
])
```

Output:

```text
[50., 60., 70.]
```

So input samples:

```text
3
```

Output predictions:

```text
3
```

---

# 13. Ek prediction ko scalar kaise nikalein?

Suppose:

```python
prediction = model.predict([[5]])
```

Result:

```text
[50.]
```

Single value:

```python
value = prediction[0]
```

Now:

```text
50.0
```

Example API me:

```python
return {
    "prediction": prediction[0]
}
```

Lekin NumPy types ke saath kabhi serialization conversion ki need ho sakti hai.

---

# 14. Input 2D kyon hona chahiye?

Bahut common beginner error.

Suppose model one feature par trained hai.

Wrong:

```python
model.predict([5])
```

Ye 1D input hai.

Scikit-learn commonly expects:

```text
(samples, features)
```

Correct:

```python
model.predict([[5]])
```

Shape:

```text
1 sample
1 feature
```

---

# 15. Multiple features ka input

Suppose model trained on:

```text
Age
Salary
Experience
```

New person:

```text
Age = 30
Salary = 50000
Experience = 5
```

Correct:

```python
new_data = [[30, 50000, 5]]

prediction = model.predict(new_data)
```

Shape:

```text
1 row × 3 features
```

---

# 16. Multiple samples

```python
new_data = [
    [30, 50000, 5],
    [45, 90000, 15],
    [22, 25000, 1]
]

prediction = model.predict(new_data)
```

Could return:

```text
[0, 1, 0]
```

Meaning:

```text
Sample 1 → class 0
Sample 2 → class 1
Sample 3 → class 0
```

---

# 17. Feature count same hona chahiye

Suppose model trained on:

```text
3 features
```

```text
Age
Salary
Experience
```

Then prediction input bhi 3 features ka hona chahiye.

Wrong:

```python
model.predict([[30, 50000]])
```

Only 2 features.

Correct:

```python
model.predict([[30, 50000, 5]])
```

---

# 18. Feature order bhi same hona chahiye

Training:

```text
Age | Salary | Experience
```

Prediction me:

```text
Salary | Experience | Age
```

bhej diya to model wrong meaning samajh sakta hai.

Example:

Expected:

```text
30, 50000, 5
```

Wrong:

```text
50000, 5, 30
```

Model samjhega:

```text
Age = 50000
Salary = 5
Experience = 30
```

Obviously wrong.

Isliye feature order consistent hona chahiye.

---

# 19. Pandas DataFrame prediction

Agar model DataFrame features se train hua hai:

```python
X_train = df[
    ["Age", "Salary", "Experience"]
]
```

New data DataFrame:

```python
import pandas as pd

new_data = pd.DataFrame({
    "Age": [30],
    "Salary": [50000],
    "Experience": [5]
})
```

Then:

```python
prediction = model.predict(new_data)
```

Columns clear rehne ki wajah se code easier to maintain ho sakta hai.

---

# 20. `y_test` aur `y_pred`

Model evaluation me ye do terms bahut common hain.

Actual answers:

```python
y_test
```

Model answers:

```python
y_pred = model.predict(X_test)
```

Compare:

```text
y_test     y_pred

1          1
0          1
1          1
0          0
```

Isi comparison se baad me hum:

```text
Accuracy
Precision
Recall
F1
MAE
MSE
R²
```

nikalenge.

---

# 21. `predict()` accuracy calculate nahi karta

Important.

```python
y_pred = model.predict(X_test)
```

sirf predictions deta hai.

Accuracy ke liye separate metric:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    y_pred
)
```

So:

```text
predict()
→ answers

metrics
→ answers kitne correct hain
```

---

# 22. `predict_proba()` kya hota hai?

Classification me sometimes sirf final class nahi, probability bhi chahiye.

Example:

```python
model.predict(X_test)
```

might return:

```text
[1]
```

But hume jaana hai:

```text
Class 0 probability?
Class 1 probability?
```

Use:

```python
model.predict_proba(X_test)
```

Example:

```text
[[0.20, 0.80]]
```

Meaning:

```text
Class 0 → 20%
Class 1 → 80%
```

Final prediction:

```text
Class 1
```

---

# 23. `predict()` vs `predict_proba()`

Suppose:

```python
model.predict([[...]])
```

returns:

```text
[1]
```

While:

```python
model.predict_proba([[...]])
```

returns:

```text
[[0.15, 0.85]]
```

Meaning:

```text
15% → class 0
85% → class 1
```

Comparison:

| Method            | Output              |
| ----------------- | ------------------- |
| `predict()`       | Final class         |
| `predict_proba()` | Class probabilities |

---

# 24. Example Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
```

Class:

```python
prediction = model.predict(X_test)
```

Probability:

```python
probability = model.predict_proba(X_test)
```

Suppose:

```text
predict:
[1]
```

and:

```text
predict_proba:
[[0.25, 0.75]]
```

Means:

```text
class 0 probability = 25%
class 1 probability = 75%
```

---

# 25. Probabilities ka column order kaise pata chalega?

Important.

Use:

```python
print(model.classes_)
```

Suppose:

```text
[0 1]
```

Then:

```text
predict_proba output

[0.25, 0.75]
   ↓      ↓
class 0  class 1
```

If classes:

```text
["cat", "dog"]
```

Then probability columns same class order follow karenge.

---

# 26. `predict_proba()` har model me available hai?

No.

Har estimator probability output support nahi karta.

Example some classifiers support:

```text
LogisticRegression
RandomForestClassifier
DecisionTreeClassifier
KNeighborsClassifier
Naive Bayes models
```

But availability estimator/configuration par depend karti hai.

So blindly assume mat karo:

```python
model.predict_proba(...)
```

har classifier me available hoga.

---

# 27. SVM ka example

`SVC` me probability predictions ke liye often:

```python
SVC(probability=True)
```

configure karna padta hai.

Example:

```python
from sklearn.svm import SVC

model = SVC(probability=True)

model.fit(X_train, y_train)

model.predict_proba(X_test)
```

Ye model-specific behavior hai.

Later SVM chapter me detail me karenge.

---

# 28. `decision_function()` kya hai?

Kuch classifiers:

```python
model.decision_function(X)
```

provide karte hain.

Ye probability necessarily nahi hoti.

Ye decision score / distance-like score ho sakta hai.

Example:

```text
negative score → class 0 side
positive score → class 1 side
```

For now:

```text
predict()
→ final class

predict_proba()
→ probability, if supported

decision_function()
→ decision score, if supported
```

---

# 29. Prediction threshold ka basic idea

Binary classification me model probability calculate kar sakta hai:

```text
P(class 1) = 0.80
```

Default decision could conceptually lead to:

```text
0.80 > threshold
→ class 1
```

Example:

```text
Probability = 0.90
→ 1

Probability = 0.20
→ 0
```

Threshold selection classification metrics ke chapter me depth me samjhenge.

---

# 30. Regression me `predict_proba()` kyon nahi?

Regression target:

```text
House price = ₹35,00,000
```

Class probability concept nahi hai.

So:

```python
LinearRegression().predict(...)
```

valid.

But:

```python
LinearRegression().predict_proba(...)
```

not valid.

Because:

```text
Regression
→ continuous prediction

Classification
→ class / probability
```

---

# 31. `predict()` training data par use kar sakte hain?

Technically yes.

```python
model.predict(X_train)
```

But evaluation ke liye sirf training performance dekhna enough nahi.

Model ne training data already dekha hai.

True generalization evaluate karne ke liye:

```python
model.predict(X_test)
```

important hai.

Later overfitting me dekhenge:

```text
Train score high
Test score low
→ possible overfitting
```

---

# 32. Training prediction vs test prediction

You can do:

```python
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)
```

Then compare both performances.

Useful for detecting:

```text
Underfitting
Good fit
Overfitting
```

---

# 33. Does `predict()` model ko modify karta hai?

Normally no.

```python
model.predict(X_test)
```

model ke learned parameters ko use karta hai.

Wo typically model ko retrain nahi karta.

So same trained model:

```python
model.predict(data1)
model.predict(data2)
model.predict(data3)
```

baar-baar use ho sakta hai.

---

# 34. Prediction deterministic hota hai?

Many standard sklearn estimators me fitted model + same input ke liye same prediction milti hai.

Example:

```python
model.predict([[5]])
model.predict([[5]])
```

same result.

Lekin training phase me kuch algorithms randomness use kar sakte hain, jaise Random Forest.

Isliye reproducibility ke liye:

```python
random_state=42
```

jaise parameters important hote hain.

Isko later detail me karenge.

---

# 35. Preprocessing exactly same hona chahiye

Suppose training:

```python
X_train_scaled = scaler.fit_transform(X_train)

model.fit(X_train_scaled, y_train)
```

Then prediction me raw data directly:

```python
model.predict(X_test)
```

dena wrong ho sakta hai.

Correct:

```python
X_test_scaled = scaler.transform(X_test)

model.predict(X_test_scaled)
```

Because model ne scaled data par learn kiya tha.

Rule:

```text
Training preprocessing
=
Prediction preprocessing
```

---

# 36. Real production example

Suppose salary prediction model.

Training:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

model.fit(X_train_scaled, y_train)
```

API receives:

```python
new_employee = [[3, 2, 7]]
```

Maybe features:

```text
Experience
Projects
Skill Score
```

API flow:

```python
processed = scaler.transform(new_employee)

prediction = model.predict(processed)
```

Then response:

```python
{
    "predicted_salary": prediction[0]
}
```

This is real-world inference flow.

---

# 37. Pipeline later isko easy bana dega

Without Pipeline:

```python
X_new_scaled = scaler.transform(X_new)

prediction = model.predict(X_new_scaled)
```

With Pipeline:

```python
prediction = pipeline.predict(X_new)
```

Pipeline automatically:

```text
raw input
↓
preprocessing
↓
model
↓
prediction
```

handle karegi.

Isi liye later Pipeline important hogi.

---

# 38. Common mistake #1 — Predict before fit

Wrong:

```python
model = LogisticRegression()

model.predict(X_test)
```

Correct:

```python
model.fit(X_train, y_train)

model.predict(X_test)
```

---

# 39. Common mistake #2 — Wrong shape

Wrong:

```python
model.predict([30, 50000])
```

Correct:

```python
model.predict([[30, 50000]])
```

Remember:

```text
Outer list
→ samples

Inner list
→ features
```

---

# 40. Common mistake #3 — Different feature order

Train:

```text
Age | Salary
```

Prediction:

```text
Salary | Age
```

Wrong.

Same order maintain karo.

---

# 41. Common mistake #4 — Missing preprocessing

Model trained:

```python
model.fit(X_train_scaled, y_train)
```

But prediction:

```python
model.predict(X_test)
```

Wrong input representation.

Correct:

```python
model.predict(X_test_scaled)
```

---

# 42. Common mistake #5 — Fitting test data

Wrong:

```python
model.fit(X_test, y_test)

prediction = model.predict(X_test)
```

Test set ka purpose evaluation hai.

Correct:

```python
model.fit(X_train, y_train)

prediction = model.predict(X_test)
```

---

# 43. Common mistake #6 — Target column input me dena

Suppose:

```text
Age | Salary | Purchased
```

Purchased target hai.

Model training:

```python
X = df[["Age", "Salary"]]

y = df["Purchased"]
```

Prediction input should contain:

```text
Age
Salary
```

Not:

```text
Age
Salary
Purchased
```

Because target prediction ke waqt unknown hota hai.

---

# 44. Prediction output ko interpret karna

Suppose:

```python
prediction = model.predict(X_test)
```

Output:

```text
[2, 0, 1]
```

Ye numbers ka meaning dataset/class mapping par depend karega.

Always know:

```python
model.classes_
```

Classification me useful hai.

Example:

```python
print(model.classes_)
```

Output:

```text
['high', 'low', 'medium']
```

Then numerical assumptions mat karo.

---

# 45. `predict()` internally algorithm ke according different hota hai

Universal method name same hai:

```python
model.predict(X)
```

But algorithms different.

### Linear Regression

Uses learned equation.

### KNN

Nearby training points dekhta hai.

### Decision Tree

Tree decisions follow karta hai.

### Random Forest

Multiple trees ki predictions combine karta hai.

### Logistic Regression

Decision score/probabilities based classification karta hai.

So:

> `predict()` API same hai, but internal logic model-specific hai.

Ye sklearn ki consistency ka major advantage hai.

---

# 46. `predict()` is inference

Industry me tum ye words dekhoge:

```text
Training
Inference
```

Training:

```python
model.fit(X_train, y_train)
```

Inference:

```python
model.predict(X_new)
```

Simple:

```text
Training
= model seekhta hai

Inference
= model learned knowledge use karta hai
```

---

# 47. Four core methods ab tak

Ab tumhare paas ye complete picture honi chahiye:

| Method            | Meaning              |
| ----------------- | -------------------- |
| `fit()`           | Learn                |
| `transform()`     | Apply transformation |
| `fit_transform()` | Learn + transform    |
| `predict()`       | Predict target       |

Example complete:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
```

Mental translation:

```text
Scaler:
learn + transform train

Scaler:
transform test

Model:
learn X → y relationship

Model:
predict test y
```

---

# 48. `predict_proba()` quick summary

Classification:

```python
model.predict(X_test)
```

returns:

```text
Final class
```

Example:

```text
[1]
```

While:

```python
model.predict_proba(X_test)
```

could return:

```text
[[0.12, 0.88]]
```

Meaning:

```text
class 0 = 12%
class 1 = 88%
```

Final:

```text
class 1
```

---

# Chapter 5 Summary

Most important mental model:

```text
fit()
→ learn

predict()
→ learned model use karke output predict
```

Regression:

```text
predict()
→ number
```

Classification:

```text
predict()
→ class
```

Classification probability:

```text
predict_proba()
→ probabilities
```

Correct workflow:

```python
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

With preprocessing:

```python
X_train_new = transformer.fit_transform(X_train)

X_test_new = transformer.transform(X_test)

model.fit(X_train_new, y_train)

y_pred = model.predict(X_test_new)
```

And production:

```text
New Raw Data
   ↓
transform()
   ↓
trained model
   ↓
predict()
   ↓
prediction
```

### Quick practice

Suppose:

```python
model.fit(X_train, y_train)

prediction = model.predict([[35, 60000]])
```

Socho:

1. `predict()` kya model ko dobara train karega?
2. `[[35, 60000]]` me outer aur inner list ka kya meaning hai?
3. Classification me `predict()` `[1]` deta hai aur `predict_proba()` `[[0.2, 0.8]]` deta hai — dono outputs me exact difference kya hai?


