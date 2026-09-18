# Chapter 2 — `fit()` Method Deep Dive

Ab hum Scikit-learn ke sabse important methods me se ek **`fit()`** ko detail me samjhenge.

Agar `fit()` clear ho gaya, to sklearn ka bahut bada part automatically easy lagne lagega.

---

## 1. `fit()` kya hota hai?

Simple definition:

> **`fit()` estimator ko diye gaye data se required information ya parameters learn karwata hai.**

Sabse important word hai:

```text
LEARN
```

Mental model:

```text
Data
 ↓
fit()
 ↓
Estimator data ko analyze karta hai
 ↓
Required parameters learn karta hai
```

Example:

```python
model.fit(X, y)
```

Iska matlab roughly:

> Model, `X` aur `y` ko use karke relationship/pattern seekho.

---

# 2. `fit()` sirf ML models ke liye nahi hota

Ye bahut important hai.

Tum ye dekhoge:

```python
model.fit(X, y)
```

Lekin ye bhi hota hai:

```python
scaler.fit(X)
```

Aur:

```python
encoder.fit(X)
```

Aur:

```python
imputer.fit(X)
```

Sab jagah `fit()` hai, lekin har jagah learn ki jane wali cheez alag hai.

---

# 3. `fit()` ka general syntax

Do common forms hain.

### Form 1

```python
estimator.fit(X)
```

Mostly transformers / unsupervised algorithms me.

Example:

```python
scaler.fit(X)
```

### Form 2

```python
estimator.fit(X, y)
```

Mostly supervised machine learning models me.

Example:

```python
model.fit(X, y)
```

---

# 4. `X` aur `y` kya hain?

Suppose dataset:

| Hours | Attendance | Result |
| ----: | ---------: | -----: |
|     2 |         50 |      0 |
|     5 |         75 |      1 |
|     7 |         90 |      1 |
|     1 |         40 |      0 |

Features:

```python
X = [
    [2, 50],
    [5, 75],
    [7, 90],
    [1, 40]
]
```

Target:

```python
y = [0, 1, 1, 0]
```

So:

```text
X
= model ko diye jaane wale input features

y
= jis output ko predict karna hai
```

---

# 5. `fit(X, y)` ka real meaning

Suppose:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X, y)
```

`fit()` ke time model:

```text
X aur y ko dekhta hai
       ↓
unke beech relationship find karta hai
       ↓
best parameters calculate karta hai
       ↓
parameters ko object ke andar save karta hai
```

Is process ko hum generally **training** kehte hain.

---

# 6. Simple Linear Regression example

Suppose data:

```python
X = [
    [1],
    [2],
    [3],
    [4]
]

y = [10, 20, 30, 40]
```

Model:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

Abhi tak:

```python
model
```

ne data se kuch nahi seekha.

Then:

```python
model.fit(X, y)
```

Ab model relationship learn karega.

Roughly:

$$
y = 10x
$$

Model ke learned parameters check kar sakte ho:

```python
print(model.coef_)
print(model.intercept_)
```

Approximately:

```text
coef_ = [10.]
intercept_ = 0
```

Ye values `fit()` ke baad aayi.

---

# 7. `fit()` se pehle vs `fit()` ke baad

Before:

```python
model = LinearRegression()
```

Object create hua hai.

Lekin model ne training data nahi dekha.

After:

```python
model.fit(X, y)
```

Object ke andar learned information aa jati hai.

Visualization:

```text
LinearRegression()

Before fit:
No learned coefficients

       ↓

fit(X, y)

       ↓

After fit:
coef_
intercept_
```

---

# 8. Agar `predict()` ko fit se pehle call karein?

Example:

```python
model = LinearRegression()

model.predict([[5]])
```

Ye error dega, because model trained nahi hai.

Usually error kuch is type ka hota hai:

```text
NotFittedError
```

Correct:

```python
model = LinearRegression()

model.fit(X, y)

model.predict([[5]])
```

Rule:

```text
Create
 ↓
Fit
 ↓
Predict
```

---

# 9. Transformers me `fit()` kya karta hai?

Ab samjho ki `fit()` ka meaning model ke according change hota hai.

### StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
```

StandardScaler kya learn karega?

Mainly:

```text
mean
standard deviation / scale
```

Example:

```python
X = [
    [10],
    [20],
    [30]
]
```

Mean:

```text
20
```

Scaler isko learn karta hai.

Check:

```python
print(scaler.mean_)
```

Output:

```text
[20.]
```

---

# 10. StandardScaler me `fit()` data ko change karta hai?

Nahi.

Ye bahut important distinction hai.

```python
scaler.fit(X)
```

sirf information learn karta hai.

Original data:

```text
10
20
30
```

abhi bhi wahi hai.

Actual data change karne ke liye:

```python
scaler.transform(X)
```

use karte hain.

So:

```text
fit()
= learn

transform()
= learned information use karke data change
```

---

# 11. `fit()` ka output kya hota hai?

Interesting point:

Most sklearn estimators ka:

```python
fit()
```

same estimator object return karta hai.

Example:

```python
result = scaler.fit(X)
```

Usually:

```python
result is scaler
```

True hoga.

Isi wajah se chaining possible hai:

```python
StandardScaler().fit(X).transform(X)
```

Lekin readability ke liye normally hum:

```python
scaler = StandardScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)
```

likhte hain.

---

# 12. OneHotEncoder me `fit()`

Suppose:

```text
City
Indore
Dewas
Bhopal
Indore
```

Code:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

encoder.fit(X)
```

Encoder categories learn karega.

Check:

```python
encoder.categories_
```

Conceptually:

```text
Bhopal
Dewas
Indore
```

So:

```text
OneHotEncoder.fit()
= categories learn karna
```

Actual encoding:

```python
encoder.transform(X)
```

---

# 13. SimpleImputer me `fit()`

Suppose:

```python
import numpy as np

X = [
    [10],
    [20],
    [np.nan],
    [40]
]
```

Imputer:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
```

Then:

```python
imputer.fit(X)
```

Ye available values ka mean calculate karega:

$$
\frac{10+20+40}{3}=23.33
$$

Check:

```python
print(imputer.statistics_)
```

Approximately:

```text
[23.33]
```

Ye value imputer ne `fit()` ke time learn ki.

Then:

```python
imputer.transform(X)
```

missing value ko `23.33` se replace karega.

---

# 14. Different estimators different things learn karte hain

Ye table yaad rakhna useful hai:

| Estimator                | `fit()` kya learn karta hai? |
| ------------------------ | ---------------------------- |
| `StandardScaler`         | Mean aur scale               |
| `MinMaxScaler`           | Min aur max                  |
| `SimpleImputer`          | Mean/median/mode etc.        |
| `OneHotEncoder`          | Categories                   |
| `LinearRegression`       | Coefficients + intercept     |
| `LogisticRegression`     | Decision coefficients        |
| `DecisionTreeClassifier` | Tree splits                  |
| `KMeans`                 | Cluster centers              |

Isliye `fit()` ka fixed internal logic nahi hai.

Estimator decide karta hai ki usko kya learn karna hai.

---

# 15. `fit(X)` vs `fit(X, y)`

Ye difference important hai.

## `fit(X)`

Use hota hai jab estimator ko target ki need nahi.

Example:

```python
scaler.fit(X)
```

Scaler ko bas features ka distribution chahiye.

Target ka scaling se relation nahi.

---

## `fit(X, y)`

Use hota hai jab estimator ko inputs aur expected outputs ka relationship seekhna hai.

Example:

```python
model.fit(X, y)
```

Linear Regression ko ye seekhna hai:

```text
Features → Target
```

Isliye dono chahiye.

---

# 16. Supervised Learning me `fit(X, y)`

Examples:

```python
LinearRegression()
LogisticRegression()
RandomForestClassifier()
DecisionTreeClassifier()
SVC()
```

Typically:

```python
model.fit(X_train, y_train)
```

Because model ko correct answers ki need hoti hai.

Example:

```text
X               y

Hours           Marks
1               20
2               30
3               40
```

Model relationship seekhta hai.

---

# 17. Unsupervised learning me `fit(X)`

Example:

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)

model.fit(X)
```

Yahan:

```text
y nahi hai
```

Algorithm khud data structure discover karta hai.

KMeans `fit()` ke time cluster centers learn karega.

---

# 18. Learned attributes me `_` kyon hota hai?

Sklearn convention:

> Training ke baad learn hone wale public attributes ke naam ke end me usually `_` hota hai.

Example StandardScaler:

```python
scaler.mean_
scaler.var_
scaler.scale_
```

LinearRegression:

```python
model.coef_
model.intercept_
```

KMeans:

```python
model.cluster_centers_
```

Ye signal deta hai:

```text
Ye information fit() ke through learn hui hai.
```

---

# 19. Hyperparameter vs learned parameter

Very important distinction.

Suppose:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5
)
```

`n_estimators`:

```text
100
```

Humne diya.

`max_depth`:

```text
5
```

Humne diya.

Ye **hyperparameters** hain.

After:

```python
model.fit(X, y)
```

model trees learn karega.

Ye learned information hai.

Mental model:

```text
Before fit
→ Hyperparameters

fit()

After fit
→ Learned parameters/state
```

---

# 20. `fit()` existing model ko overwrite kar sakta hai

Suppose:

```python
model.fit(X1, y1)
```

Model ne dataset 1 seekha.

Phir:

```python
model.fit(X2, y2)
```

Normally sklearn me second `fit()` model ko dobara train karega aur previous learned state replace ho sakti hai.

Conceptually:

```text
fit(X1, y1)
→ Model learns Dataset 1

fit(X2, y2)
→ Model gets fitted again on Dataset 2
```

Ye `partial_fit()` se different hai.

---

# 21. `partial_fit()` kya hota hai?

Kuch sklearn estimators incremental learning support karte hain.

Example:

```python
model.partial_fit(X_batch1, y_batch1)

model.partial_fit(X_batch2, y_batch2)
```

Concept:

```text
Batch 1
 ↓
learn

Batch 2
 ↓
continue learning

Batch 3
 ↓
continue learning
```

Ye large datasets ya streaming data me useful ho sakta hai.

Lekin har estimator:

```python
partial_fit()
```

support nahi karta.

Abhi bas difference remember karo:

```text
fit()
→ normally fresh fitting

partial_fit()
→ previous learning continue karne ke liye
```

Isko advanced section me detail me padhenge.

---

# 22. Training data par hi `fit()` kyon?

Ab ek bahut important Machine Learning rule.

Suppose hum data divide karte hain:

```text
100 rows

80 → Training
20 → Testing
```

Scaler ke saath correct:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Notice:

```python
scaler.fit(X_train)
```

sirf training data par.

Test data par:

```python
transform()
```

only.

---

# 23. Test data par `fit()` kyon nahi karte?

Galat:

```python
scaler.fit(X_train)
X_train = scaler.transform(X_train)

scaler.fit(X_test)
X_test = scaler.transform(X_test)
```

Yahan problem hai.

Train aur test ke liye different mean/std use ho raha hai.

Aur test dataset ki information preprocessing process me use ho gayi.

Ye evaluation ko incorrect bana sakta hai.

Correct:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Same learned parameters both par use karo.

---

# 24. Train data par fit, test data par transform

Is rule ko strong kar lo:

```text
TRAIN DATA

fit()
 ↓
learn parameters


TRAIN DATA
 ↓
transform()


TEST DATA
 ↓
same transform()
```

Example:

Train Age:

```text
20
30
40
```

Mean:

```text
30
```

Scaler:

```python
scaler.fit(X_train)
```

learned:

```text
mean = 30
```

Ab test value:

```text
50
```

aayi.

Hum new mean calculate nahi karenge.

Instead training se learned:

```text
mean = 30
```

use karenge.

Because real world me new data aane par bhi exactly yahi situation hoti hai.

---

# 25. Production ko imagine karo

Suppose model deploy ho gaya.

Training data:

```text
100,000 customers
```

Model ko train kiya:

```python
model.fit(X_train, y_train)
```

Ab kal ek new customer aata hai.

Kya model ko dubara us one customer par fit karoge?

No.

Instead:

```python
prediction = model.predict(new_customer)
```

Because training already ho chuki.

Same preprocessing:

```python
scaler.transform(new_customer)
```

not:

```python
scaler.fit(new_customer)
```

---

# 26. `fit()` aur `transform()` me exact difference

Suppose:

```python
scaler = StandardScaler()
```

### Step 1

```python
scaler.fit(X_train)
```

Meaning:

```text
X_train ka mean/std find karo
aur store karo.
```

### Step 2

```python
scaler.transform(X_train)
```

Meaning:

```text
Jo mean/std seekha tha,
usko use karke X_train scale karo.
```

### Step 3

```python
scaler.transform(X_test)
```

Meaning:

```text
Wahi training mean/std
X_test par apply karo.
```

---

# 27. `fit_transform()` kahan se aata hai?

Hum next chapter me detail karenge.

Lekin short version:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
```

ko shortcut me:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

likh sakte hain.

But test data:

```python
X_test_scaled = scaler.transform(X_test)
```

not:

```python
scaler.fit_transform(X_test)
```

Ye distinction bahut important hai.

---

# 28. Complete correct example

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
```

Dataset:

```python
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
```

Split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
```

Scaler:

```python
scaler = StandardScaler()
```

Learn only from training:

```python
scaler.fit(X_train)
```

Transform:

```python
X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Model:

```python
model = LogisticRegression()
```

Training:

```python
model.fit(X_train_scaled, y_train)
```

Prediction:

```python
prediction = model.predict(X_test_scaled)
```

Full mental flow:

```text
Raw Data
   ↓
train_test_split
   ↓
             X_train            X_test
                ↓
          scaler.fit()
                ↓
         learn mean/std
           ↙          ↘
transform train     transform test
      ↓                  ↓
 X_train_scaled      X_test_scaled
      ↓
model.fit(X_train_scaled, y_train)
      ↓
model learns
      ↓
model.predict(X_test_scaled)
```

---

# 29. Common mistake #1 — Fit model on test data

Wrong:

```python
model.fit(X_test, y_test)
```

Then evaluate same test set.

Testing data ka role model ko sikhana nahi hai.

Correct:

```python
model.fit(X_train, y_train)
```

Then:

```python
model.predict(X_test)
```

---

# 30. Common mistake #2 — Fit scaler before split

Suppose:

```python
scaler.fit(X)

X_scaled = scaler.transform(X)

train_test_split(X_scaled, y)
```

Problem:

Scaler ne test portion ko bhi pehle hi dekh liya.

Correct order:

```text
Data
 ↓
Split
 ↓
fit preprocessing on training data only
```

Code:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler.fit(X_train)
```

Is problem ko:

```text
Data Leakage
```

kehte hain.

Isko later dedicated chapter me deeply karenge.

---

# 31. Common mistake #3 — Test set par scaler fit karna

Wrong:

```python
train_scaler = StandardScaler()
test_scaler = StandardScaler()

X_train = train_scaler.fit_transform(X_train)

X_test = test_scaler.fit_transform(X_test)
```

Correct:

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```

One scaler.

One training fit.

---

# 32. Common mistake #4 — Wrong dimensions

Suppose:

```python
X = [1, 2, 3, 4]
```

LinearRegression often expects:

```text
2D X
```

Correct:

```python
X = [
    [1],
    [2],
    [3],
    [4]
]
```

Shape:

```text
4 rows × 1 feature
```

`y`:

```python
y = [10, 20, 30, 40]
```

1D reh sakta hai.

---

# 33. `X.shape` ko samjho

Suppose:

```python
X.shape
```

returns:

```text
(1000, 5)
```

Meaning:

```text
1000 samples
5 features
```

`fit()`:

```python
model.fit(X, y)
```

expects that `y` me normally 1000 target values hon.

```text
X:
1000 rows

y:
1000 values
```

---

# 34. Fit ke baad parameters inspect karna

Scikit-learn seekhne ka best way hai:

> `fit()` ke baad dekho estimator ne kya learn kiya.

Example:

```python
scaler.fit(X)
```

Then:

```python
print(scaler.mean_)
print(scaler.scale_)
```

Imputer:

```python
imputer.fit(X)

print(imputer.statistics_)
```

Encoder:

```python
encoder.fit(X)

print(encoder.categories_)
```

Linear regression:

```python
model.fit(X, y)

print(model.coef_)
print(model.intercept_)
```

Isse tumhara intuition strong hoga.

---

# 35. Ek important question: `fit()` algorithm ko run karta hai?

Generally yes.

Example LinearRegression:

```python
model.fit(X, y)
```

tab algorithm required mathematical calculations perform karta hai aur best coefficients find karta hai.

Decision Tree:

```python
tree.fit(X, y)
```

algorithm different splits evaluate karke tree construct karta hai.

KMeans:

```python
kmeans.fit(X)
```

algorithm cluster centers search karta hai.

So:

> `fit()` ek universal method name hai, lekin andar chalne wala algorithm estimator ke hisaab se different hota hai.

---

# 36. `fit()` aur training exactly same hain?

Model ke context me mostly yes.

Example:

```python
model.fit(X_train, y_train)
```

bol sakte ho:

```text
Train the model
```

Lekin transformer ke liye "training" word thoda confusing ho sakta hai.

Example:

```python
scaler.fit(X_train)
```

Isse hum better bolenge:

```text
Fit the scaler
```

ya:

```text
Learn scaling parameters
```

So broad definition:

```text
fit()
= Learn from data
```

---

# 37. Interview-level understanding

Agar interviewer puche:

**What does `fit()` do in Scikit-learn?**

Strong answer:

> `fit()` learns estimator-specific parameters from the supplied training data. For a predictive model these may be model coefficients or tree structures, while for transformers they can be statistics such as means, scales, or categories. The learned state is then used by methods like `predict()` or `transform()`.

Simple Hinglish:

> `fit()` training data ko use karke estimator ke required parameters learn karta hai. Model me ye model parameters ho sakte hain, aur transformer me mean, standard deviation ya categories jaise statistics ho sakte hain.

---

# 38. Chapter 2 ka most important rule

Agar is chapter se sirf ek cheez yaad rakhni ho:

```text
fit()
=
LEARN FROM DATA
```

Aur ML workflow me:

```text
Train data
   ↓
fit
   ↓
learn

New/Test data
   ↓
use learned information
```

---

# Chapter 2 Summary

```text
fit()
→ estimator ko data se information learn karwata hai

fit(X)
→ jab target ki zarurat nahi

fit(X, y)
→ supervised learning

StandardScaler.fit()
→ mean/scale learn

OneHotEncoder.fit()
→ categories learn

SimpleImputer.fit()
→ replacement statistics learn

LinearRegression.fit()
→ coefficients/intercept learn
```

Most important:

```text
Training data
→ fit()

Test/new data
→ normally fit nahi
→ transform() / predict()
```

And:

```text
Hyperparameters
→ hum set karte hain

Learned parameters
→ fit() learn karta hai
```

### Quick check

Ye socho:

```python
imputer = SimpleImputer(strategy="median")

imputer.fit(X_train)
```

`fit()` yahan kya learn karega?

Aur:

```python
model = RandomForestClassifier()

model.fit(X_train, y_train)
```

yahan `X_train` aur `y_train` dono kyon chahiye?

**Next Chapter 3: `transform()` Deep Dive** — `transform()` exactly kya karta hai, training/test data par kaise use hota hai, `fit()` vs `transform()`, aur new/unseen data ko safely transform kaise karte hain.
