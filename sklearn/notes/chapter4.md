# Chapter 4 — `fit_transform()` Deep Dive

Ab hum `fit_transform()` ko detail me samjhenge.

Pichhle chapters me:

```text
fit()
= learn

transform()
= learned rule apply
```

Ab:

```text
fit_transform()
= learn + apply
```

Matlab ek hi step me dono kaam.

---

## 1. `fit_transform()` kya hota hai?

Simple definition:

> **`fit_transform()` pehle input data par `fit()` karta hai, phir usi data ko `transform()` karta hai.**

Conceptually:

```python
transformer.fit(X)
X_new = transformer.transform(X)
```

aur:

```python
X_new = transformer.fit_transform(X)
```

same purpose serve karte hain.

Mental model:

```text
X
↓
fit()
↓
parameters learn
↓
transform()
↓
transformed X
```

Shortcut:

```text
X
↓
fit_transform()
↓
transformed X
```

---

# 2. Basic syntax

```python
X_transformed = transformer.fit_transform(X)
```

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Yahan `scaler`:

1. `X` ka mean aur scale learn karega
2. same `X` ko scale karega
3. transformed data return karega

---

# 3. StandardScaler example

Suppose:

```python
X = [
    [10],
    [20],
    [30]
]
```

Without `fit_transform()`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)
```

With `fit_transform()`:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Dono me scaler training data se roughly learn karega:

```text
Mean = 20
Scale ≈ 8.16
```

Then transformed values:

```text
10 → -1.22
20 →  0.00
30 →  1.22
```

---

# 4. `fit_transform()` ka return kya hota hai?

Ye important hai.

```python
result = scaler.fit_transform(X)
```

`result` me:

```text
transformed data
```

aayega.

While:

```python
result = scaler.fit(X)
```

me fitted scaler object return hota hai.

Difference:

| Method            | Return           |
| ----------------- | ---------------- |
| `fit()`           | fitted estimator |
| `transform()`     | transformed data |
| `fit_transform()` | transformed data |

---

# 5. `fit_transform()` ke baad transformer fitted hota hai?

**Yes.**

Example:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Ab scaler ke paas learned attributes available hain:

```python
print(scaler.mean_)
print(scaler.scale_)
```

Because `fit_transform()` ke andar fitting bhi hui hai.

So:

```text
fit_transform()
↓
Transformer fitted bhi hua
+
transformed data bhi mila
```

---

# 6. Sabse important use case: Training data

Normally `fit_transform()` training data par use hota hai.

Correct pattern:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Isko strong kar lo:

```text
TRAIN DATA
→ fit_transform()

TEST DATA
→ transform()
```

Ye pattern bahut common hai.

---

# 7. Test data par `fit_transform()` kyon nahi?

Suppose:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

Correct.

But phir:

```python
X_test_scaled = scaler.fit_transform(X_test)
```

Wrong.

Why?

Because second line means:

```text
X_test se parameters LEARN karo
+
X_test ko transform karo
```

Hume test data se kuch learn nahi karna.

Correct:

```python
X_test_scaled = scaler.transform(X_test)
```

Yahan scaler training data ke learned parameters use karega.

---

# 8. Example se samjho

Training data:

```text
10
20
30
```

Test data:

```text
100
110
```

Training par:

```python
scaler.fit_transform(X_train)
```

Scaler learn karega:

```text
training mean = 20
```

Ab test par correct:

```python
scaler.transform(X_test)
```

To:

```text
100 aur 110
training mean/std ke according scale honge
```

Lekin agar tum:

```python
scaler.fit_transform(X_test)
```

karoge, scaler new mean learn karega:

```text
test mean = 105
```

Ab training aur test ke transformations inconsistent ho gaye.

---

# 9. Ek dangerous side effect

Suppose:

```python
scaler.fit_transform(X_train)
```

After this:

```text
scaler.mean_
= training mean
```

Then accidentally:

```python
scaler.fit_transform(X_test)
```

Ab:

```text
scaler.mean_
= test mean
```

Training ki learned information overwrite ho gayi.

Matlab scaler ka fitted state badal gaya.

Isi wajah se test par `fit_transform()` avoid karte hain.

---

# 10. Correct workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Flow:

```text
Raw Data
   ↓
Train-Test Split
   ↓
   ├─────────────┐
   ↓             ↓
X_train        X_test
   ↓             |
fit_transform()  |
   ↓             |
learn + scale    |
   ↓             ↓
Train scaled   transform()
                 ↓
              Test scaled
```

---

# 11. SimpleImputer ke saath `fit_transform()`

Suppose:

```python
import numpy as np

X_train = [
    [10],
    [20],
    [np.nan],
    [40]
]
```

Create imputer:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
```

Without shortcut:

```python
imputer.fit(X_train)

X_train_clean = imputer.transform(X_train)
```

Shortcut:

```python
X_train_clean = imputer.fit_transform(X_train)
```

Imputer:

```text
fit
→ mean learn

transform
→ missing value replace
```

So `fit_transform()`:

```text
mean learn + missing value replace
```

---

# 12. Test data with SimpleImputer

Suppose training mean:

```text
23.33
```

Test:

```python
X_test = [
    [50],
    [np.nan]
]
```

Correct:

```python
X_test_clean = imputer.transform(X_test)
```

Result conceptually:

```text
50
23.33
```

Not:

```python
imputer.fit_transform(X_test)
```

Because then test values se new statistic learn hoga.

---

# 13. OneHotEncoder ke saath

Training data:

```text
City

Indore
Dewas
Bhopal
Indore
```

Create:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore"
)
```

Training:

```python
X_train_encoded = encoder.fit_transform(X_train)
```

`fit_transform()`:

```text
Step 1:
Categories learn

Bhopal
Dewas
Indore

Step 2:
Same data encode
```

Test:

```python
X_test_encoded = encoder.transform(X_test)
```

Same training categories use hongi.

---

# 14. `fit_transform(X)` vs `fit(X).transform(X)`

Scikit-learn me ye pattern bhi possible hai:

```python
X_new = scaler.fit(X).transform(X)
```

Because `fit()` fitted object return karta hai.

Conceptually:

```python
scaler.fit(X)
scaler.transform(X)
```

ke equal.

Lekin readability ke liye:

```python
X_new = scaler.fit_transform(X)
```

better hota hai.

---

# 15. Kya `fit_transform()` always exactly `fit()` + `transform()` hi hota hai?

Conceptually **yes**.

Lekin implementation-wise ek important advanced point hai:

> Kuch sklearn transformers `fit_transform()` ka optimized implementation provide kar sakte hain.

Matlab:

```python
transformer.fit_transform(X)
```

kabhi-kabhi separate:

```python
transformer.fit(X)
transformer.transform(X)
```

se more efficient ho sakta hai.

But learning ke liye abhi mental model:

```text
fit_transform()
≈ fit() + transform()
```

rakho.

---

# 16. `fit_transform()` `y` bhi le sakta hai?

Method signature commonly aisa ho sakta hai:

```python
transformer.fit_transform(X, y)
```

Kai transformers `y` ignore kar dete hain, but kuch supervised transformers target ko use kar sakte hain.

Example conceptual:

```python
transformer.fit_transform(X_train, y_train)
```

Lekin common preprocessing:

```text
StandardScaler
MinMaxScaler
SimpleImputer
OneHotEncoder
```

usually `X` se hi required information learn karte hain.

---

# 17. StandardScaler complete example

```python
import numpy as np

from sklearn.preprocessing import StandardScaler

X_train = np.array([
    [20, 20000],
    [30, 40000],
    [40, 60000],
    [50, 80000]
])

X_test = np.array([
    [25, 30000],
    [45, 70000]
])
```

Create:

```python
scaler = StandardScaler()
```

Training:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

Test:

```python
X_test_scaled = scaler.transform(X_test)
```

Check what scaler learned:

```python
print(scaler.mean_)
```

Output approximately:

```text
[   35. 50000.]
```

Meaning:

```text
Age mean = 35
Salary mean = 50000
```

Test values bhi isi mean/scale ke according transform hongi.

---

# 18. Complete ML workflow

Ab preprocessing + model ko combine karke dekho.

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

Preprocessing:

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

Full mental model:

```text
Raw Data
   ↓
Split
   ↓
X_train
   ↓
fit_transform()
   ↓
learn preprocessing parameters
+
transform train
   ↓
X_train_scaled
   ↓
model.fit()
   ↓
trained model


X_test
   ↓
transform()
   ↓
X_test_scaled
   ↓
model.predict()
   ↓
prediction
```

---

# 19. `fit_transform()` aur `predict()` ka difference

Confuse mat karna.

```python
scaler.fit_transform(X_train)
```

Output:

```text
transformed features
```

While:

```python
model.fit(X_train, y_train)
model.predict(X_test)
```

Output:

```text
target predictions
```

So:

```text
fit_transform()
→ preprocessing/transformation

predict()
→ prediction
```

---

# 20. Three methods together

Ab tak humare paas:

```text
fit()
transform()
fit_transform()
```

Inko compare karo:

| Method            | Learn? | Transform? |
| ----------------- | -----: | ---------: |
| `fit()`           |      ✅ |          ❌ |
| `transform()`     |      ❌ |          ✅ |
| `fit_transform()` |      ✅ |          ✅ |

Ye table yaad rakhna.

---

# 21. Training vs test table

Ye aur important table hai:

| Dataset             |    `fit()` | `transform()` | `fit_transform()` |
| ------------------- | ---------: | ------------: | ----------------: |
| Training            |          ✅ |             ✅ |                 ✅ |
| Validation          | ❌ normally |             ✅ |                 ❌ |
| Test                |          ❌ |             ✅ |                 ❌ |
| New production data |          ❌ |             ✅ |                 ❌ |

Normally training me:

```python
fit_transform()
```

shortcut hota hai.

New/unseen data:

```python
transform()
```

---

# 22. `fit_transform()` before train-test split?

Galat workflow:

```python
X_scaled = scaler.fit_transform(X)

X_train, X_test = train_test_split(X_scaled)
```

Problem:

Scaler ne poore dataset ko fit ke time dekh liya:

```text
training data
+
future test data
```

This can cause:

```text
Data Leakage
```

Correct:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y
)

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Order important hai.

---

# 23. Common Mistake #1

Wrong:

```python
X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
```

Correct:

```python
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```

Reason:

```text
Fit only training data.
```

---

# 24. Common Mistake #2

Wrong:

```python
X = scaler.fit_transform(X)

train_test_split(X, y)
```

Correct:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```

Reason:

```text
Split first.
Fit later.
```

---

# 25. Common Mistake #3 — Different transformers

Wrong:

```python
train_scaler = StandardScaler()
test_scaler = StandardScaler()

X_train = train_scaler.fit_transform(X_train)

X_test = test_scaler.fit_transform(X_test)
```

Do separate scalers ki need nahi.

Correct:

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```

Same transformer should preserve same transformation logic.

---

# 26. Common Mistake #4 — Returned value save na karna

Suppose:

```python
scaler.fit_transform(X_train)
```

tumne call to kiya but result store nahi kiya.

`X_train` usually automatically transformed version se replace nahi hota.

Better:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

or:

```python
X_train = scaler.fit_transform(X_train)
```

depending on your workflow.

---

# 27. `fit_transform()` Pandas DataFrame ke saath

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "Age": [20, 30, 40, 50],
    "Salary": [20000, 40000, 60000, 80000]
})
```

Then:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled = scaler.fit_transform(df)
```

By default many sklearn transformers NumPy-like output return karte hain.

Print:

```python
print(scaled)
```

Agar DataFrame chahiye:

```python
scaled_df = pd.DataFrame(
    scaled,
    columns=df.columns
)
```

Later hum sklearn ke output configuration options bhi dekhenge.

---

# 28. `fit_transform()` OneHotEncoder ke saath sparse output

Example:

```python
encoder = OneHotEncoder()

X_encoded = encoder.fit_transform(X_train)
```

OneHotEncoder sparse matrix return kar sakta hai.

Agar dense output chahiye:

```python
encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)
```

Then:

```python
X_encoded = encoder.fit_transform(X_train)
```

Dense array milega.

Encoding chapter me isko depth me padhenge.

---

# 29. PCA ke saath `fit_transform()`

Suppose:

```text
10 features
```

PCA:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
```

Training:

```python
X_train_pca = pca.fit_transform(X_train)
```

Here:

```text
fit
→ principal components learn

transform
→ 10 dimensions ko 2 components me convert
```

Test:

```python
X_test_pca = pca.transform(X_test)
```

Same learned components use honge.

---

# 30. Pipeline aane ke baad kya hoga?

Abhi manually:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
```

Later Pipeline:

```python
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
```

Pipeline internally preprocessing ko sahi tarike se fit/transform karegi.

Isi liye sklearn Pipeline bahut powerful hai.

---

# 31. `fit_transform()` ko ek real-life analogy

Suppose tumhare paas ek teacher hai.

Training data:

```text
Exam papers
```

### `fit()`

Teacher papers dekhkar grading rules samajhta hai.

```text
fit()
= rule learn
```

### `transform()`

Teacher learned rule use karke paper ko standardized score me convert karta hai.

```text
transform()
= learned rule apply
```

### `fit_transform()`

Teacher:

```text
same papers se rule seekhta hai
+
same papers par immediately apply karta hai
```

New student's paper aaye to:

```text
transform only
```

because grading rule already learned hai.

---

# 32. Important question: Kya test data transform karna cheating hai?

No.

Ye bahut important conceptual point hai.

Test data ko:

```python
scaler.transform(X_test)
```

karna perfectly valid hai.

Why?

Because `transform()`:

```text
test data se new information learn nahi karta
```

Wo sirf training se learned rule apply karta hai.

Cheating/data leakage tab hoti hai jab:

```python
scaler.fit(X_test)
```

ya:

```python
scaler.fit_transform(X_test)
```

kiya jaye during evaluation workflow.

---

# 33. Validation data bhi same rule follow karta hai

Suppose:

```text
Train
Validation
Test
```

Correct:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

Shortcut:

```python
X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)
```

Only training data gets fitted.

---

# 34. New production data bhi test jaise treat hota hai

Model deploy hone ke baad:

```python
new_customer = [[28, 45000]]
```

Use:

```python
new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)
```

Not:

```python
scaler.fit_transform(new_customer)
```

Production rule:

```text
Already fitted preprocessing
+
already trained model
```

use karo.

---

# 35. `fit_transform()` aur model training me ek important distinction

Transformer:

```python
scaler.fit_transform(X_train)
```

Model:

```python
model.fit(X_train, y_train)
```

Model ke liye generally:

```python
model.fit_transform(...)
```

nahi hota, unless wo estimator transformer behavior bhi support karta ho.

For example:

```text
StandardScaler
→ Transformer

LinearRegression
→ Predictor
```

So:

```python
StandardScaler.fit_transform()
```

valid.

But:

```python
LinearRegression.fit_transform()
```

normally invalid.

---

# 36. Ek estimator transformer aur predictor dono ho sakta hai?

Advanced sklearn me kuch estimators multiple interfaces support kar sakte hain.

Isliye rigid rule:

```text
Every estimator must be only one type
```

sahi nahi hai.

Better understanding:

> Estimator ek broad category hai. Uske available methods decide karte hain ki wo transformation, prediction, ya other operation support karta hai.

But standard cases:

```text
StandardScaler
→ fit + transform

LinearRegression
→ fit + predict
```

---

# 37. Interview question

### Q: `fit()` aur `fit_transform()` me difference?

Answer:

> `fit()` sirf estimator ke parameters learn karta hai, jabki `fit_transform()` parameters learn karne ke baad same input data ko transform bhi karta hai.

---

### Q: Training data par `fit_transform()` aur test data par `transform()` kyon?

Answer:

> Because preprocessing parameters should be learned only from training data. `fit_transform()` training data se parameters learn aur apply karta hai, while `transform()` test data par wahi learned parameters apply karta hai without learning from the test set.

---

# 38. Ek mini example manually

Training:

```text
10
20
30
```

Suppose mean:

```text
20
```

`fit_transform()`:

```text
Learn mean = 20
+
transform:
10 → based on 20
20 → based on 20
30 → based on 20
```

Test:

```text
40
50
```

`transform()`:

```text
40 → based on training mean 20
50 → based on training mean 20
```

Test ka own mean:

```text
45
```

ignore hoga.

Exactly ye hi desired behavior hai.

---

# 39. Most important mental pattern

Machine Learning preprocessing me ye pattern baar-baar dikhega:

```python
transformer = SomeTransformer()

X_train_new = transformer.fit_transform(X_train)

X_test_new = transformer.transform(X_test)
```

Chahe transformer ho:

```text
StandardScaler
MinMaxScaler
SimpleImputer
OneHotEncoder
PCA
PolynomialFeatures
```

general workflow often same rahega.

---

# 40. Chapter 4 Final Summary

Three core methods:

```text
fit()
→ LEARN

transform()
→ APPLY

fit_transform()
→ LEARN + APPLY
```

Training:

```python
X_train_new = transformer.fit_transform(X_train)
```

Testing:

```python
X_test_new = transformer.transform(X_test)
```

Remember:

```text
TRAIN
→ fit / fit_transform

VALIDATION
→ transform

TEST
→ transform

NEW DATA
→ transform
```

Never casually do:

```python
transformer.fit_transform(X_test)
```

because:

```text
test data se learning
→ data leakage / inconsistent preprocessing
```

## Quick practice

Suppose:

```python
imputer = SimpleImputer(strategy="median")

X_train_clean = imputer.fit_transform(X_train)

X_test_clean = imputer.transform(X_test)
```

Question: `X_test` ki missing values replace karne ke liye **training median** use hoga ya **test median**?

Aur:

```python
scaler.fit_transform(X_train)
scaler.transform(X_test)
```

second line me scaler new mean/std calculate karega ya previously learned mean/std use karega?


