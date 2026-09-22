# Chapter 3 — `transform()` Method Deep Dive

Ab hum `transform()` ko detail me samjhenge. Chapter 2 me tumne seekha tha:

```text
fit()
= data se learn karna
```

Ab:

```text
transform()
= jo learn kiya hai usko data par apply karna
```

Ye difference Scikit-learn ka core concept hai.

---

## 1. `transform()` kya hota hai?

Simple definition:

> **`transform()` ek fitted transformer ke learned parameters ko use karke input data ko new form me convert karta hai.**

Basic flow:

```text
Data
 ↓
fit()
 ↓
parameters learn
 ↓
transform()
 ↓
new/transformed data
```

Example:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
```

Yahan:

```text
fit()
→ mean/std learn kiya

transform()
→ us mean/std ko use karke values scale ki
```

---

# 2. Sabse important difference

Isko strongly yaad rakho:

```text
fit()
= LEARN

transform()
= APPLY
```

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
```

Ab scaler ne kuch information learn ki.

Then:

```python
X_train_scaled = scaler.transform(X_train)
```

Ab wo learned information apply hui.

---

# 3. StandardScaler ka example

Suppose data:

```python
X = [
    [10],
    [20],
    [30]
]
```

Hum:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)
```

karenge.

Scaler learn karega:

```text
Mean = 20
Standard deviation ≈ 8.16
```

Abhi tak data change nahi hua.

Original:

```text
10
20
30
```

ab bhi same hai.

Ab:

```python
X_scaled = scaler.transform(X)
```

StandardScaler roughly ye formula apply karega:

$$
z = \frac{x-\mu}{\sigma}
$$

For `10`:

$$
\frac{10-20}{8.16}
\approx -1.22
$$

For `20`:

$$
\frac{20-20}{8.16}
= 0
$$

For `30`:

$$
\frac{30-20}{8.16}
\approx 1.22
$$

Result:

```text
-1.22
 0.00
 1.22
```

Important:

```text
fit()
→ mean/std calculate

transform()
→ scaling formula apply
```

---

# 4. `transform()` khud se kuch learn karta hai?

Normally **nahi**.

Suppose:

```python
scaler.transform(X_test)
```

Scaler `X_test` ka new mean calculate nahi karega.

Wo wahi values use karega jo training data se:

```python
scaler.fit(X_train)
```

ke time learn hui thi.

Yahi reason hai ki test data ko safe way me transform kar sakte hain.

---

# 5. Train aur Test Data example

Suppose:

```text
Training Data

10
20
30
```

Training mean:

```text
20
```

Hum:

```python
scaler.fit(X_train)
```

karte hain.

Ab scaler ke paas:

```text
mean_ = 20
```

stored hai.

Suppose test data:

```text
40
50
```

Ab:

```python
X_test_scaled = scaler.transform(X_test)
```

Scaler test data ka mean calculate nahi karega.

Instead:

```text
Training mean = 20
```

use karega.

So roughly:

```text
40 → (40 - 20) / training_std
50 → (50 - 20) / training_std
```

Yahi correct ML workflow hai.

---

# 6. Agar test data par dobara `fit()` karein?

Wrong:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)
```

Problem:

Second:

```python
scaler.fit(X_test)
```

training wale learned mean/std ko replace kar dega.

Ab train aur test different scaling rules use kar rahe hain.

Correct:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

One fitted transformer.

Same learned parameters.

---

# 7. Real-world intuition

Suppose tum ek model train kar rahe ho customer salary par.

Training data:

```text
₹20,000
₹30,000
₹40,000
₹50,000
```

Scaler training data se kuch statistics learn karta hai.

Kal new user aaya:

```text
₹70,000
```

Tum us new user ke basis par scaler ko dobara fit nahi karoge.

Tum karoge:

```python
new_user_scaled = scaler.transform([[70000]])
```

Because production me bhi:

```text
New data
 ↓
existing preprocessing
 ↓
existing trained model
```

use hota hai.

---

# 8. `transform()` ka return kya hota hai?

`transform()` usually transformed data return karta hai.

Example:

```python
X_scaled = scaler.transform(X)
```

Yahan:

```python
X_scaled
```

new transformed data hai.

Important point:

```python
scaler.transform(X)
```

generally original `X` ko inplace change nahi karta.

Example:

```python
print(X)
```

still original values ho sakti hain.

While:

```python
print(X_scaled)
```

transformed values dikhayega.

So:

```text
X
= original data

X_scaled
= transformed copy/result
```

---

# 9. `fit()` ka return vs `transform()` ka return

Compare:

```python
result = scaler.fit(X)
```

`fit()` generally fitted estimator itself return karta hai.

Whereas:

```python
result = scaler.transform(X)
```

`transform()` transformed data return karta hai.

Simple:

| Method        | Main return      |
| ------------- | ---------------- |
| `fit()`       | fitted estimator |
| `transform()` | transformed data |

---

# 10. SimpleImputer me `transform()`

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

Create:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
```

Fit:

```python
imputer.fit(X)
```

Mean learn:

```text
(10 + 20 + 40) / 3
= 23.33
```

Then:

```python
X_new = imputer.transform(X)
```

Result:

```text
10
20
23.33
40
```

So:

```text
fit()
→ mean learn

transform()
→ NaN ko learned mean se replace
```

---

# 11. Test data ke saath Imputer

Training:

```text
Age

20
30
NaN
40
```

Suppose training median:

```text
30
```

Fit:

```python
imputer.fit(X_train)
```

New/test data:

```text
Age

NaN
60
```

Transform:

```python
X_test_clean = imputer.transform(X_test)
```

Result:

```text
30
60
```

Notice:

Test data ka own median calculate nahi kiya.

Training ka learned:

```text
30
```

use kiya.

---

# 12. OneHotEncoder me `transform()`

Suppose training data:

```text
City

Indore
Dewas
Bhopal
Indore
```

Fit:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

encoder.fit(X_train)
```

Encoder learn karega:

```text
Bhopal
Dewas
Indore
```

Ab:

```python
encoded = encoder.transform(X_train)
```

conceptually data kuch aisa ban sakta hai:

```text
City      Bhopal   Dewas   Indore

Indore       0       0       1
Dewas        0       1       0
Bhopal       1       0       0
Indore       0       0       1
```

So:

```text
fit()
→ categories learn

transform()
→ categories ko encoded columns me convert
```

---

# 13. Unseen category problem

Suppose training me:

```text
Indore
Dewas
Bhopal
```

tha.

Test data me:

```text
Ujjain
```

aa gaya.

Default configuration depending on encoder setup error de sakti hai.

Common approach:

```python
encoder = OneHotEncoder(
    handle_unknown="ignore"
)
```

Then:

```python
encoder.fit(X_train)
```

and:

```python
encoder.transform(X_test)
```

Unknown category ko gracefully handle kiya ja sakta hai.

Isliye training aur new data ka difference important hai.

---

# 14. MinMaxScaler me `transform()`

Suppose:

```text
10
20
30
```

Fit:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X)
```

Learned:

```text
min = 10
max = 30
```

MinMax formula:

$$
x' =
\frac{x-x_{min}}
{x_{max}-x_{min}}
$$

Transform:

```python
scaler.transform(X)
```

Result:

```text
10 → 0.0
20 → 0.5
30 → 1.0
```

Again:

```text
fit()
→ min/max learn

transform()
→ formula apply
```

---

# 15. Transformer ke paas `predict()` kyon nahi hota?

Because transformer ka kaam prediction karna nahi hai.

Example:

```python
scaler.predict(X)
```

Invalid.

StandardScaler ka job hai:

```text
data convert karna
```

Prediction nahi.

So:

```python
scaler.transform(X)
```

correct.

Meanwhile:

```python
model.predict(X)
```

predictive model ke liye correct.

---

# 16. Predictive model ke paas `transform()` hota hai?

Normally ordinary models me nahi.

Example:

```python
model = LinearRegression()
```

Use:

```python
model.fit(X_train, y_train)
model.predict(X_test)
```

Not normally:

```python
model.transform(X_test)
```

Because model ka purpose output predict karna hai, feature representation transform karna nahi.

---

# 17. `fit()` vs `transform()` vs `predict()`

Is table ko strong kar lo:

| Method        | Meaning                                 |
| ------------- | --------------------------------------- |
| `fit()`       | Data se learn karo                      |
| `transform()` | Learned information se data change karo |
| `predict()`   | Learned model se output predict karo    |

Flow:

```text
Transformer:

X_train
   ↓
fit()
   ↓
learn parameters
   ↓
transform()
   ↓
X_train_transformed
```

Predictor:

```text
X_train + y_train
       ↓
      fit()
       ↓
model learns
       ↓
X_test
       ↓
predict()
       ↓
prediction
```

---

# 18. `fit()` aur `transform()` ko alag kyon rakha gaya?

Bahut important question.

Agar sklearn sirf:

```python
transform(X)
```

se automatically learn + transform dono karta, to problem hoti.

Training:

```text
mean = 30
```

Test:

```text
mean = 100
```

Agar har `transform()` par new mean calculate hota, to training aur test data different transformations use karte.

Isliye Scikit-learn separate karta hai:

```text
fit()
→ ek baar learning

transform()
→ same learned rule baar-baar apply
```

Ye production aur test data consistency ke liye critical hai.

---

# 19. Example: same scaler, multiple data sets

```python
scaler.fit(X_train)
```

Once fitted, tum:

```python
X_train_scaled = scaler.transform(X_train)

X_validation_scaled = scaler.transform(X_validation)

X_test_scaled = scaler.transform(X_test)

new_customer_scaled = scaler.transform(new_customer)
```

kar sakte ho.

Har jagah same:

```text
mean
std
```

use honge.

That's exactly what we want.

---

# 20. `transform()` before `fit()` kya hoga?

Suppose:

```python
scaler = StandardScaler()

scaler.transform(X)
```

Ye generally error dega:

```text
NotFittedError
```

Reason:

Scaler ko pata hi nahi:

```text
mean kya hai?
std kya hai?
```

Correct:

```python
scaler.fit(X)

scaler.transform(X)
```

Mental sequence:

```text
Create
 ↓
Fit
 ↓
Transform
```

---

# 21. Complete example with Pandas

```python
import pandas as pd

df = pd.DataFrame({
    "Age": [20, 30, 40, 50],
    "Salary": [20000, 40000, 60000, 80000]
})

print(df)
```

Output:

```text
   Age  Salary
0   20   20000
1   30   40000
2   40   60000
3   50   80000
```

Scaler:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Fit:

```python
scaler.fit(df)
```

Check learned information:

```python
print(scaler.mean_)
```

Approximately:

```text
[3.5e+01 5.0e+04]
```

Meaning:

```text
Age mean = 35
Salary mean = 50000
```

Transform:

```python
scaled_data = scaler.transform(df)

print(scaled_data)
```

Now Age aur Salary standardized values me convert ho jayenge.

---

# 22. Multiple columns kaise transform hote hain?

Suppose:

```text
Age   Salary

20    20000
30    40000
40    60000
```

StandardScaler har feature ko separately treat karta hai.

Age ke liye:

```text
Age mean
Age std
```

Salary ke liye:

```text
Salary mean
Salary std
```

learn karega.

So:

```python
scaler.mean_
```

me har feature ke liye separate value hogi.

Example:

```text
[30, 40000]
```

---

# 23. Shape generally same rehti hai?

Ye transformer par depend karta hai.

StandardScaler:

```text
Before:
100 rows × 5 columns

After:
100 rows × 5 columns
```

Usually same shape.

But OneHotEncoder shape increase kar sakta hai.

Example:

```text
Before:

City
1 column
```

After:

```text
Bhopal
Dewas
Indore

3 columns
```

So:

> `transform()` ka matlab ye nahi ki shape always same hogi.

Transformation estimator-specific hota hai.

---

# 24. PCA example

PCA bhi transformer hai.

Suppose:

```text
10 features
```

PCA:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
```

Fit:

```python
pca.fit(X_train)
```

PCA components learn karega.

Then:

```python
X_train_pca = pca.transform(X_train)
```

Data:

```text
10 features
```

se:

```text
2 components
```

me transform ho sakta hai.

Yahan shape deliberately change hui.

---

# 25. `inverse_transform()` kya hota hai?

Kuch transformers reverse transformation support karte hain.

Example:

```python
X_scaled = scaler.transform(X)
```

Then:

```python
X_original = scaler.inverse_transform(X_scaled)
```

Concept:

```text
Original
 ↓
transform()
 ↓
Scaled
 ↓
inverse_transform()
 ↓
Approx original
```

StandardScaler example:

```text
Age = 30
 ↓
Scaled = -0.5
 ↓
inverse_transform
 ↓
Age ≈ 30
```

Ye useful hota hai jab transformed predictions/data ko original scale me lana ho.

---

# 26. `inverse_transform()` har transformer me nahi hota

Important:

Har transformer reverse operation support nahi karta.

For example, information-losing transformations ko exactly reverse karna possible nahi hota.

So blindly assume mat karo:

```python
transformer.inverse_transform(...)
```

hamesha available hoga.

---

# 27. Train-Test complete example

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
```

Data:

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

Learn only training:

```python
scaler.fit(X_train)
```

Transform train:

```python
X_train_scaled = scaler.transform(X_train)
```

Transform test:

```python
X_test_scaled = scaler.transform(X_test)
```

Train model:

```python
model = LogisticRegression()

model.fit(X_train_scaled, y_train)
```

Predict:

```python
prediction = model.predict(X_test_scaled)
```

Full flow:

```text
Raw Data
   ↓
train_test_split()
   ↓
X_train                  X_test
   ↓                       |
scaler.fit()               |
   ↓                       |
learn mean/std             |
   ↓                       |
transform()             transform()
   ↓                       ↓
X_train_scaled        X_test_scaled
   ↓                       |
model.fit()                |
   ↓                       |
learn model                |
                           ↓
                      model.predict()
                           ↓
                       prediction
```

---

# 28. One very common mistake

Wrong:

```python
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)
```

First line correct hai.

Second line wrong.

Why?

Because:

```python
fit_transform(X_test)
```

means:

```text
Test data se learn bhi karo
+
Test data transform bhi karo
```

Hume test data se learn nahi karna.

Correct:

```python
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Is rule ko yaad kar lo.

---

# 29. `fit_transform()` ka quick preview

Ye next chapter ka main topic hai.

These:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
```

can usually be shortened to:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

So:

```text
fit_transform()
=
fit()
+
transform()
```

Conceptually.

But again:

```python
X_test_scaled = scaler.transform(X_test)
```

Test pe `fit_transform()` nahi.

---

# 30. Data leakage ka connection

Suppose:

```python
scaler.fit(X)
```

and then:

```python
train_test_split(...)
```

Problem:

`X` me train + test dono included the.

Scaler test information dekh chuka.

This is:

```text
Data Leakage
```

Better:

```python
X_train, X_test, y_train, y_test = train_test_split(...)

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 31. Transformer ka state kya hota hai?

When:

```python
scaler.fit(X_train)
```

transformer **fitted state** me aa jata hai.

Before:

```text
StandardScaler object
↓
unfitted
```

After:

```text
StandardScaler object
↓
mean_ learned
scale_ learned
↓
fitted
```

Then:

```python
transform()
```

use kar sakte ho.

---

# 32. New data ke columns same hone chahiye

Suppose training data:

```text
Age
Salary
Experience
```

Scaler ko 3 features par fit kiya:

```python
scaler.fit(X_train)
```

New data:

```text
Age
Salary
```

sirf 2 features ke saath bhej diya.

Usually problem hogi because fitted transformer 3 input features expect karta hai.

So production me same preprocessing structure maintain karna important hai.

---

# 33. Column order bhi important ho sakta hai

Training:

```text
Age | Salary
```

Agar new NumPy data:

```text
Salary | Age
```

order me bhej diya, technically shape same ho sakti hai but meaning galat ho jayega.

Example:

Training:

```text
Age = 25
Salary = 50000
```

Wrong new representation:

```text
50000, 25
```

Scaler samajh sakta hai:

```text
Age = 50000
Salary = 25
```

which is completely wrong.

Isi wajah se later hum:

```text
ColumnTransformer
Pipeline
```

padhenge.

---

# 34. Why Pipeline becomes useful

Manual preprocessing me:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)

prediction = model.predict(X_test_scaled)
```

Bahut steps hain.

Pipeline later isko organize karega:

```python
pipeline.fit(X_train, y_train)

pipeline.predict(X_test)
```

Pipeline automatically correct order me preprocessing + prediction handle karega.

Isi wajah se pipeline real projects me bahut important hai.

---

# 35. `transform()` ka mathematical meaning

Transformer often ek function learn karta hai:

$$
T(x)
$$

Fit:

```text
Determine T using training data
```

Transform:

```text
Apply T(x)
```

For StandardScaler:

$$
T(x) =
\frac{x-\mu_{train}}
{\sigma_{train}}
$$

Important:

$$
\mu_{train}
$$

and:

$$
\sigma_{train}
$$

training set se aaye.

Test data ke own:

$$
\mu_{test}
$$

use nahi hote.

---

# 36. Different transformers ka `transform()`

| Transformer          | `fit()`                            | `transform()`                    |
| -------------------- | ---------------------------------- | -------------------------------- |
| `StandardScaler`     | mean/std learn                     | standardize                      |
| `MinMaxScaler`       | min/max learn                      | scale range                      |
| `SimpleImputer`      | fill statistic learn               | NaN replace                      |
| `OneHotEncoder`      | categories learn                   | encode categories                |
| `OrdinalEncoder`     | categories/order learn             | integer codes                    |
| `PCA`                | components learn                   | lower-dimensional representation |
| `PolynomialFeatures` | output feature structure establish | polynomial features create       |

Main principle same:

```text
Learn
↓
Apply
```

---

# 37. Important subtle point: har transformer same way "learn" nahi karta

Some transformers ka `fit()` almost little/no statistical learning kar sakta hai, depending on transformer.

But sklearn API consistent rakhta hai:

```python
transformer.fit(...)
transformer.transform(...)
```

Taaki transformer pipelines ke andar easily work kare.

So `fit()` always means:

> transformer apne required fitted state ko establish kare.

---

# 38. `transform()` target `y` ko bhi transform kar sakta hai?

Usually preprocessing me hum `X` ko transform karte hain:

```python
X_scaled = scaler.transform(X)
```

But kuch use cases me target `y` bhi transform kiya ja sakta hai.

Example:

```text
Highly skewed house prices
```

Could use log transformation or specialized target transformation techniques.

Scikit-learn me later:

```text
TransformedTargetRegressor
```

jaise concepts milenge.

But abhi focus:

```text
Most preprocessing transformers
→ X par
```

---

# 39. `transform()` aur `predict()` me conceptual difference

Suppose input:

```text
Age = 30
Salary = 50000
```

Scaler:

```python
scaler.transform([[30, 50000]])
```

Output:

```text
scaled features
```

Meaning:

> input ka representation badla.

Model:

```python
model.predict(scaled_features)
```

Output:

```text
0 or 1
```

Meaning:

> target ka answer predict hua.

So:

```text
transform()
→ feature representation

predict()
→ target/output
```

---

# 40. Interview answer

Agar interviewer puche:

**What does `transform()` do in Scikit-learn?**

Good answer:

> `transform()` applies the parameters learned during `fit()` to convert input data into a new representation. For example, a `StandardScaler` uses the mean and scale learned from the training data to standardize both training and unseen data.

Hinglish:

> `transform()` fitted transformer ke learned parameters ko input data par apply karke data ko new form me convert karta hai. Ye normally khud se new parameters learn nahi karta.

---

# 41. Ek line me difference

```text
fit()
→ rule seekho

transform()
→ rule lagao
```

Example:

```text
fit()
→ Mean = 50 seekha

transform()
→ Har value par Mean = 50 use kiya
```

---

# Chapter 3 Summary

Is chapter ka final mental model:

```text
Transformer create
      ↓
fit(X_train)
      ↓
training data se parameters learn
      ↓
transform(X_train)
      ↓
train data transformed

same transformer
      ↓
transform(X_test)
      ↓
test data transformed using TRAINING parameters
```

Most important rules:

```text
fit()
= learn

transform()
= apply learned transformation

Test data
= transform only

New/production data
= transform only
```

Aur:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Ye pattern tumhe bahut baar use karna hai.

## Quick practice

Suppose:

```python
imputer = SimpleImputer(strategy="median")

imputer.fit(X_train)

X_test_new = imputer.transform(X_test)
```

Socho: `X_test` ka median use hoga ya `X_train` se learned median?

Aur:

```python
encoder.fit(X_train)
encoder.transform(X_test)
```

`transform()` kya new categories learn karega?

