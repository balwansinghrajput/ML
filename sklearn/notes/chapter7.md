# Chapter 7 — `train_test_split()` Deep Dive

Ab hum Scikit-learn ke bahut important function **`train_test_split()`** ko samjhenge.

Ab tak humne seekha:

```text
X = Features
y = Target

model.fit(X, y)
→ Model training

model.predict(X_new)
→ Prediction
```

Lekin ek important problem hai: agar hum model ko **poora dataset training ke liye de dein**, to fir ye kaise check karenge ki model naye/unseen data par actually achha perform karta hai ya sirf training data ko memorize kar raha hai?

Isi problem ko solve karne ke liye hum dataset ko **Training Set** aur **Test Set** me divide karte hain.

---

## 1. Train-Test Split kya hai?

Simple definition:

> **Train-test split dataset ko do parts me divide karta hai: ek model ko train karne ke liye aur doosra trained model ko evaluate karne ke liye.**

Mental model:

```text
Complete Dataset
       ↓
train_test_split()
       ↓
 ┌───────────────┐
 ↓               ↓
Training Data   Test Data
 ↓               ↓
model.fit()     model.predict()
                 ↓
             Evaluation
```

For example, agar hamare paas:

```text
1000 records
```

hain, hum kar sakte hain:

```text
800 → Training
200 → Testing
```

Yaani:

```text
80% Training
20% Testing
```

---

## 2. Training data ka kaam kya hai?

Training data wo data hai jise model **dekh sakta hai aur usse learn kar sakta hai**.

```python
model.fit(X_train, y_train)
```

Yahan model ko:

```text
X_train
+
y_train
```

dono milte hain.

Example:

```text
Hours   Marks

1       20
2       30
3       40
4       50
```

Model in examples se relationship learn karta hai.

---

## 3. Test data ka kaam kya hai?

Test data ko training ke time model ko nahi dikhaya jata.

Training hone ke baad:

```python
y_pred = model.predict(X_test)
```

Then:

```text
y_test
vs
y_pred
```

compare karte hain.

Example:

```text
Actual      Predicted

1           1
0           0
1           0
1           1
```

Isi se hume pata chalta hai model unseen data par kitna achha perform kar raha hai.

---

# 4. Train-Test Split ki zarurat kyon?

Suppose tumhare paas 100 students hain aur model banana hai jo Pass/Fail predict kare.

Agar:

```python
model.fit(X, y)
```

poore 100 students par train kar diya aur fir:

```python
model.predict(X)
```

same 100 students par prediction check kiya, to model ko ye data already training me mil chuka tha.

Model ka score:

```text
98%
```

aa sakta hai.

Lekin iska matlab ye nahi ki new students ke liye bhi model 98% accurate hoga.

Isliye hum kuch data model se **hide** kar dete hain.

```text
100 students

80 students
→ model ko sikhao

20 students
→ model ko pehle mat dikhao
→ training ke baad test karo
```

Ye model ki **generalization ability** check karta hai.

---

# 5. `train_test_split()` kahan hota hai?

Import:

```python
from sklearn.model_selection import train_test_split
```

Notice:

```text
sklearn
   ↓
model_selection
   ↓
train_test_split
```

---

# 6. Basic syntax

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y
)
```

Is function ko:

```text
X
y
```

dete hain aur ye 4 outputs deta hai:

```text
X_train
X_test
y_train
y_test
```

---

# 7. Ye 4 outputs kya hain?

Suppose original:

```text
X
= features

y
= target
```

Split ke baad:

| Variable  | Meaning           |
| --------- | ----------------- |
| `X_train` | Training features |
| `X_test`  | Testing features  |
| `y_train` | Training targets  |
| `y_test`  | Testing targets   |

Training:

```python
model.fit(X_train, y_train)
```

Prediction:

```python
y_pred = model.predict(X_test)
```

Evaluation:

```text
y_test
vs
y_pred
```

---

# 8. Complete basic example

```python
import pandas as pd

from sklearn.model_selection import train_test_split
```

Dataset:

```python
df = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [40, 45, 50, 55, 60, 65, 70, 80, 90, 95],
    "Pass": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})
```

Create features:

```python
X = df[
    ["Hours", "Attendance"]
]
```

Target:

```python
y = df["Pass"]
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

Now:

```text
Total = 10 records

80% train
= 8 records

20% test
= 2 records
```

---

# 9. `test_size` kya hota hai?

`test_size` decide karta hai ki total dataset ka kitna portion testing ke liye jayega.

Example:

```python
test_size=0.2
```

means:

```text
20% Test
80% Train
```

Similarly:

```python
test_size=0.3
```

means:

```text
30% Test
70% Train
```

And:

```python
test_size=0.25
```

means:

```text
25% Test
75% Train
```

---

# 10. `test_size` percentage ya number dono ho sakta hai

Percentage style:

```python
train_test_split(
    X,
    y,
    test_size=0.2
)
```

If total rows:

```text
1000
```

approximately:

```text
800 train
200 test
```

But integer bhi de sakte ho:

```python
train_test_split(
    X,
    y,
    test_size=200
)
```

Meaning:

```text
Exactly 200 samples test set me rakho.
```

---

# 11. `train_size` kya hota hai?

Test size ki tarah training portion bhi specify kar sakte hain.

```python
train_test_split(
    X,
    y,
    train_size=0.8
)
```

Means:

```text
80% Training
```

Usually hum sirf:

```python
test_size=0.2
```

dete hain aur sklearn automatically remaining training me rakhta hai.

So generally:

```python
test_size=0.2
```

enough hai.

---

# 12. Kitna test size choose karein?

Koi universal fixed rule nahi hai.

Common splits:

```text
80% Train / 20% Test

75% Train / 25% Test

70% Train / 30% Test
```

Large dataset ho:

```text
90 / 10
```

bhi practical ho sakta hai.

Main idea:

> Training ke liye enough data hona chahiye aur testing ke liye bhi enough unseen examples hone chahiye.

Later Cross Validation padhne ke baad ye decision aur clear ho jayega.

---

# 13. `random_state` kya hota hai?

Ye parameter beginners ko bahut confuse karta hai.

Example:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Dataset splitting usually random hoti hai.

Suppose rows:

```text
1
2
3
4
5
6
7
8
9
10
```

First execution test set ho sakta hai:

```text
2, 8
```

Another random split:

```text
4, 9
```

Another:

```text
1, 7
```

Agar har run me data split change hoga, model result bhi change ho sakta hai.

`random_state` random process ko reproducible banata hai.

---

# 14. Example without `random_state`

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)
```

Script dobara run karo.

Possible hai:

```text
Run 1:
Test rows = A, B

Run 2:
Test rows = C, D
```

Split change ho sakta hai.

---

# 15. With `random_state`

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Same dataset + same parameters + same random state hone par split reproducible hota hai.

```text
Run 1
→ same split

Run 2
→ same split

Run 3
→ same split
```

Ye debugging, learning aur experiments compare karne me bahut useful hai.

---

# 16. `random_state=42` hi kyon?

Important:

```text
42 koi magical ML number nahi hai.
```

Tum use kar sakte ho:

```python
random_state=1
```

ya:

```python
random_state=10
```

ya:

```python
random_state=100
```

Same number ka purpose:

```text
reproducibility
```

hai.

`42` bas commonly used convention hai.

---

# 17. `random_state` model accuracy improve karta hai?

Nahi.

Ye:

```text
accuracy badhane ke liye
```

nahi hai.

Its main purpose:

```text
same random split ko reproduce karna
```

hai.

Agar:

```python
random_state=42
```

se accuracy 90% aur:

```python
random_state=20
```

se 85% aa rahi hai, iska matlab ye nahi ki 42 "better model" hai.

Different split ki wajah se score change ho raha hai.

Isi issue ko handle karne ke liye later:

```text
Cross Validation
```

use karenge.

---

# 18. `shuffle` kya hota hai?

By default:

```python
shuffle=True
```

hota hai.

Meaning:

> Dataset ko split karne se pehle rows ko randomly shuffle karo.

Suppose data:

```text
First 50 rows → Class 0
Next 50 rows  → Class 1
```

Agar simply last 20 rows test me chale gaye:

```text
Test
→ mostly Class 1
```

bad split ho sakta hai.

Shuffle data ko mix karta hai.

---

# 19. `shuffle=False`

You can write:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)
```

Then rows randomize nahi hongi.

Example:

```text
Rows 1–80
→ training

Rows 81–100
→ testing
```

Ye normal classification/regression me generally default choice nahi hoti, but ordered/time-dependent data me random shuffling inappropriate ho sakta hai.

---

# 20. Time Series me special caution

Suppose dataset:

```text
January
February
March
...
December
```

and tum future sales predict kar rahe ho.

Randomly shuffle karne par:

```text
December training me
March test me
```

aa sakta hai.

Ye real future-prediction scenario ko represent nahi karega.

Time series me often:

```text
Past
→ training

Future
→ testing
```

rakhte hain.

Example:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)
```

ya later:

```text
TimeSeriesSplit
```

use karenge.

So:

```text
Normal independent records
→ shuffle often useful

Time-series ordered records
→ blindly shuffle mat karo
```

---

# 21. `stratify` kya hota hai?

Ye classification me bahut important parameter hai.

Suppose target:

```text
Class 0 = 90 students
Class 1 = 10 students
```

Total:

```text
100
```

Class distribution:

```text
0 → 90%
1 → 10%
```

Agar normal random split hua, test set me accidentally:

```text
0 = 20
1 = 0
```

bhi ho sakta hai, especially small datasets me.

Then test set Class 1 ko represent hi nahi karega.

---

# 22. `stratify=y`

Use:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Meaning:

> Train aur test sets me target class proportions ko approximately preserve karo.

Original:

```text
Class 0 = 90%
Class 1 = 10%
```

Then approximately:

```text
Training:
Class 0 ≈ 90%
Class 1 ≈ 10%

Testing:
Class 0 ≈ 90%
Class 1 ≈ 10%
```

---

# 23. Stratify example

Suppose:

```text
1000 samples

Class 0 = 800
Class 1 = 200
```

Ratio:

```text
80% / 20%
```

Test size:

```text
20%
```

With stratification, test set around:

```text
160 Class 0
40 Class 1
```

ho sakta hai.

And training around:

```text
640 Class 0
160 Class 1
```

So class proportions preserved.

---

# 24. Classification me recommended pattern

Often:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Especially when classes imbalanced hain.

But `stratify=y` blindly har problem me nahi lagana; ye mainly class/group distribution preserve karne ka mechanism hai.

---

# 25. Regression me `stratify=y`?

Normally continuous regression target ke saath direct:

```python
stratify=y
```

use nahi karte.

Why?

Suppose `y`:

```text
25134
27123
31560
42331
...
```

Almost every target unique ho sakta hai.

Stratification naturally classes/groups ke saath fit hoti hai.

So beginner rule:

```text
Classification
→ stratify=y often useful

Regression
→ normally stratify=None
```

---

# 26. Complete Classification example

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

Split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Train:

```python
model = LogisticRegression()

model.fit(
    X_train,
    y_train
)
```

Predict:

```python
y_pred = model.predict(X_test)
```

Evaluate:

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(accuracy)
```

Flow:

```text
X + y
  ↓
Split
  ↓
X_train + y_train
  ↓
fit()
  ↓
trained model
  ↓
X_test
  ↓
predict()
  ↓
y_pred
  ↓
compare with y_test
```

---

# 27. Regression example

Suppose house prices predict kar rahe hain.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

Train:

```python
model = LinearRegression()

model.fit(
    X_train,
    y_train
)
```

Predict:

```python
y_pred = model.predict(X_test)
```

Then later regression metrics se:

```text
Actual Price
vs
Predicted Price
```

compare karenge.

---

# 28. Shapes samjho

Suppose original:

```python
X.shape
```

is:

```text
(1000, 5)
```

And:

```python
y.shape
```

is:

```text
(1000,)
```

Using:

```python
test_size=0.2
```

approximately:

```text
X_train.shape
(800, 5)

X_test.shape
(200, 5)

y_train.shape
(800,)

y_test.shape
(200,)
```

Notice:

Feature count:

```text
5
```

same rehta hai.

Rows split hoti hain.

---

# 29. Train-test split columns ko divide nahi karta

Important misconception.

Dataset:

```text
1000 rows × 5 features
```

Split:

```text
800 rows × 5 features
200 rows × 5 features
```

Not:

```text
Train → 4 columns
Test → 1 column
```

Train-test split **samples/rows** ko divide karta hai, features ko nahi.

---

# 30. X aur y alignment maintain hota hai

Suppose original:

```text
X row:

Age=40
Salary=70000

y:
Purchased=1
```

After random split, agar ye sample training me gaya:

```text
X_train:
Age=40 Salary=70000
```

to corresponding:

```text
y_train:
1
```

hi saath jayega.

Sklearn X aur y ki matching preserve karta hai.

Ye extremely important hai.

---

# 31. Isliye X aur y separately shuffle nahi karne

Wrong approach:

```python
X = X.sample(frac=1)
y = y.sample(frac=1)
```

alag-alag shuffle karne se potentially mapping toot sakti hai.

Then:

```text
Age=40 Salary=70000
```

ka target kisi aur person ka ho sakta hai.

`train_test_split(X, y)` dono ko coordinated way me split karta hai.

---

# 32. Split preprocessing se pehle ya baad?

Usually:

```text
Raw Dataset
↓
X/y create
↓
Train-Test Split
↓
Fit preprocessing on training data
```

Correct:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Or:

```python
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

---

# 33. Scaling before split kyon problematic?

Wrong:

```python
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y
)
```

Scaler ne:

```text
training data
+
future test data
```

dono se statistics learn kar liye.

For StandardScaler:

```text
mean/std
```

test information bhi include karega.

This is:

```text
Data Leakage
```

Correct sequence:

```text
Split first
↓
Fit preprocessing on train
↓
Transform train and test
```

---

# 34. Imputation ke saath bhi same rule

Wrong:

```python
X_clean = imputer.fit_transform(X)

train_test_split(X_clean, y)
```

Correct:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y
)

X_train = imputer.fit_transform(X_train)

X_test = imputer.transform(X_test)
```

Training statistics only.

---

# 35. Encoding ke saath bhi same idea

Suppose:

```python
encoder.fit(X)
```

split se pehle.

Encoder test categories ko bhi dekh sakta hai.

Better workflow:

```text
Split
↓
encoder.fit(X_train)
↓
encoder.transform(X_train)
encoder.transform(X_test)
```

Later `Pipeline` aur `ColumnTransformer` is process ko safer bana denge.

---

# 36. Training score aur test score

Suppose:

```python
model.fit(X_train, y_train)
```

You can check:

```python
train_score = model.score(
    X_train,
    y_train
)

test_score = model.score(
    X_test,
    y_test
)
```

Example:

```text
Train Score = 99%
Test Score  = 70%
```

This may indicate model training data par bahut achha but new data par poor hai.

Possible:

```text
Overfitting
```

Later model evaluation chapters me depth me karenge.

---

# 37. Test data ko baar-baar use karna bhi problem ban sakta hai

Conceptually test set final unbiased evaluation ke liye hota hai.

Agar tum continuously:

```text
Model change
→ Test score check
→ Model change
→ Test score check
→ Model change
```

karte rahoge aur test performance ke according model tune karte rahoge, to indirectly test set ke according decisions lene lagoge.

Isliye real ML workflow me often:

```text
Training Set
Validation Set
Test Set
```

use karte hain.

Ya:

```text
Train
+
Cross Validation
+
Final Test
```

Later hum isko detail me cover karenge.

---

# 38. Train / Validation / Test

Basic idea:

```text
Training Set
→ model learn kare

Validation Set
→ model/hyperparameters choose karo

Test Set
→ final evaluation
```

Example:

```text
70% Training
15% Validation
15% Test
```

`train_test_split()` ko two times use karke bhi 3 sets bana sakte hain.

But abhi hamara main focus:

```text
Train + Test
```

hai.

---

# 39. Three-way split ka preview

First:

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

Now:

```text
70% Train
30% Temporary
```

Then temporary ko half:

```python
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)
```

Final:

```text
70% Train
15% Validation
15% Test
```

Later iska practical use karenge.

---

# 40. `shuffle` aur `stratify` ka relation

If:

```python
shuffle=False
```

then random stratified splitting ka concept applicable nahi hota in the usual way, and sklearn does not allow stratification with `shuffle=False`.

So normal classification:

```python
train_test_split(
    X,
    y,
    shuffle=True,
    stratify=y
)
```

common pattern hai.

---

# 41. Small datasets me issue

Suppose:

```text
10 samples
```

and:

```text
test_size=0.2
```

means only:

```text
2 test samples
```

2 samples par evaluation unreliable ho sakti hai.

Ek prediction change hone se score dramatically change ho jayega.

Example:

```text
2/2 correct
→ 100%

1/2 correct
→ 50%
```

Isliye small datasets me later:

```text
Cross Validation
```

bahut useful hoti hai.

---

# 42. Imbalanced classification example

Suppose:

```text
1000 transactions

990 Normal
10 Fraud
```

Normal random split me possible hai ki test set me very few fraud examples aaye.

Use:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Then roughly:

```text
Test:
198 Normal
2 Fraud
```

At least distribution preserve karne ki koshish hogi.

Lekin 10 fraud samples overall hi bahut few hain, so evaluation still challenging hai. Stratification har imbalance problem solve nahi karti.

---

# 43. `random_state` preprocessing ka parameter nahi

Don't confuse:

```python
train_test_split(
    random_state=42
)
```

random split control karta hai.

`StandardScaler` me usually:

```text
random_state
```

ki need nahi hoti because standard scaling deterministic hai.

But algorithms like:

```text
RandomForest
KMeans
DecisionTree-related randomized operations
```

may have their own `random_state`.

Different objects ka `random_state` apne random operations control karta hai.

---

# 44. Common Mistake — X aur y reverse karna

Wrong:

```python
train_test_split(
    y,
    X
)
```

Then output variable naming confusing ho jayegi.

Normal:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y
)
```

Order ko consistent rakho.

---

# 45. Common Mistake — Output order wrong likhna

Correct:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y
)
```

Not:

```python
X_train, y_train, X_test, y_test = train_test_split(
    X,
    y
)
```

Function returns:

```text
First input X:
→ X_train
→ X_test

Second input y:
→ y_train
→ y_test
```

So order:

```text
X_train
X_test
y_train
y_test
```

---

# 46. Easy trick to remember output order

Input:

```python
train_test_split(
    X,
    y
)
```

Think:

```text
X → train, test
y → train, test
```

Therefore:

```text
X_train
X_test
y_train
y_test
```

---

# 47. Multiple arrays bhi split kar sakte ho

Advanced but useful.

Suppose:

```text
X
y
sample_weights
```

Then `train_test_split()` multiple aligned arrays split kar sakta hai.

Conceptually:

```python
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X,
    y,
    weights,
    test_size=0.2
)
```

All arrays ka row correspondence preserved rahega.

Abhi primarily X/y par focus karo.

---

# 48. Real example with printing shapes

```python
from sklearn.model_selection import train_test_split
```

Suppose:

```python
print(X.shape)
print(y.shape)
```

Output:

```text
(1000, 4)
(1000,)
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

Check:

```python
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

Approximately:

```text
(800, 4)
(200, 4)
(800,)
(200,)
```

Ye habit useful hai.

---

# 49. Classification me class distribution check karo

Before:

```python
print(y.value_counts(normalize=True))
```

Then after split:

```python
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
```

If using:

```python
stratify=y
```

ratios similar dikhne chahiye.

Example:

```text
Original:
0 → 80%
1 → 20%

Train:
0 → 80%
1 → 20%

Test:
0 → 80%
1 → 20%
```

Approximately.

---

# 50. Complete correct preprocessing workflow

Ye part bahut important hai because Chapters 2–7 ko combine karta hai.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
```

Create X/y:

```python
X = df.drop(columns=["Purchased"])
y = df["Purchased"]
```

Split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
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

model.fit(
    X_train_scaled,
    y_train
)
```

Prediction:

```python
y_pred = model.predict(
    X_test_scaled
)
```

Complete flow:

```text
Dataset
   ↓
X + y
   ↓
train_test_split()
   ↓
┌───────────────────────┐
↓                       ↓
X_train                X_test
y_train                y_test
↓
scaler.fit_transform()
↓
X_train_scaled
↓
model.fit(X_train_scaled, y_train)
↓
trained model
                        ↓
                  scaler.transform()
                        ↓
                  X_test_scaled
                        ↓
                   model.predict()
                        ↓
                     y_pred
                        ↓
                 compare y_test
```

---

# 51. Ek important question: `y_train` ko scale kyon nahi kiya?

Classification example me:

```text
y = 0 / 1
```

target labels hain.

Usually features:

```text
Age
Salary
```

scale karte hain.

Target labels ko StandardScaler se scale karna required nahi.

Regression me target transformation ek separate use case hai, jise later discuss karenge.

So current workflow:

```text
X
→ preprocessing

y
→ target
```

---

# 52. `train_test_split()` model hai?

No.

Ye:

```text
Estimator
Transformer
Predictor
```

nahi hai.

Ye simply ek **utility function** hai.

That's why:

```python
train_test_split(...)
```

directly call karte hain.

Not:

```python
splitter = train_test_split()
splitter.fit(...)
```

---

# 53. Difference from previous sklearn classes

We had:

```python
scaler = StandardScaler()
scaler.fit(...)
```

Because StandardScaler:

```text
Class → object
```

But:

```python
train_test_split(...)
```

is a function.

So:

```text
StandardScaler()
→ Class instance

train_test_split()
→ Function call
```

Ye sklearn API ko understand karne ke liye useful distinction hai.

---

# 54. Parameters recap

Ek common call:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)
```

Meaning:

```text
X, y
→ split karne wala data

test_size=0.2
→ 20% testing

random_state=42
→ reproducible random split

shuffle=True
→ rows randomize karo

stratify=y
→ class distribution preserve karo
```

---

# 55. Classification ke liye practical template

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Ye common starting template hai.

Regression:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Time series:

```text
Special handling required
```

Normally blindly random splitting mat karo.

---

# 56. `random_state=None`

Default effectively random behavior use kar sakta hai.

```python
train_test_split(
    X,
    y,
    test_size=0.2
)
```

Har execution me split change ho sakta hai.

Learning, debugging aur experiments ke waqt fixed:

```python
random_state=42
```

rakhna convenient hai.

---

# 57. Kya same test set hamesha use karna chahiye?

Development ke ek experiment me fixed test set useful hai.

Lekin ek single split se model quality ke baare me complete picture nahi milti.

Example:

```text
Split A → 92%
Split B → 84%
Split C → 89%
```

Model performance data split ke according fluctuate kar sakti hai.

Isi wajah se later:

```text
K-Fold Cross Validation
```

use karenge.

Train-test split foundation hai; cross-validation uska advanced extension samajh sakte ho.

---

# 58. Training set ka goal

Training set ka purpose:

```text
Model ke parameters learn karna
```

Example Linear Regression:

```text
coef_
intercept_
```

Decision Tree:

```text
splits
```

Random Forest:

```text
trees
```

All learned from:

```text
X_train
y_train
```

---

# 59. Test set ka goal

Test set ka purpose:

```text
Unseen data par model ki performance estimate karna
```

Test set ka purpose:

```text
model ko sikhana
```

nahi hai.

Ye distinction strongly yaad rakhna.

---

# 60. Chapter 7 ka core mental model

Isko yaad kar lo:

```text
Complete Data
      ↓
X and y
      ↓
train_test_split()
      ↓
┌─────────────────────────┐
↓                         ↓
TRAIN                     TEST
X_train                   X_test
y_train                   y_test
↓                         ↓
Learn                     Don't learn
↓                         ↓
fit()                     predict()
                            ↓
                        compare
                        with y_test
```

## Chapter 7 Summary

`train_test_split()` ka basic template:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Classification me often:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Remember:

```text
X_train + y_train
→ Learning

X_test
→ Prediction

y_test
→ Prediction ko verify karne ke liye
```

Aur preprocessing order:

```text
Create X/y
↓
Split
↓
Fit preprocessing on X_train
↓
Transform X_train
↓
Transform X_test
↓
Train model
↓
Predict X_test
↓
Compare with y_test
```

Sabse important parameter meanings:

| Parameter      | Meaning                          |
| -------------- | -------------------------------- |
| `test_size`    | Test set kitna bada hoga         |
| `train_size`   | Training set kitna bada hoga     |
| `random_state` | Random split reproducible banana |
| `shuffle`      | Rows ko split se pehle mix karna |
| `stratify`     | Class proportions preserve karna |

### Quick practice

Agar:

```python
X.shape = (2000, 6)
```

aur:

```python
test_size=0.25
```

hai, to approximately:

```text
X_train.shape = ?
X_test.shape = ?
```

Aur agar target distribution:

```text
0 → 90%
1 → 10%
```

hai, to socho `stratify=y` lagane ka kya benefit hoga.


