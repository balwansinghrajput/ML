# Chapter 11 — Feature Scaling Deep Dive  
## `StandardScaler` / Standardization

Ab hum **Feature Scaling** start kar rahe hain. Ye topic bahut important hai, especially jab features ki ranges bahut different hon.

Example:

```text
Age        Salary
25         25000
40         80000
30         50000
```

Yahan:

```text
Age
→ roughly 20–60

Salary
→ roughly 20,000–2,00,000
```

Salary ke numbers Age se bahut bade hain.

Kuch algorithms ke liye ye problem create kar sakta hai.

---

# 1. Feature Scaling kya hoti hai?

Simple definition:

> **Feature scaling ka matlab numerical features ko comparable scale/range me transform karna hota hai.**

Example:

Before:

```text
Age      Salary

20       20000
30       50000
40       80000
```

After scaling:

```text
Age      Salary

-1.2     -1.2
 0        0
 1.2      1.2
```

Actual values ka scale change hua, lekin relative information largely preserve hoti hai.

---

# 2. Feature Scaling ki zarurat kyon hoti hai?

Suppose model distance calculate kar raha hai.

Two people:

```text
Person A:
Age = 25
Salary = 30000

Person B:
Age = 35
Salary = 90000
```

Differences:

```text
Age difference
= 10

Salary difference
= 60000
```

Agar Euclidean distance use hui:

\[
d = \sqrt{(35-25)^2 + (90000-30000)^2}
\]

Salary term:

\[
60000^2
\]

Age term:

\[
10^2
\]

Salary completely dominate kar degi.

Model almost:

```text
Age ko ignore
Salary ko importance
```

kar sakta hai, sirf numerical scale ki wajah se.

---

# 3. Kya bada number automatically important feature hai?

No.

Suppose:

```text
Age = 40
Salary = 80000
```

Salary ka number bada hone ka matlab ye nahi:

```text
Salary Age se 2000 times important hai.
```

Units different hain:

```text
Age
→ years

Salary
→ rupees
```

Feature scaling unke numerical magnitudes ko comparable banane me help karti hai.

---

# 4. Standardization kya hoti hai?

Feature scaling ke multiple methods hain.

Sabse important:

```text
Standardization
```

Scikit-learn:

```python
from sklearn.preprocessing import StandardScaler
```

`StandardScaler` har feature ka training mean remove karta hai aur usse training standard deviation se divide karta hai. Scikit-learn ke current stable docs isi transformation ko \(z=(x-u)/s\) ke form me define karte hain. :chatgpt-content-reference{index="0"}

Formula:

\[
z=\frac{x-\mu}{\sigma}
\]

Where:

```text
x
= original value

μ
= feature mean

σ
= feature standard deviation

z
= standardized value
```

---

# 5. Formula ko manually samjho

Suppose Age:

```text
20
30
40
```

Mean:

\[
\mu = \frac{20+30+40}{3}=30
\]

Standard deviation approximately:

\[
\sigma \approx 8.16
\]

For Age = 20:

\[
z=
\frac{20-30}{8.16}
\]

\[
z\approx -1.22
\]

For Age = 30:

\[
z=
\frac{30-30}{8.16}
=0
\]

For Age = 40:

\[
z=
\frac{40-30}{8.16}
\approx1.22
\]

So:

```text
Original       Standardized

20             -1.22
30              0
40              1.22
```

---

# 6. StandardScaler ke baad mean kya hota hai?

Standardized training feature ka mean approximately:

```text
0
```

hota hai.

And standard deviation approximately:

```text
1
```

So hum often bolte hain:

```text
StandardScaler
→ Mean ≈ 0
→ Standard Deviation ≈ 1
```

Scikit-learn feature-wise centering/scaling independently karta hai. :chatgpt-content-reference{index="1"}

---

# 7. Mean exactly 0 kyon nahi dikhta kabhi?

Computer floating-point calculations ki wajah se output:

```text
2.220446e-16
```

jaise tiny numbers aa sakte hain.

Ye practically:

```text
0
```

ke equal hi consider kiye jate hain.

---

# 8. StandardScaler har column separately scale karta hai

Suppose:

```text
Age    Salary

20     20000
30     40000
40     60000
```

StandardScaler ek combined overall mean nahi nikalega.

Instead:

```text
Age:
mean = 30
std = ...

Salary:
mean = 40000
std = ...
```

Har feature ke training statistics independently compute hote hain. :chatgpt-content-reference{index="2"}

---

# 9. Basic Scikit-learn code

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Then:

```python
scaler.fit(X_train)
```

Then:

```python
X_train_scaled = scaler.transform(X_train)
```

Test:

```python
X_test_scaled = scaler.transform(X_test)
```

Shortcut:

```python
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

---

# 10. Chapters 2–4 ab yahan connect hote hain

### `fit()`

```python
scaler.fit(X_train)
```

Means:

```text
Training features ka:
mean learn karo
variance/std learn karo
```

### `transform()`

```python
scaler.transform(X_test)
```

Means:

```text
Training se learned mean/std
test data par apply karo
```

### `fit_transform()`

```python
scaler.fit_transform(X_train)
```

Means:

```text
Learn
+
Apply
```

---

# 11. StandardScaler kya learn karta hai?

After:

```python
scaler.fit(X_train)
```

important learned attributes hain:

```python
scaler.mean_
scaler.var_
scaler.scale_
```

Current stable sklearn docs me `mean_` per-feature training mean, `var_` variance aur `scale_` relative scaling factor store karta hai. :chatgpt-content-reference{index="3"}

Example:

```python
print(scaler.mean_)
print(scaler.var_)
print(scaler.scale_)
```

---

# 12. Example

```python
import numpy as np

X = np.array([
    [20, 20000],
    [30, 40000],
    [40, 60000]
])

scaler = StandardScaler()

scaler.fit(X)

print(scaler.mean_)
```

Output approximately:

```text
[   30. 40000.]
```

Meaning:

```text
Age mean = 30
Salary mean = 40000
```

---

# 13. `scale_` kya hai?

```python
scaler.scale_
```

generally har feature ke scaling factor ko store karta hai.

StandardScaler roughly:

```text
standard deviation
```

use karta hai.

Current docs ke according `scale_` generally `sqrt(var_)` se calculated hota hai; agar variance zero ho to scaling factor 1 reh sakta hai. :chatgpt-content-reference{index="4"}

---

# 14. Complete example

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.array([
    [20, 20000],
    [30, 40000],
    [40, 60000]
])

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

Approximately:

```text
[[-1.2247  -1.2247]
 [ 0.       0.    ]
 [ 1.2247   1.2247]]
```

---

# 15. Standardization data ko 0–1 me convert karti hai?

**No.**

Ye common confusion hai.

StandardScaler:

```text
not necessarily:
0 to 1
```

Output:

```text
-2.1
-0.5
0
1.4
3.2
```

kuch bhi ho sakta hai.

If you specifically want:

```text
0 to 1
```

then typically:

```text
MinMaxScaler
```

use hota hai.

Wo next chapter me detail me aayega.

---

# 16. Negative values kyon aati hain?

Formula:

\[
z=\frac{x-\mu}{\sigma}
\]

If:

```text
x < mean
```

then:

\[
x-\mu < 0
\]

So standardized value negative hogi.

Example:

```text
Mean = 50

Value = 30

30 - 50 = -20
```

Hence negative standardized score.

---

# 17. `z = 0` ka meaning

If:

```text
z = 0
```

then original value approximately:

```text
feature mean
```

ke equal thi.

Because:

\[
\frac{x-\mu}{\sigma}=0
\]

means:

\[
x=\mu
\]

---

# 18. `z = 1` ka meaning

Means approximately:

> Value mean se 1 standard deviation above hai.

```text
z = +1
```

Similarly:

```text
z = -1
```

means:

> Mean se 1 standard deviation below.

---

# 19. Scaling se ranking/change hoti hai?

Suppose:

```text
20 < 30 < 40
```

After StandardScaler:

```text
-1.22 < 0 < 1.22
```

Order preserve hota hai.

Standardization linear transformation hai.

---

# 20. Train-Test rule sabse important

Wrong:

```python
scaler.fit_transform(X)
```

then:

```python
train_test_split(...)
```

Problem:

Scaler ne:

```text
training data
+
test data
```

dono ka mean/std dekh liya.

This is **data leakage**.

Correct:

```python
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

Current docs bhi specifically training statistics compute karke later `transform` me use karne ka behavior define karti hain. :chatgpt-content-reference{index="5"}

---

# 21. Test ka mean use hota hai?

No.

Suppose:

Training:

```text
20
30
40
```

Training mean:

```text
30
```

Test:

```text
100
200
```

Test mean:

```text
150
```

But:

```python
scaler.transform(X_test)
```

uses:

```text
mean = 30
```

not:

```text
150
```

Because model ko same coordinate system/preprocessing chahiye.

---

# 22. Production data par bhi same

Suppose model deployed hai.

New customer:

```python
new_customer = [[32, 55000]]
```

Do:

```python
new_customer_scaled = scaler.transform(
    new_customer
)
```

Then:

```python
prediction = model.predict(
    new_customer_scaled
)
```

Do **not**:

```python
scaler.fit_transform(new_customer)
```

---

# 23. Scaling kin algorithms ke liye important hai?

Especially algorithms jo:

```text
distance
gradient/optimization
regularization
kernel calculations
```

par strongly depend karte hain.

Common examples:

```text
KNN
K-Means
SVM
Logistic Regression
Ridge
Lasso
ElasticNet
PCA
Neural Networks
SGD-based models
```

Scikit-learn docs specifically SVM RBF kernel aur linear models ke L1/L2 regularization ko examples ke roop me mention karti hain jahan unequal feature variances objective ko distort kar sakti hain. :chatgpt-content-reference{index="6"}

---

# 24. KNN me scaling kyon important?

KNN nearest point find karta hai.

Suppose:

```text
Age difference = 5
Salary difference = 50000
```

Distance calculation:

\[
\sqrt{5^2 + 50000^2}
\]

Clearly Salary dominate karegi.

Scaling ke baad:

```text
Age difference ≈ 0.5
Salary difference ≈ 0.7
```

Now both meaningful contribution de sakte hain.

---

# 25. K-Means me scaling kyon important?

K-Means cluster assignment distance par based hota hai.

Features:

```text
Age
Salary
```

Salary scale huge hai.

Without scaling:

```text
Clusters mostly Salary se decide
```

ho sakte hain.

With scaling:

```text
Age + Salary
```

dono ka influence more balanced ho sakta hai.

---

# 26. SVM me scaling kyon important?

Especially RBF-based SVM me feature distances/kernels important hote hain.

Agar ek feature ka variance bahut huge hai to wo objective/kernel behavior dominate kar sakta hai. Scikit-learn docs explicitly RBF SVM ko standardization-sensitive example batati hain. :chatgpt-content-reference{index="7"}

---

# 27. Logistic Regression me scaling?

Logistic Regression mathematically scaling ke bina bhi train ho sakti hai.

Lekin scaling useful ho sakti hai because:

```text
optimization
regularization
coefficient comparability
```

improve ho sakte hain.

Especially L1/L2 regularization me feature scale important hai. Scikit-learn user guide isi point ko note karti hai. :chatgpt-content-reference{index="8"}

---

# 28. Linear Regression ko scaling chahiye?

Plain ordinary least-squares LinearRegression prediction theoretically feature scaling ke bina bhi work kar sakti hai.

But scaling useful ho sakti hai when:

```text
Numerical stability
Coefficient comparison
Regularized regression
Optimization-based models
```

involved ho.

Important:

```text
Linear Regression
≠ always mandatory scaling
```

---

# 29. Ridge / Lasso me scaling especially important

Suppose:

```text
Age coefficient
Salary coefficient
```

but features vastly different units me hain.

Regularization coefficients ko penalize karti hai.

Without scaling:

```text
Penalty feature scales se unfairly influenced
```

ho sakti hai.

Isliye:

```text
Ridge
Lasso
ElasticNet
```

ke saath scaling common/important practice hai.

---

# 30. PCA me scaling kyon important?

PCA variance ko use karta hai.

Suppose:

```text
Age variance = 100

Salary variance = billions
```

Then PCA could mostly Salary variation ko principal component bana de.

Not necessarily because Salary more informative hai, but because numerical scale much larger hai.

Therefore standardized PCA common workflow hai.

---

# 31. Decision Tree ko scaling kyon usually nahi chahiye?

Decision Tree generally threshold-based decisions karta hai.

Example:

```text
Age < 30?
```

Suppose standardized Age:

```text
Age_scaled < -0.25?
```

Relative ordering same hai.

Tree essentially same type ka split find kar sakta hai.

It doesn't calculate Euclidean distances between feature magnitudes.

So usually:

```text
DecisionTree
RandomForest
ExtraTrees
GradientBoosted tree models
```

ko StandardScaler ki need nahi hoti.

---

# 32. Tree example

Original:

```text
Salary

30000
50000
70000
```

Tree split:

```text
Salary < 60000?
```

After scaling:

```text
-1
0
1
```

Equivalent threshold could become:

```text
Salary_scaled < 0.5?
```

Ordering same:

```text
30000 < 50000 < 70000
```

So tree's basic split logic unaffected in a meaningful way.

---

# 33. Scaling algorithms quick table

| Algorithm | Scaling generally? |
|---|---|
| KNN | ✅ Important |
| K-Means | ✅ Important |
| SVM | ✅ Important |
| Logistic Regression | ✅ Often recommended |
| Ridge/Lasso | ✅ Important |
| PCA | ✅ Usually important |
| SGD models | ✅ Often important |
| Neural Networks | ✅ Usually important |
| Decision Tree | ❌ Usually not needed |
| Random Forest | ❌ Usually not needed |
| Extra Trees | ❌ Usually not needed |

Remember: this is a strong practical guideline, not an absolute mathematical law for every dataset/configuration.

---

# 34. StandardScaler outliers ke saath problem

Suppose:

```text
Salary

20000
30000
40000
50000
10000000
```

Huge outlier:

```text
10000000
```

Mean ko bahut shift karega.

Standard deviation bhi very large ho jayegi.

Since StandardScaler mean/std use karta hai, it's sensitive to outliers; current sklearn docs explicitly ye warning deti hain. :chatgpt-content-reference{index="9"}

---

# 35. Outlier example

Without outlier:

```text
20
30
40
50
```

Mean:

```text
35
```

Add:

```text
1000
```

Mean suddenly:

```text
228
```

So standardization ka center strongly shift ho gaya.

Aise case me:

```text
RobustScaler
```

useful ho sakta hai.

Later chapter me karenge.

---

# 36. StandardScaler outliers remove karta hai?

No.

Very important:

```text
StandardScaler
≠ Outlier Removal
```

It only scales values.

Suppose:

```text
10000000
```

outlier tha.

After scaling it might become:

```text
2.0 or 5.0 etc.
```

but point still outlier ho sakta hai.

Scaling and outlier handling alag topics hain.

---

# 37. Standardization distribution ko normal banati hai?

No.

Another major misconception.

StandardScaler:

```text
Mean center karta hai
Std scale karta hai
```

But skewed distribution:

```text
right-skewed
```

standardization ke baad automatically normal/Gaussian nahi ban jati.

Scikit-learn guide bhi note karti hai ki practically distribution shape ko often ignore karke mean removal aur variance scaling ki jati hai. :chatgpt-content-reference{index="10"}

---

# 38. Example

Original highly skewed:

```text
1
2
3
4
1000
```

Standardize karne ke baad numbers center/scale honge.

But:

```text
1000
```

still extreme point rahega.

Distribution magically Gaussian nahi hogi.

---

# 39. `with_mean`

StandardScaler constructor:

```python
StandardScaler(
    with_mean=True,
    with_std=True
)
```

Current defaults exactly ye hain. :chatgpt-content-reference{index="11"}

`with_mean=True`:

```text
Mean subtract karo.
```

Formula part:

\[
x-\mu
\]

---

# 40. `with_mean=False`

```python
scaler = StandardScaler(
    with_mean=False
)
```

Then mean centering disable ho jata hai.

Useful especially sparse matrices ke case me.

Scikit-learn notes that sparse CSR/CSC matrices ke saath centering sparsity destroy kar sakti hai, isliye `with_mean=False` required hota hai. :chatgpt-content-reference{index="12"}

---

# 41. `with_std`

Default:

```python
with_std=True
```

means:

```text
standard deviation se scale karo
```

Current docs me this is described as scaling to unit variance/std. :chatgpt-content-reference{index="13"}

---

# 42. `with_std=False`

```python
scaler = StandardScaler(
    with_std=False
)
```

Then:

```text
mean removal
```

ho sakta hai, but standard deviation scaling nahi.

Usually default change karne ki need beginners ko nahi hoti.

---

# 43. `copy`

Default:

```python
copy=True
```

Means scaler generally input ko overwrite karne ki bajay transformed result return karta hai.

```python
X_scaled = scaler.transform(X)
```

Current docs note karti hain ki `copy=False` bhi inplace guarantee nahi karta in every input type. :chatgpt-content-reference{index="14"}

Abhi default hi rakho.

---

# 44. Missing values ke saath StandardScaler

Current stable docs ke according NaN values `fit()` ke during missing values ki tarah ignore hoti hain aur `transform()` me NaN maintained rehti hain. :chatgpt-content-reference{index="15"}

But ML workflow me often:

```text
Imputer
↓
Scaler
```

use karte hain.

Example:

```text
SimpleImputer
↓
StandardScaler
↓
Model
```

Later Pipeline me exactly ye karenge.

---

# 45. Categorical columns ko StandardScaler?

Normally raw categorical strings:

```text
City
Indore
Dewas
```

ko StandardScaler nahi diya jata.

StandardScaler numerical features ke liye hai.

Mixed dataset:

```text
Age
Salary
City
```

typically:

```text
Age, Salary
→ StandardScaler

City
→ OneHotEncoder
```

Later `ColumnTransformer` isi problem ko solve karega.

---

# 46. Ordinal encoded feature ko scale karna?

Suppose:

```text
Education:

School = 0
Bachelor = 1
Master = 2
PhD = 3
```

Kya scale karna chahiye?

Depends on model and interpretation.

If distance-based/linear model me treat kar rahe ho:

```text
may scale
```

But ordinal coding already semantic ordering represent karti hai.

Ye model-specific decision hai.

---

# 47. One-hot encoded columns ko scale karna?

Suppose:

```text
City_Indore
0/1
```

Generally:

```text
OneHot features
```

already same 0/1 scale par hain.

Often unhe StandardScaler karna necessary nahi.

Especially sparse one-hot matrices me mean centering problematic ho sakta hai.

Common workflow:

```text
Numerical
→ StandardScaler

Categorical
→ OneHotEncoder
```

---

# 48. Complete Pandas example

```python
import pandas as pd

df = pd.DataFrame({
    "Age": [20, 30, 40, 50],
    "Salary": [
        20000,
        40000,
        60000,
        80000
    ]
})
```

Create scaler:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Transform:

```python
scaled = scaler.fit_transform(df)

print(scaled)
```

Check mean:

```python
print(scaler.mean_)
```

Output:

```text
[   35. 50000.]
```

---

# 49. DataFrame output issue

Input DataFrame ho sakta hai:

```python
type(df)
```

But:

```python
scaled = scaler.fit_transform(df)
```

may give array-like output.

If needed:

```python
scaled_df = pd.DataFrame(
    scaled,
    columns=df.columns,
    index=df.index
)
```

Later Pipeline/ColumnTransformer ke saath feature names handling aur clean hogi.

---

# 50. Complete Train-Test example

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
```

Data:

```python
df = pd.DataFrame({
    "Age": [
        20, 25, 30, 35,
        40, 45, 50, 55
    ],
    "Salary": [
        20000, 30000, 40000, 50000,
        60000, 70000, 80000, 90000
    ],
    "Purchased": [
        0, 0, 0, 0,
        1, 1, 1, 1
    ]
})
```

X/y:

```python
X = df.drop(
    columns=["Purchased"]
)

y = df["Purchased"]
```

Split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

Scaler:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)
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

---

# 51. Full flow

```text
Raw Data
    ↓
Create X/y
    ↓
Train-Test Split
    ↓
┌──────────────────────┐
↓                      ↓
X_train               X_test
↓                      |
StandardScaler.fit()   |
↓                      |
learn mean/std         |
↓                      |
transform()         transform()
↓                      ↓
X_train_scaled     X_test_scaled
↓
model.fit()
                       ↓
                  model.predict()
```

---

# 52. Scaling y/target?

Suppose classification:

```text
y = 0 / 1
```

Do not StandardScale target labels.

Usually:

```text
X
→ scale

y
→ leave as target labels
```

---

# 53. Regression target scaling?

Regression me target transformation/scaling kabhi useful ho sakti hai.

Example:

```text
House Price
```

But that's a separate problem.

Scikit-learn:

```text
TransformedTargetRegressor
```

jaise tools provide karta hai.

Abhi rule:

```text
StandardScaler mainly X/features ke liye
```

rakho.

---

# 54. Scaling se model performance always improve hogi?

No.

Example:

```text
RandomForest
```

me almost no meaningful benefit mil sakta hai.

But:

```text
KNN
SVM
```

me significant difference aa sakta hai.

So:

> Scaling ka importance algorithm-dependent hai.

---

# 55. Feature units disappear ho jati hain

Before:

```text
Age = 30 years
Salary = ₹50000
```

After StandardScaler:

```text
Age = -0.42
Salary = 0.18
```

These standardized values no longer directly:

```text
years
rupees
```

represent karte.

They represent position relative to training distribution.

---

# 56. Original values wapas kaise laayein?

Use:

```python
X_original = scaler.inverse_transform(
    X_scaled
)
```

Flow:

```text
Original
↓
transform()
↓
Scaled
↓
inverse_transform()
↓
Original scale
```

Useful for debugging/interpretability.

---

# 57. Example inverse transform

```python
X_scaled = scaler.fit_transform(X)

X_original = scaler.inverse_transform(
    X_scaled
)
```

`X_original` approximately original values ke equal hoga.

---

# 58. Zero variance feature

Suppose:

```text
CountryCode

1
1
1
1
1
```

Variance:

```text
0
```

Standard deviation:

```text
0
```

Normally divide-by-zero issue hota.

Current StandardScaler docs ke according agar feature variance zero hai to us feature ko unit variance banana possible nahi; scaler scaling factor 1 use karke feature ko as-is scaling-wise chhod deta hai. :chatgpt-content-reference{index="16"}

---

# 59. StandardScaler vs normalization confusion

Terminology important:

```text
Standardization
≠ Normalization
```

Standardization:

\[
z=\frac{x-\mu}{\sigma}
\]

Normalization term kabhi MinMax scaling ke liye loosely use hota hai, aur sklearn me `Normalizer` naam ka separate transformer bhi hai jo per-sample vector norms scale karta hai.

So exact term use karo.

---

# 60. StandardScaler vs MinMaxScaler quick preview

### StandardScaler

```text
Mean ≈ 0
Std ≈ 1
```

Formula:

\[
\frac{x-\mu}{\sigma}
\]

### MinMaxScaler

Usually:

```text
0 to 1
```

Formula:

\[
\frac{x-x_{min}}{x_{max}-x_{min}}
\]

Next chapter me MinMaxScaler ko detail me compare karenge.

---

# 61. StandardScaler vs RobustScaler quick preview

StandardScaler:

```text
Mean
Std
```

so outlier sensitive.

RobustScaler:

```text
Median
IQR
```

use karta hai.

Strong outliers:

```text
RobustScaler
```

often more suitable starting option ho sakta hai.

---

# 62. Common Mistake #1 — Split se pehle scaling

Wrong:

```python
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y
)
```

Correct:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y
)

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)
```

---

# 63. Common Mistake #2 — Test par `fit_transform`

Wrong:

```python
X_test_scaled = scaler.fit_transform(
    X_test
)
```

Correct:

```python
X_test_scaled = scaler.transform(
    X_test
)
```

---

# 64. Common Mistake #3 — Separate scaler for test

Wrong:

```python
train_scaler = StandardScaler()
test_scaler = StandardScaler()

X_train = train_scaler.fit_transform(
    X_train
)

X_test = test_scaler.fit_transform(
    X_test
)
```

Correct:

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)
```

Same learned training scale.

---

# 65. Common Mistake #4 — Scale every algorithm blindly

Don't automatically:

```text
Every model
→ StandardScaler
```

Think:

```text
Distance/kernel/regularization/gradient-based?
→ scaling likely important

Tree-based?
→ usually unnecessary
```

---

# 66. Common Mistake #5 — Assume scaling removes outliers

Wrong thinking:

```text
StandardScaler lagaya
→ outlier solved
```

No.

Outlier still exists.

Need:

```text
outlier analysis
RobustScaler
transformations
domain handling
```

depending on problem.

---

# 67. Common Mistake #6 — Standardize categorical strings

Wrong:

```python
scaler.fit_transform(
    df[["City"]]
)
```

where:

```text
City = Indore/Dewas
```

Use categorical encoding instead.

---

# 68. Common Mistake #7 — Assume standardized values must be -1 to +1

No.

Values can be:

```text
-3.4
-1.2
0
1.7
5.8
```

Standardization does not impose fixed range.

---

# 69. Common Mistake #8 — Assume mean=0 means all values near zero

Not necessarily.

Dataset:

```text
-100
-5
0
5
100
```

can still have mean 0.

Mean zero only tells center.

It doesn't mean small range.

---

# 70. A practical algorithm rule

When starting:

```text
KNN
SVM
Logistic Regression
Ridge/Lasso
PCA
K-Means
SGD
```

think:

```text
Should I scale?
→ Very likely yes.
```

When:

```text
Decision Tree
Random Forest
Extra Trees
```

think:

```text
Scaling probably unnecessary.
```

---

# 71. Real-world Pipeline preview

Right now:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

model.fit(
    X_train_scaled,
    y_train
)
```

Later:

```python
pipeline.fit(
    X_train,
    y_train
)
```

Pipeline internally:

```text
StandardScaler
↓
Model
```

handle karegi.

And:

```python
pipeline.predict(X_test)
```

automatically test transformation karega.

---

# 72. Chapter 11 mental model

```text
Raw Numerical Feature

Salary
20000
40000
60000

      ↓

StandardScaler.fit()

Learn:
mean
std

      ↓

transform()

Standardized Salary

-1.22
 0
 1.22
```

Core:

```text
fit()
→ training mean/std learn

transform()
→ learned mean/std apply
```

---

# Chapter 11 Summary

Import:

```python
from sklearn.preprocessing import StandardScaler
```

Create:

```python
scaler = StandardScaler()
```

Training:

```python
X_train_scaled = scaler.fit_transform(
    X_train
)
```

Testing:

```python
X_test_scaled = scaler.transform(
    X_test
)
```

Formula:

\[
\boxed{
z=\frac{x-\mu}{\sigma}
}
\]

Result approximately:

```text
Training feature mean
→ 0

Training feature standard deviation
→ 1
```

Important attributes:

```python
scaler.mean_
scaler.var_
scaler.scale_
```

And remember:

```text
StandardScaler
→ fixed 0–1 range nahi deta

StandardScaler
→ outliers remove nahi karta

StandardScaler
→ distribution automatically normal nahi banata
```

Most important algorithm distinction:

```text
KNN / SVM / K-Means / PCA /
Logistic Regression / Ridge / Lasso
→ Scaling important/often recommended

Decision Tree / Random Forest
→ Usually scaling required nahi
```

### Quick practice

Suppose:

```text
Age range
20–60

Salary range
20,000–5,00,000
```

1. KNN me scaling na karne par kaunsa feature distance dominate kar sakta hai?
2. `StandardScaler` ka formula kya hai?
3. `z = 0` ka kya meaning hai?
4. `StandardScaler` output always `0–1` me hota hai?
5. `scaler.fit(X_train)` kya learn karta hai?
6. `X_test` par `fit_transform()` kyon nahi karna?
7. RandomForest ko usually scaling ki need kyon nahi hoti?

