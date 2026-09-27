# Chapter 12 — `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler` & `Normalizer`

Chapter 11 me humne padha:

```text
StandardScaler
→ mean ≈ 0
→ std ≈ 1
```

Ab hum dekhenge ki har situation me `StandardScaler` best choice nahi hota. Scikit-learn me alag scaling methods alag problems solve karte hain.

Is chapter ke 4 main tools:

```text
MinMaxScaler
RobustScaler
MaxAbsScaler
Normalizer
```

---

# 1. Sabse pehle big picture

| Tool | Kis basis par? | Typical output |
|---|---|---|
| `StandardScaler` | Mean + Std | mean≈0, std≈1 |
| `MinMaxScaler` | Min + Max | usually 0–1 |
| `RobustScaler` | Median + IQR | outlier-resistant scaling |
| `MaxAbsScaler` | Max absolute value | roughly -1 to 1 |
| `Normalizer` | Har row ka norm | each row length = 1 |

Sabse important difference:

```text
StandardScaler / MinMaxScaler /
RobustScaler / MaxAbsScaler

→ FEATURE/COLUMN-wise scaling
```

while:

```text
Normalizer

→ SAMPLE/ROW-wise scaling
```

Yahi Chapter 12 ka sabse important concept hai.

---

# 2. `MinMaxScaler`

Import:

```python
from sklearn.preprocessing import MinMaxScaler
```

Default:

```python
scaler = MinMaxScaler()
```

Normally training data ko:

```text
0 se 1
```

range me convert karta hai.

Formula:

\[
x' = \frac{x-x_{min}}
{x_{max}-x_{min}}
\]

Current sklearn me default `feature_range=(0, 1)` hai, aur har feature ko independently scale kiya jata hai. :chatgpt-content-reference{index="0"}

---

# 3. Manual MinMax example

Suppose:

```text
Age:

20
30
40
```

Minimum:

```text
20
```

Maximum:

```text
40
```

For 20:

\[
\frac{20-20}{40-20}
=0
\]

For 30:

\[
\frac{30-20}{40-20}
=0.5
\]

For 40:

\[
\frac{40-20}{40-20}
=1
\]

Result:

```text
20 → 0
30 → 0.5
40 → 1
```

---

# 4. Basic code

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Same rule:

```text
TRAIN
→ fit_transform()

TEST
→ transform()
```

---

# 5. `fit()` kya learn karta hai?

MinMaxScaler training data se mainly:

```text
minimum
maximum
range
```

learn karta hai.

You can inspect:

```python
print(scaler.data_min_)
print(scaler.data_max_)
print(scaler.data_range_)
```

Current API ye fitted attributes expose karti hai. :chatgpt-content-reference{index="1"}

---

# 6. Custom range

Default:

```python
MinMaxScaler(
    feature_range=(0, 1)
)
```

But you can use:

```python
MinMaxScaler(
    feature_range=(-1, 1)
)
```

Then training values desired range me scale hongi. :chatgpt-content-reference{index="2"}

---

# 7. Important: Test value 0–1 ke bahar ja sakti hai

Suppose training:

```text
Min = 0
Max = 100
```

New/test value:

```text
150
```

Then:

\[
\frac{150-0}{100-0}
=1.5
\]

So output:

```text
1.5
```

ho sakta hai.

Common misconception:

> MinMaxScaler ke baad every future value always 0–1 hoti hai.

Wrong.

**Training data** specified range me hota hai. New data training min/max ke outside ho to transformed value bhi range ke outside ja sakti hai. Current API `clip=True` se held-out data ko desired feature range me clip kar sakti hai. :chatgpt-content-reference{index="3"}

Example:

```python
scaler = MinMaxScaler(
    clip=True
)
```

But clipping information distort bhi kar sakti hai, so blindly use mat karo.

---

# 8. MinMaxScaler ka outlier problem

Suppose:

```text
10
20
30
40
1000
```

Min:

```text
10
```

Max:

```text
1000
```

Then normal values:

```text
10
20
30
40
```

bahut narrow region me compress ho jayengi.

Example roughly:

```text
10   → 0
20   → 0.01
30   → 0.02
40   → 0.03
1000 → 1
```

So:

```text
MinMaxScaler
→ outliers ke liye sensitive
```

Scikit-learn docs bhi explicitly kehti hain ki MinMaxScaler outliers ka effect reduce nahi karta; wo unhe fixed range me linearly scale karta hai. :chatgpt-content-reference{index="4"}

---

# 9. MinMaxScaler kab useful hai?

Useful when:

```text
Features ko bounded range chahiye
Data me major outliers nahi
Distance/optimization algorithms use ho rahe hain
```

Common examples:

```text
Neural networks
KNN
K-Means
Some optimization-based models
```

But always:

```text
StandardScaler vs MinMaxScaler
```

validation ke through compare karna better hai.

---

# 10. Ab `RobustScaler`

Import:

```python
from sklearn.preprocessing import RobustScaler
```

Basic:

```python
scaler = RobustScaler()
```

RobustScaler especially useful hai jab:

```text
OUTLIERS
```

present hon.

Instead of:

```text
Mean
Standard deviation
```

it uses:

```text
Median
IQR
```

Current sklearn default IQR = 25th percentile se 75th percentile tak hai. :chatgpt-content-reference{index="5"}

---

# 11. IQR kya hota hai?

IQR:

\[
IQR=Q3-Q1
\]

Where:

```text
Q1
= 25th percentile

Q3
= 75th percentile
```

Example sorted data:

```text
10
20
30
40
50
1000
```

Outlier:

```text
1000
```

Mean ko strongly affect karega.

But median aur middle 50% data relatively stable rahenge.

Isi wajah se RobustScaler outliers ke against more robust hai.

---

# 12. RobustScaler formula intuition

Roughly:

\[
x' =
\frac{x-\text{median}}
{IQR}
\]

So:

```text
Center
→ median

Scale
→ IQR
```

Scikit-learn training set ke median aur quantile-range statistics learn karta hai aur later test/new data par same values use karta hai. :chatgpt-content-reference{index="6"}

---

# 13. Code

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)
```

Check learned values:

```python
print(scaler.center_)
print(scaler.scale_)
```

Conceptually:

```text
center_
→ training median

scale_
→ training IQR-based scale
``` :chatgpt-content-reference{index="7"}


---

# 14. StandardScaler vs RobustScaler

Suppose:

```text
Salary

25000
30000
35000
40000
5000000
```

StandardScaler:

```text
mean/std
```

huge outlier se strongly affected.

RobustScaler:

```text
median/IQR
```

par based hone ki wajah se normal salaries ka scaling zyada sensible reh sakta hai.

So beginner rule:

```text
No major outliers
→ StandardScaler is a strong default

Strong outliers
→ RobustScaler consider karo
```

---

# 15. Important: RobustScaler outlier remove nahi karta

Ye bahut important hai.

Suppose:

```text
10000000
```

outlier hai.

RobustScaler:

```text
outlier ko delete nahi karega
outlier ko cap nahi karega
```

Sirf scaling statistics ko outlier se less sensitive banata hai.

So:

```text
RobustScaler
≠ Outlier Removal
```

---

# 16. `quantile_range`

Default:

```python
RobustScaler(
    quantile_range=(25.0, 75.0)
)
```

Meaning:

```text
Q1 to Q3
```

But customize:

```python
RobustScaler(
    quantile_range=(10, 90)
)
```

possible hai. Current API `quantile_range` configurable rakhti hai. :chatgpt-content-reference{index="8"}

Abhi default IQR ko hi strong understanding ke saath use karo.

---

# 17. Ab `MaxAbsScaler`

Import:

```python
from sklearn.preprocessing import MaxAbsScaler
```

Idea:

> Har feature ko uske maximum absolute training value se divide karo.

Formula roughly:

\[
x' =
\frac{x}
{\max(|x|)}
\]

Current sklearn me har feature ka maximum absolute training value `1.0` ban jata hai. Ye data ko center nahi karta, isliye sparse data ki sparsity preserve hoti hai. :chatgpt-content-reference{index="9"}

---

# 18. Manual example

Suppose feature:

```text
-10
0
5
20
```

Absolute values:

```text
10
0
5
20
```

Maximum absolute:

```text
20
```

Divide:

```text
-10 / 20 = -0.5
0 / 20   = 0
5 / 20   = 0.25
20 / 20  = 1
```

Output:

```text
-0.5
0
0.25
1
```

---

# 19. Positive-only example

Input:

```text
10
20
40
```

Maximum absolute:

```text
40
```

Output:

```text
0.25
0.50
1.00
```

So positive-only data me behavior MinMaxScaler se somewhat similar lag sakta hai.

But difference:

```text
MinMaxScaler
→ min and max use

MaxAbsScaler
→ only maximum absolute magnitude
```

---

# 20. Negative numbers ke saath advantage

Suppose:

```text
-100
-50
0
50
100
```

MaxAbs:

```text
100
```

Output:

```text
-1
-0.5
0
0.5
1
```

Data center shift nahi hota.

Zero stays:

```text
0 → 0
```

---

# 21. Sparse data me MaxAbsScaler kyon useful hai?

Sparse matrix:

```text
0
0
5
0
0
10
```

me zeros bahut important hain.

Agar centering karoge:

```text
x - mean
```

zeros non-zero ban sakte hain.

Then:

```text
sparsity destroy
```

ho sakti hai.

MaxAbsScaler:

```text
center nahi karta
```

so zeros remain zero.

That's why sparse CSR/CSC matrices ke saath useful hai. :chatgpt-content-reference{index="10"}

Common use cases:

```text
Sparse text features
large sparse matrices
```

---

# 22. MaxAbsScaler outlier-resistant hai?

No.

Suppose:

```text
1
2
3
100000
```

MaxAbs:

```text
100000
```

Then:

```text
1 → 0.00001
2 → 0.00002
3 → 0.00003
100000 → 1
```

Normal values compress ho gayi.

Current docs explicitly kehti hain ki MaxAbsScaler outlier effect reduce nahi karta. :chatgpt-content-reference{index="11"}

---

# 23. MaxAbs fitted attributes

After:

```python
scaler.fit(X_train)
```

you can inspect:

```python
scaler.max_abs_
scaler.scale_
```

`max_abs_` per-feature maximum absolute value store karta hai. :chatgpt-content-reference{index="12"}

---

# 24. Ab sabse different tool — `Normalizer`

Import:

```python
from sklearn.preprocessing import Normalizer
```

Default:

```python
normalizer = Normalizer(
    norm="l2"
)
```

Important:

> **Normalizer columns ko scale nahi karta. Har row/sample ko independently normalize karta hai.**

Current sklearn `Normalizer` each non-zero row ko unit norm me rescale karta hai. :chatgpt-content-reference{index="13"}

---

# 25. StandardScaler vs Normalizer

Suppose:

```text
        Feature1   Feature2

Row A      3          4
Row B      6          8
```

StandardScaler:

```text
feature-wise/column-wise
```

operate karega.

Normalizer:

```text
row-wise/sample-wise
```

operate karega.

This difference **bahut important** hai.

---

# 26. L2 normalization

Default:

```python
Normalizer(
    norm="l2"
)
```

For a row:

```text
[3, 4]
```

L2 norm:

\[
\sqrt{3^2+4^2}
=
5
\]

Divide each value by 5:

```text
3/5 = 0.6
4/5 = 0.8
```

Output:

```text
[0.6, 0.8]
```

Now row length:

\[
\sqrt{0.6^2+0.8^2}=1
\]

That's:

```text
unit norm
```

---

# 27. Another example

Input:

```python
X = [
    [3, 4],
    [5, 12]
]
```

Normalizer:

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer()

X_new = normalizer.fit_transform(X)
```

Results:

```text
[3, 4]
→ [0.6, 0.8]

[5, 12]
→ [0.3846, 0.9231]
```

Har row separately normalize hui.

---

# 28. `Normalizer` stateless hai

Ye interesting point hai.

Unlike StandardScaler:

```text
mean/std learn karta hai
```

or MinMaxScaler:

```text
min/max learn karta hai
```

Normalizer ko dataset-wide statistics learn karne ki zarurat nahi hoti.

Each row independently normalize hoti hai.

Current sklearn docs ise **stateless** kehti hain; `fit()` mostly API consistency/parameter validation ke liye hai. :chatgpt-content-reference{index="14"}

So:

```python
normalizer.fit(X)
```

koi training mean/max type information learn nahi karta.

---

# 29. `l1`, `l2`, `max`

Normalizer supports:

```python
Normalizer(norm="l1")
Normalizer(norm="l2")
Normalizer(norm="max")
```

Current default:

```text
l2
```

hai. :chatgpt-content-reference{index="15"}

---

# 30. L1 normalization

For:

```text
[3, 4]
```

L1 norm:

\[
|3|+|4|=7
\]

Output:

```text
3/7
4/7
```

approximately:

```text
[0.4286, 0.5714]
```

Absolute values ka sum becomes:

```text
1
```

---

# 31. `norm="max"`

Example:

```text
[2, 4, 8]
```

Maximum absolute value:

```text
8
```

Normalize row:

```text
[0.25, 0.5, 1]
```

Difference from MaxAbsScaler:

```text
MaxAbsScaler
→ each COLUMN ke training max absolute value

Normalizer(norm="max")
→ each ROW ke max absolute value
```

Extremely important distinction.

---

# 32. Normalizer ka main use case

A very common use case:

```text
Text data
TF-IDF vectors
document similarity
cosine similarity
```

For example document:

```text
word_A = 100
word_B = 20
word_C = 5
```

and another document different total length ka ho sakta hai.

Sometimes magnitude se zyada:

```text
direction / relative proportions
```

matter karte hain.

Scikit-learn docs specifically text classification/clustering aur cosine-similarity style workflows ko common uses ke roop me mention karti hain. :chatgpt-content-reference{index="16"}

---

# 33. Normalizer normal tabular Age/Salary data par?

Usually:

```text
Age
Salary
Experience
```

jaise ordinary tabular features par `Normalizer` automatically correct choice nahi hai.

Why?

Because it each person's row ko unit length bana dega.

Example:

```text
Age = 30
Salary = 50000
```

dono ko ek row vector ke parts ke roop me normalize karega.

Ye original feature meanings ko unusual way me mix kar sakta hai.

Ordinary tabular ML ke liye usually consider:

```text
StandardScaler
MinMaxScaler
RobustScaler
```

before Normalizer.

---

# 34. Most important comparison example

Suppose:

```text
X =

Age   Salary
20    20000
40    80000
```

### StandardScaler

```text
Column-wise:

Age column
→ scale

Salary column
→ scale
```

### MinMaxScaler

```text
Age:
20 → 0
40 → 1

Salary:
20000 → 0
80000 → 1
```

### RobustScaler

```text
Each column:
median + IQR based
```

### MaxAbsScaler

```text
Age:
divide by 40

Salary:
divide by 80000
```

### Normalizer

```text
Row 1:
[20, 20000]
→ row vector normalized

Row 2:
[40, 80000]
→ row vector normalized
```

So Normalizer is fundamentally different.

---

# 35. Scaler selection cheat sheet

Use this as a starting point:

```text
Data roughly normal / no strong outliers
→ StandardScaler
```

```text
Need bounded feature range
→ MinMaxScaler
```

```text
Strong outliers
→ RobustScaler
```

```text
Sparse data + preserve zeros
→ MaxAbsScaler
```

```text
Each sample's vector magnitude should become 1
→ Normalizer
```

---

# 36. Outlier comparison

| Tool | Outlier sensitive? |
|---|---:|
| StandardScaler | ✅ |
| MinMaxScaler | ✅ |
| MaxAbsScaler | ✅ |
| RobustScaler | Much less |
| Normalizer | Different concept; row magnitude based |

RobustScaler uses median/percentiles, which is why a few extreme values influence its scaling statistics less. :chatgpt-content-reference{index="17"}

---

# 37. Output range comparison

| Scaler | Fixed range? |
|---|---|
| StandardScaler | ❌ |
| MinMaxScaler | Training data: configured range |
| RobustScaler | ❌ |
| MaxAbsScaler | Training max abs becomes 1 |
| Normalizer | Row norm becomes 1 |

Don't think:

```text
Scaling = always 0 to 1
```

That's only one specific approach.

---

# 38. Train-test rule

For feature scalers that learn dataset statistics:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)
```

or:

```python
X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)
```

This applies to:

```text
StandardScaler
MinMaxScaler
RobustScaler
MaxAbsScaler
```

For `Normalizer`, because it is stateless, leakage concern from learned scaling statistics is different; still, keeping it inside the same sklearn pipeline workflow is clean and consistent. :chatgpt-content-reference{index="18"}

---

# 39. Complete comparison code

```python
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer
)

standard = StandardScaler()
minmax = MinMaxScaler()
robust = RobustScaler()
maxabs = MaxAbsScaler()
normalizer = Normalizer()
```

Then, for ordinary learned scalers:

```python
X_train_standard = standard.fit_transform(
    X_train
)

X_test_standard = standard.transform(
    X_test
)
```

Same pattern for:

```text
MinMaxScaler
RobustScaler
MaxAbsScaler
```

---

# 40. Common mistake #1 — Calling MinMax scaling “normalization”

You'll often hear:

```text
MinMax normalization
```

informally.

But in sklearn specifically:

```python
Normalizer
```

has a different precise meaning:

```text
Normalize EACH SAMPLE to unit norm.
```

So don't confuse:

```text
MinMaxScaler
≠ Normalizer
```

---

# 41. Common mistake #2 — RobustScaler removes outliers

Wrong:

```text
RobustScaler
→ outlier delete
```

Correct:

```text
RobustScaler
→ scaling statistics less affected by outliers
```

Outliers still remain.

---

# 42. Common mistake #3 — MinMax always 0–1

Training:

```text
0–1
```

yes by default.

New test value outside training range:

```text
can become <0 or >1
```

unless clipping is deliberately enabled. :chatgpt-content-reference{index="19"}

---

# 43. Common mistake #4 — Normalizer as replacement for StandardScaler

They solve different problems.

```text
StandardScaler
→ columns/features

Normalizer
→ rows/samples
```

So don't simply replace:

```python
StandardScaler()
```

with:

```python
Normalizer()
```

and assume same purpose.

---

# 44. Common mistake #5 — Scale before split

Wrong:

```python
X_scaled = scaler.fit_transform(X)

train_test_split(X_scaled, y)
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

This prevents test-set statistics from influencing training preprocessing.

---

# 45. Chapter 12 mental model

```text
StandardScaler
→ Mean + Std
```

```text
MinMaxScaler
→ Min + Max
→ bounded training range
```

```text
RobustScaler
→ Median + IQR
→ strong outliers ke liye
```

```text
MaxAbsScaler
→ Max absolute value
→ no centering
→ sparse-friendly
```

```text
Normalizer
→ each ROW independently
→ unit vector
```

---

# Chapter 12 Final Summary

### `MinMaxScaler`

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Formula:

\[
x'=\frac{x-x_{min}}{x_{max}-x_{min}}
\]

Good when:

```text
bounded range useful
few serious outliers
```

---

### `RobustScaler`

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
```

Uses:

```text
Median
+
IQR
```

Good when:

```text
outliers present
```

---

### `MaxAbsScaler`

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
```

Formula:

\[
x'=\frac{x}{max(|x|)}
\]

Useful:

```text
Sparse data
Preserve zero
No centering
```

---

### `Normalizer`

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(
    norm="l2"
)
```

Works:

```text
ROW BY ROW
```

not column-by-column.

Common use:

```text
Text vectors
TF-IDF
Cosine similarity
```

---

## Final decision table

| Situation | Try first |
|---|---|
| General scaling | `StandardScaler` |
| Need 0–1-like bounded training range | `MinMaxScaler` |
| Strong outliers | `RobustScaler` |
| Sparse features | `MaxAbsScaler` |
| Unit-length samples / text vectors | `Normalizer` |

### Quick practice

Suppose:

```text
Salary:

25000
30000
35000
40000
5000000
```

1. `StandardScaler` aur `MinMaxScaler` me outlier ka kya effect hoga?
2. Is case me `RobustScaler` kyon useful ho sakta hai?
3. `MaxAbsScaler` ka maximum absolute training value kya banata hai?
4. `Normalizer` feature-wise kaam karta hai ya row-wise?
5. `[3,4]` ko L2 Normalizer se transform karoge to kya output hoga?
6. MinMaxScaler training range `0–100` ho aur new value `150` ho, to result necessarily `1` hi hoga?
7. `MaxAbsScaler` sparse data ke liye useful kyon hai?
