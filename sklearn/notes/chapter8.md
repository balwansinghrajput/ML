# Chapter 8 — Missing Values + `SimpleImputer` Deep Dive

Ab hum Machine Learning preprocessing ke ek bahut important topic par hain:

```text
Missing Values
+
SimpleImputer
```

Real-world datasets me missing values bahut common hote hain. Is chapter me hum samjhenge:

```text
Missing value kya hai?
↓
Problem kyon hai?
↓
Kaise detect karein?
↓
Delete vs Fill kab karein?
↓
SimpleImputer
↓
mean / median / most_frequent / constant
↓
fit(), transform(), fit_transform()
↓
Train-test handling
↓
statistics_
↓
missing_values
↓
add_indicator
↓
Common mistakes
```

Current stable Scikit-learn docs me `SimpleImputer` simple column-wise strategies se missing values fill karta hai; current API `mean`, `median`, `most_frequent`, `constant`, aur custom callable strategy support karti hai. ([scikit-learn][1])

---

## 1. Missing Value kya hoti hai?

Suppose dataset:

| Age | Salary | City   |
| --: | -----: | ------ |
|  22 |  25000 | Indore |
|  30 |    NaN | Dewas  |
| NaN |  50000 | Bhopal |
|  40 |  70000 | NaN    |

Yahan:

```text
NaN
```

missing value represent kar raha hai.

Meaning:

```text
Salary unknown
Age unknown
City unknown
```

Python/Pandas me missing values commonly:

```python
np.nan
```

ya:

```python
pd.NA
```

ke form me mil sakti hain.

---

# 2. Missing Values aati kyon hain?

Real-world me bahut reasons ho sakte hain:

```text
User ne form field nahi bhara
Sensor data record nahi hua
Database value missing hai
Data collection error hua
Survey question skip hua
Different datasets merge karne par value unavailable thi
```

Example:

```text
Name: Rahul
Age: ?
Salary: 45000
```

`Age` missing hai.

---

# 3. Missing Values model ke liye problem kyon hain?

Kai ML algorithms directly missing `NaN` values ko handle nahi kar pate.

Suppose:

```python
X = [
    [25, 30000],
    [30, np.nan],
    [40, 60000]
]
```

Algorithm ko mathematical operations perform karni hain:

```text
distance
average
matrix calculations
coefficients
```

`NaN` ke saath calculation problematic ho sakti hai.

Example mathematically:

```text
50000 + NaN
= NaN
```

Isliye preprocessing me missing values ko handle karna padta hai.

Note: kuch sklearn estimators missing values natively handle kar sakte hain, isliye "har sklearn model NaN reject karta hai" kehna correct nahi hoga. Lekin imputation ab bhi bahut common workflow hai.

---

# 4. Pandas me missing values kaise check karein?

Suppose:

```python
import pandas as pd

df = pd.DataFrame({
    "Age": [20, 30, None, 40],
    "Salary": [25000, None, 50000, 70000],
    "City": ["Indore", "Dewas", None, "Bhopal"]
})
```

Check:

```python
df.isnull()
```

Ya:

```python
df.isna()
```

Dono commonly same purpose ke liye use hote hain.

---

# 5. Har column me kitni missing values hain?

Very useful:

```python
df.isnull().sum()
```

Output:

```text
Age       1
Salary    1
City      1
dtype: int64
```

Meaning:

```text
Age → 1 missing
Salary → 1 missing
City → 1 missing
```

---

# 6. Total missing values

```python
df.isnull().sum().sum()
```

Could return:

```text
3
```

---

# 7. Missing percentage

Real datasets me useful:

```python
(df.isnull().sum() / len(df)) * 100
```

Example:

```text
Age        5%
Salary    20%
City       2%
```

Isse decide karne me help milti hai ki missing data कितना serious hai.

---

# 8. Missing values handle karne ke basic options

Broadly:

```text
Missing Values
    ↓
┌───────────────┬────────────────┐
↓               ↓
Delete          Impute / Fill
```

### Delete

Rows remove:

```python
df.dropna()
```

Column remove:

```python
df.drop(columns=["column"])
```

### Impute

Missing value ko estimated/replacement value se fill karna:

```text
Mean
Median
Mode
Constant
KNN
etc.
```

---

# 9. Rows delete karna always good idea nahi

Suppose:

```text
1,000,000 rows
```

and only:

```text
100 rows missing
```

Dropping may be acceptable.

But suppose:

```text
100 rows total
40 rows missing
```

Agar sab missing rows remove kar diye:

```text
40% data lost
```

Model ko valuable information kam mil sakti hai.

Isliye blindly:

```python
df.dropna()
```

use nahi karna.

---

# 10. Imputation kya hoti hai?

> **Missing values ko kisi suitable estimated/replacement value se fill karne ke process ko imputation kehte hain.**

Example:

```text
Age

20
30
NaN
40
```

Mean:

$$
\frac{20+30+40}{3}=30
$$

After imputation:

```text
20
30
30
40
```

---

# 11. `SimpleImputer` kya hai?

Scikit-learn me:

```python
from sklearn.impute import SimpleImputer
```

`SimpleImputer` ek **Transformer** hai.

Meaning:

```text
fit()
+
transform()
```

support karta hai.

Its purpose:

> Har feature/column me missing values ko simple column-wise strategy se replace karna.

Official docs ise univariate imputer describe karti hain—yaani har feature ki imputation primarily usi feature ke observed values se calculate hoti hai. ([scikit-learn][1])

---

# 12. Basic syntax

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy="mean"
)
```

Then:

```python
imputer.fit(X_train)
```

and:

```python
X_train_new = imputer.transform(X_train)
```

Shortcut:

```python
X_train_new = imputer.fit_transform(X_train)
```

Test:

```python
X_test_new = imputer.transform(X_test)
```

---

# 13. `strategy="mean"`

Example:

```python
import numpy as np

X = np.array([
    [20],
    [30],
    [np.nan],
    [40]
])
```

Create imputer:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy="mean"
)
```

Fit:

```python
imputer.fit(X)
```

Mean:

$$
\frac{20+30+40}{3}=30
$$

Then:

```python
X_new = imputer.transform(X)
```

Result:

```text
20
30
30
40
```

So:

```text
fit()
→ mean learn

transform()
→ NaN ko mean se replace
```

---

# 14. `mean` numeric data ke liye

`strategy="mean"` numeric data ke liye hoti hai. ([scikit-learn][1])

Example appropriate:

```text
Age
Salary
Height
Weight
Temperature
```

Not:

```text
City
Color
Gender
```

Tum city names ka mathematical mean nahi nikal sakte.

---

# 15. Mean ka problem — Outliers

Suppose salary:

```text
20,000
25,000
30,000
35,000
10,00,000
NaN
```

Last value huge outlier hai.

Mean significantly high ho jayega.

So missing salary ko mean se fill karna misleading ho sakta hai.

Aise case me:

```text
Median
```

better ho sakta hai.

---

# 16. `strategy="median"`

```python
imputer = SimpleImputer(
    strategy="median"
)
```

Example:

```text
10
20
30
1000
NaN
```

Sorted:

```text
10
20
30
1000
```

Median:

$$
\frac{20+30}{2}=25
$$

Mean:

$$
265
$$

Compare:

```text
Mean = 265
Median = 25
```

Because `1000` outlier mean ko strongly affect kar raha hai.

Median more robust hai.

---

# 17. Mean vs Median

Easy rule:

```text
Numeric data
+
distribution roughly normal / few outliers
→ Mean can be reasonable
```

```text
Numeric data
+
strong outliers / skewed distribution
→ Median often better
```

But ye absolute rule nahi hai.

Best strategy data distribution aur model performance se validate karni chahiye.

---

# 18. `strategy="most_frequent"`

Most frequent = Mode.

Example:

```text
City

Indore
Dewas
Indore
NaN
Bhopal
Indore
```

Most frequent:

```text
Indore
```

Use:

```python
imputer = SimpleImputer(
    strategy="most_frequent"
)
```

After transform:

```text
Indore
Dewas
Indore
Indore
Bhopal
Indore
```

`most_frequent` numeric aur string data dono ke saath use kiya ja sakta hai. ([scikit-learn][1])

---

# 19. Most Frequent kahan useful hai?

Categorical features:

```text
City
Gender
Education
Department
Color
```

Example:

```python
categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)
```

Often useful.

Numeric discrete data me bhi possible hai.

---

# 20. `strategy="constant"`

Kabhi hum statistics calculate nahi karna chahte.

Instead fixed value fill karna chahte hain.

Example:

```text
City

Indore
NaN
Dewas
```

Fill missing with:

```text
"Unknown"
```

Use:

```python
imputer = SimpleImputer(
    strategy="constant",
    fill_value="Unknown"
)
```

Output:

```text
Indore
Unknown
Dewas
```

Official API me `constant` strategy `fill_value` use karti hai. ([scikit-learn][1])

---

# 21. Numeric constant example

```python
imputer = SimpleImputer(
    strategy="constant",
    fill_value=0
)
```

Input:

```text
10
NaN
30
```

Output:

```text
10
0
30
```

---

# 22. Strategies summary

| Strategy          | Replacement        | Data type             |
| ----------------- | ------------------ | --------------------- |
| `"mean"`          | Column mean        | Numeric               |
| `"median"`        | Column median      | Numeric               |
| `"most_frequent"` | Mode               | Numeric / categorical |
| `"constant"`      | Fixed `fill_value` | Numeric / categorical |

Current sklearn also supports a callable strategy for custom per-column scalar imputation, but hum ise advanced usage me dekhenge. ([scikit-learn][1])

---

# 23. Multiple columns par mean kaise work karta hai?

Suppose:

```python
X = np.array([
    [20, 20000],
    [30, np.nan],
    [np.nan, 40000],
    [40, 60000]
])
```

Columns:

```text
Age | Salary
```

`SimpleImputer(strategy="mean")` **har column ka separate mean** calculate karega.

Age:

$$
(20+30+40)/3=30
$$

Salary:

$$
(20000+40000+60000)/3=40000
$$

So transformed:

```text
20   20000
30   40000
30   40000
40   60000
```

Important:

```text
One mean for entire dataset
```

nahi.

Instead:

```text
One statistic per column
```

---

# 24. Learned values `statistics_`

Remember chapter 2?

Fitted attributes often `_` par end hote hain.

SimpleImputer:

```python
imputer.fit(X_train)
```

Then:

```python
print(imputer.statistics_)
```

Example:

```text
[30.0, 40000.0]
```

Meaning:

```text
Age fill value = 30
Salary fill value = 40000
```

`statistics_` har feature ke liye learned imputation value contain karta hai. ([scikit-learn][1])

---

# 25. Isme Chapter 2 ka `fit()` concept dekho

```python
imputer.fit(X_train)
```

Meaning:

```text
Missing values fill nahi hue yet
```

Mainly required statistics learn hui.

Then:

```python
imputer.transform(X_train)
```

Meaning:

```text
Learned statistics use karke
missing values replace karo
```

So exactly:

```text
fit()
= LEARN

transform()
= APPLY
```

---

# 26. `fit_transform()`

Instead of:

```python
imputer.fit(X_train)

X_train_clean = imputer.transform(X_train)
```

write:

```python
X_train_clean = imputer.fit_transform(
    X_train
)
```

Meaning:

```text
Learn statistics
+
Fill missing values
```

---

# 27. Train-Test rule

Ye chapter ka sabse important ML rule hai.

Correct:

```python
X_train_clean = imputer.fit_transform(
    X_train
)

X_test_clean = imputer.transform(
    X_test
)
```

Meaning:

```text
Training data
→ statistics learn

Test data
→ training statistics use
```

---

# 28. Test set ka own mean kyon nahi?

Suppose:

Training Age:

```text
20
30
NaN
40
```

Training mean:

```text
30
```

Test:

```text
NaN
80
100
```

If we do:

```python
imputer.transform(X_test)
```

missing value becomes:

```text
30
```

because training mean use hoga.

Correct:

```text
30
80
100
```

---

# 29. Wrong approach

```python
X_train = imputer.fit_transform(X_train)

X_test = imputer.fit_transform(X_test)
```

Wrong because test set se new statistics learn ho jayengi.

Test mean:

$$
(80+100)/2=90
$$

Then missing test Age becomes:

```text
90
```

But model training data preprocessing:

```text
mean = 30
```

par based tha.

Now inconsistent preprocessing.

---

# 30. Correct approach

```python
imputer = SimpleImputer(
    strategy="mean"
)

X_train = imputer.fit_transform(
    X_train
)

X_test = imputer.transform(
    X_test
)
```

Golden rule:

```text
Train
→ fit_transform()

Test
→ transform()
```

---

# 31. Split before imputation

Wrong:

```python
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y
)
```

Why wrong?

Because imputer ne complete dataset se statistic calculate ki.

Test data ki information training preprocessing me leak ho gayi.

This is:

```text
Data Leakage
```

Correct order:

```text
X/y
↓
Train-Test Split
↓
Fit imputer on X_train
↓
Transform X_train
↓
Transform X_test
```

---

# 32. Correct complete flow

```python
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
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

Imputer:

```python
imputer = SimpleImputer(
    strategy="median"
)
```

Train:

```python
X_train_clean = imputer.fit_transform(
    X_train
)
```

Test:

```python
X_test_clean = imputer.transform(
    X_test
)
```

Then:

```python
model.fit(
    X_train_clean,
    y_train
)
```

And:

```python
y_pred = model.predict(
    X_test_clean
)
```

---

# 33. Full mental flow

```text
Raw Dataset
      ↓
Create X/y
      ↓
Train-Test Split
      ↓
┌──────────────────────────┐
↓                          ↓
X_train                   X_test
↓                          |
imputer.fit()              |
↓                          |
learn median/mean/etc.     |
↓                          |
transform()             transform()
↓                          ↓
X_train_clean          X_test_clean
↓
model.fit()
                           ↓
                       model.predict()
```

---

# 34. Numerical + categorical columns together problem

Suppose:

| Age | Salary | City   |
| --: | -----: | ------ |
|  20 |  20000 | Indore |
| NaN |  40000 | Dewas  |
|  40 |    NaN | NaN    |

We want:

```text
Age
→ median

Salary
→ median

City
→ most_frequent
```

One `SimpleImputer` object normally has one configured strategy.

So simply:

```python
SimpleImputer(strategy="median")
```

whole mixed dataset par lagana correct nahi.

Instead separate imputers:

```text
Numerical
→ median

Categorical
→ most_frequent
```

Later **ColumnTransformer** exactly isi problem ko elegantly solve karega.

---

# 35. Manual approach preview

```python
numeric_columns = [
    "Age",
    "Salary"
]

categorical_columns = [
    "City"
]
```

Numeric:

```python
num_imputer = SimpleImputer(
    strategy="median"
)
```

Categorical:

```python
cat_imputer = SimpleImputer(
    strategy="most_frequent"
)
```

Later:

```text
ColumnTransformer
```

automatically correct columns par correct transformations apply karega.

---

# 36. `missing_values` parameter

Default:

```python
SimpleImputer(
    missing_values=np.nan
)
```

Current API `np.nan` ke saath numeric/string/`None`/`pd.NA` jaise configured missing markers support karti hai. ([scikit-learn][1])

Suppose dataset me missing values weirdly:

```text
-999
```

se represented hain.

Example:

```text
Age

20
30
-999
40
```

We know:

```text
-999 = missing
```

Then:

```python
imputer = SimpleImputer(
    missing_values=-999,
    strategy="mean"
)
```

Now `-999` ko missing consider kiya jayega.

---

# 37. Example `missing_values=-1`

```python
X = [
    [20],
    [-1],
    [40]
]
```

If `-1` means missing:

```python
imputer = SimpleImputer(
    missing_values=-1,
    strategy="mean"
)
```

Mean from:

```text
20 and 40
```

will be:

```text
30
```

Transformed:

```text
20
30
40
```

---

# 38. `fill_value`

`fill_value` mainly:

```python
strategy="constant"
```

ke saath use hota hai.

Example categorical:

```python
imputer = SimpleImputer(
    strategy="constant",
    fill_value="Unknown"
)
```

Numeric:

```python
imputer = SimpleImputer(
    strategy="constant",
    fill_value=0
)
```

---

# 39. When use `"Unknown"`?

Suppose:

```text
Employment Type

Full Time
Part Time
NaN
Self Employed
```

Instead of most frequent, sometimes missing itself meaningful hai.

We can create:

```text
Full Time
Part Time
Unknown
Self Employed
```

Then:

```python
SimpleImputer(
    strategy="constant",
    fill_value="Unknown"
)
```

This preserves explicit information that value was not known.

---

# 40. Missingness itself information ho sakti hai

Interesting example:

```text
Income missing
```

Maybe applicant ne income reveal nahi ki.

That fact itself:

```text
Income was missing
```

predictive ho sakta hai.

If we simply fill:

```text
Income = median
```

model forget kar deta hai ki originally missing tha.

Isi case ke liye:

```python
add_indicator=True
```

useful ho sakta hai.

---

# 41. `add_indicator=True`

Example:

```python
imputer = SimpleImputer(
    strategy="median",
    add_indicator=True
)
```

Suppose original:

```text
Age

20
NaN
40
```

Median:

```text
30
```

Normal output:

```text
20
30
40
```

With indicator conceptually:

```text
Age   Age_was_missing

20       0
30       1
40       0
```

So model ko dono information milti hai:

```text
Filled Age
+
Was Age originally missing?
```

Official docs ke according `add_indicator=True` output me missingness indicator features append karta hai; indicator only un features ke liye banta hai jahan fit/train time par missing values dekhi gayi hon. ([scikit-learn][1])

---

# 42. Important `add_indicator` detail

Suppose training me:

```text
Age had NaN
Salary had no NaN
```

Then indicator:

```text
Age_missing
```

ban sakta hai.

But:

```text
Salary_missing
```

indicator fit time par nahi banega.

Even if test me later Salary missing mil jaye, fit time me missingness observed nahi thi, so that feature indicator output me automatically appear nahi karega. ([scikit-learn][1])

---

# 43. `statistics_` example

```python
import numpy as np
from sklearn.impute import SimpleImputer

X = np.array([
    [20, 20000],
    [30, 40000],
    [np.nan, 60000],
    [40, np.nan]
])

imputer = SimpleImputer(
    strategy="mean"
)

imputer.fit(X)
```

Then:

```python
print(imputer.statistics_)
```

Expected:

```text
[30, 40000]
```

Because:

```text
Age Mean = 30

Salary Mean = 40000
```

---

# 44. `n_features_in_`

After fit:

```python
print(imputer.n_features_in_)
```

If training data has:

```text
Age
Salary
Experience
```

then:

```text
3
```

This tells how many features estimator saw during fit. Current `SimpleImputer` exposes this fitted attribute. ([scikit-learn][1])

---

# 45. `feature_names_in_`

If input was a DataFrame with string column names, fitted estimator may have:

```python
imputer.feature_names_in_
```

Example:

```text
['Age', 'Salary']
```

Useful for debugging preprocessing.

---

# 46. All values in a column missing

Suppose training:

```text
Age     Salary

NaN     20000
NaN     30000
NaN     40000
```

What is:

```text
Age mean?
```

Impossible because Age has no observed values.

Current `SimpleImputer` has:

```python
keep_empty_features=False
```

by default; `keep_empty_features=True` can preserve features that were entirely missing at fit time. ([scikit-learn][1])

This is more advanced, but good to know.

---

# 47. `keep_empty_features=True`

Example:

```python
imputer = SimpleImputer(
    strategy="mean",
    keep_empty_features=True
)
```

Purpose:

```text
Column completely missing during fit
→ still preserve feature in transformed result
```

This can matter when maintaining a fixed feature structure.

---

# 48. Target `y` missing ho to?

Important.

Suppose:

| Age | Salary | Purchased |
| --: | -----: | --------: |
|  20 |  25000 |         0 |
|  30 |  40000 |       NaN |
|  40 |  60000 |         1 |

`Purchased` target hai.

Ordinary workflow me:

```python
SimpleImputer
```

ko target labels blindly fill karne ke liye use nahi karna chahiye.

Why?

Because if target unknown:

```text
Hum correct training answer hi nahi jaante.
```

For supervised ML, usually such rows ko target-specific logic se handle karte hain, often remove from labeled training data.

Features and targets ki missingness different problems hain.

---

# 49. Missing values ko zero se blindly fill mat karo

Beginners often:

```python
df.fillna(0)
```

kar dete hain.

Problem:

Suppose Age missing:

```text
Age = 0
```

does this actually mean newborn?

Or missing?

Model distinguish nahi karega unless context/indicator diya.

Salary:

```text
Salary = 0
```

may mean actually unemployed, which is different from unknown salary.

So zero is not neutral automatically.

---

# 50. Mean imputation ke disadvantages

Mean easy hai but limitations hain:

```text
Distribution distort kar sakta hai
Variance reduce kar sakta hai
Outliers se affect hota hai
Artificial repeated value create karta hai
Feature relationships ignore karta hai
```

Example:

```text
20
30
40
50
NaN
NaN
NaN
```

Mean:

```text
35
```

After:

```text
20
30
40
50
35
35
35
```

Many artificial 35 values create ho gaye.

---

# 51. Median ke disadvantages

Median outlier-resistant hai but still:

```text
Single constant value se many missing entries fill hoti hain
Other feature relationships consider nahi karta
Distribution change kar sakta hai
```

So SimpleImputer simple baseline hai, necessarily perfect solution nahi.

---

# 52. `SimpleImputer` is univariate

Suppose:

```text
Age
Salary
Experience
```

Salary missing hai.

SimpleImputer median strategy:

```text
Salary column ke observed salaries
```

use karega.

Ye normally nahi dekhega:

```text
Age kya hai?
Experience kya hai?
```

Salary estimate karne ke liye.

That's why it's called:

```text
Univariate imputation
```

Advanced approaches:

```text
KNNImputer
IterativeImputer
```

multiple features ki information use kar sakte hain. Current sklearn docs `KNNImputer` aur `IterativeImputer` ko multivariate alternatives ke roop me describe karti hain. ([scikit-learn][1])

---

# 53. `KNNImputer` ka preview

Suppose:

```text
Age  Experience  Salary

25       2        30000
27       3        33000
26       2        NaN
50      20        90000
```

Missing Salary wale person ke similar rows:

```text
Age ≈ 26
Experience ≈ 2
```

find karke Salary estimate kar sakte hain.

We'll cover this separately later.

---

# 54. SimpleImputer kab use karna chahiye?

Good starting point when:

```text
You need a simple baseline
Missingness isn't extremely complex
Numerical columns need median/mean
Categorical columns need most-frequent/constant
You want pipeline-compatible preprocessing
```

Especially:

```text
Median numeric
+
Most frequent categorical
```

is a common baseline.

---

# 55. Mean ya median kaise choose karein?

Suppose:

```python
df["Salary"].describe()
```

and visualization shows strong right skew:

```text
Most salaries:
20k–80k

Few salaries:
10 lakh+
```

Then:

```text
Median
```

often reasonable.

If distribution symmetric and no strong outliers:

```text
Mean
```

may be reasonable.

Better approach eventually:

```text
Compare through cross-validation
```

instead of assuming.

---

# 56. Categorical strategy kaise choose karein?

Suppose:

```text
City
```

missing.

Options:

```text
most_frequent
```

means:

> Missing city ko most common city maan lo.

While:

```text
constant = "Unknown"
```

means:

> Hume city pata nahi, explicitly Unknown rakho.

Sometimes `"Unknown"` is more honest and informative.

Choice depends on what missingness means.

---

# 57. Missing data mechanisms ka basic intuition

Advanced statistics me missingness ke types discuss kiye jate hain:

```text
MCAR
MAR
MNAR
```

Abhi detailed theory nahi, but concept important:

Missing value random hai ya kisi pattern ki wajah se?

Example:

```text
Income missing
```

because random form error:

different situation.

Income missing because high-income people intentionally skip:

different situation.

So imputation strategy ko context ke saath choose karna chahiye.

---

# 58. Practical Pandas example

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Age": [20, 30, np.nan, 50, 40],
    "Salary": [
        25000,
        np.nan,
        50000,
        80000,
        60000
    ],
    "Purchased": [0, 0, 1, 1, 1]
})
```

Create X/y:

```python
X = df.drop(
    columns=["Purchased"]
)

y = df["Purchased"]
```

Split:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Imputer:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy="median"
)
```

Fit + transform training:

```python
X_train_clean = imputer.fit_transform(
    X_train
)
```

Test:

```python
X_test_clean = imputer.transform(
    X_test
)
```

Check:

```python
print(imputer.statistics_)
```

---

# 59. Why output may become NumPy array

Suppose input:

```python
X_train
```

is DataFrame.

After:

```python
X_train_clean = imputer.fit_transform(
    X_train
)
```

you may receive array-like output rather than original DataFrame representation.

So:

```python
type(X_train_clean)
```

may differ from:

```python
type(X_train)
```

If you need DataFrame:

```python
X_train_clean = pd.DataFrame(
    X_train_clean,
    columns=X_train.columns,
    index=X_train.index
)
```

Later ColumnTransformer/Pipeline me is handling ko better organize karenge.

---

# 60. Complete model example

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
```

Dataset:

```python
df = pd.DataFrame({
    "Age": [
        20, 25, np.nan, 35,
        40, 45, 50, np.nan
    ],
    "Salary": [
        20000, 30000, 40000, np.nan,
        60000, 70000, 80000, 90000
    ],
    "Purchased": [
        0, 0, 0, 0,
        1, 1, 1, 1
    ]
})
```

Separate:

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

Imputer:

```python
imputer = SimpleImputer(
    strategy="median"
)

X_train_clean = imputer.fit_transform(
    X_train
)

X_test_clean = imputer.transform(
    X_test
)
```

Model:

```python
model = LogisticRegression()

model.fit(
    X_train_clean,
    y_train
)
```

Predict:

```python
y_pred = model.predict(
    X_test_clean
)
```

---

# 61. New production data

Suppose model deployed.

New customer:

```python
new_customer = pd.DataFrame({
    "Age": [np.nan],
    "Salary": [55000]
})
```

Important:

Don't do:

```python
imputer.fit_transform(new_customer)
```

Do:

```python
new_customer_clean = imputer.transform(
    new_customer
)
```

Then:

```python
prediction = model.predict(
    new_customer_clean
)
```

Flow:

```text
New Data
↓
Existing fitted imputer
↓
transform()
↓
Existing trained model
↓
predict()
```

---

# 62. Why Pipeline will become important

Right now:

```python
X_train = imputer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)

X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)

model.fit(X_train, y_train)
```

Many steps.

Later Pipeline:

```python
pipeline.fit(
    X_train,
    y_train
)

pipeline.predict(
    X_test
)
```

Internally:

```text
Imputation
↓
Scaling
↓
Model
```

handle karegi.

This greatly reduces preprocessing mistakes.

---

# 63. Common Mistake #1 — Impute before split

Wrong:

```python
X = imputer.fit_transform(X)

train_test_split(X, y)
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

---

# 64. Common Mistake #2 — Test par `fit_transform`

Wrong:

```python
X_test = imputer.fit_transform(X_test)
```

Correct:

```python
X_test = imputer.transform(X_test)
```

---

# 65. Common Mistake #3 — Mean categorical column par

Wrong idea:

```text
City:
Indore
Dewas
Bhopal
```

Using:

```python
SimpleImputer(strategy="mean")
```

Mean only numeric strategy hai. ([scikit-learn][1])

For categorical:

```python
SimpleImputer(
    strategy="most_frequent"
)
```

or:

```python
SimpleImputer(
    strategy="constant",
    fill_value="Unknown"
)
```

---

# 66. Common Mistake #4 — Mean always best assume karna

Don't think:

```text
Missing numeric
→ Mean always
```

Instead inspect:

```text
Distribution
Outliers
Meaning of feature
Amount of missingness
```

Mean vs median can matter significantly.

---

# 67. Common Mistake #5 — Target impute karna

Feature missing:

```text
Age = NaN
```

and target missing:

```text
Purchased = NaN
```

same issue nahi.

Feature can often be estimated.

Target:

```text
correct label itself unknown
```

hai.

Supervised training ke liye different handling required.

---

# 68. Common Mistake #6 — Missing code ko real number samajhna

Dataset may use:

```text
-1
999
9999
```

as missing value markers.

Example:

```text
Age = 999
```

If you don't know dataset documentation, model may treat:

```text
999 years old
```

as genuine data.

Always understand special missing codes.

---

# 69. Common Mistake #7 — Different train/test imputers

Wrong:

```python
train_imputer = SimpleImputer(
    strategy="median"
)

test_imputer = SimpleImputer(
    strategy="median"
)
```

Then independently fit.

Correct:

```python
imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(X_train)

X_test = imputer.transform(X_test)
```

Same fitted imputer.

---

# 70. SimpleImputer and `fit(X, y)`

You might see API signature:

```python
imputer.fit(X, y=None)
```

But for SimpleImputer:

```text
y is ignored
```

because it learns imputation statistics from X. Official documentation explicitly notes that `y` is unused and exists for API consistency. ([scikit-learn][1])

So normally:

```python
imputer.fit(X_train)
```

is enough.

---

# 71. Chapter 8 mental model

Remember:

```text
Missing values
      ↓
SimpleImputer
      ↓
fit(X_train)
      ↓
Learn replacement statistic
      ↓
┌─────────────────┐
↓                 ↓
X_train           X_test
transform         transform
↓                 ↓
clean train       clean test
```

Example mean:

```text
X_train:

20
30
NaN
40

↓ fit

Learn mean = 30
```

Then:

```text
Train NaN
→ 30
```

And test:

```text
Test NaN
→ also 30
```

because test must use **training statistics**.

---

# 72. Strategy selection cheat sheet

```text
Numeric + no major outliers
→ mean can be tried

Numeric + outliers/skew
→ median is often a strong baseline

Categorical
→ most_frequent

Missing has its own meaning
→ constant ("Unknown", etc.)

Need model to know value was missing
→ add_indicator=True
```

This is a starting framework—not a universal law.

---

# 73. Chapter 8 Final Summary

Core import:

```python
from sklearn.impute import SimpleImputer
```

Numeric:

```python
imputer = SimpleImputer(
    strategy="median"
)
```

Categorical:

```python
imputer = SimpleImputer(
    strategy="most_frequent"
)
```

Constant:

```python
imputer = SimpleImputer(
    strategy="constant",
    fill_value="Unknown"
)
```

Training:

```python
X_train = imputer.fit_transform(
    X_train
)
```

Testing:

```python
X_test = imputer.transform(
    X_test
)
```

Learned fill values:

```python
imputer.statistics_
```

Core concept:

```text
fit()
→ replacement values learn

transform()
→ missing values replace

fit_transform()
→ learn + replace
```

And the golden rule:

```text
Split first
↓
Fit imputer on training data only
↓
Transform train
↓
Transform test
```

### Quick practice

Suppose training Salary:

```text
25000
30000
NaN
35000
500000
```

Think:

1. Mean ya median me se kaunsa safer starting choice hoga, aur kyon?
2. Agar training median `32500` learn hua aur test Salary `NaN` hai, test me kya value jayegi?
3. `most_frequent` numerical aur categorical dono me use ho sakta hai ya nahi?
4. `imputer.statistics_` kya store karta hai?
5. `add_indicator=True` ka model ko kya extra information milti hai?


