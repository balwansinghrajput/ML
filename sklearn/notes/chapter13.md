# Chapter 13 — `ColumnTransformer` Deep Dive

Ab tak humne alag-alag preprocessing tools seekhe:

```text
SimpleImputer
StandardScaler
OneHotEncoder
OrdinalEncoder
```

Real-world dataset me problem ye hoti hai ki **har column par same preprocessing apply nahi hoti**.

Example:

| Age | Salary | City | Education |
|---:|---:|---|---|
| 25 | 30000 | Indore | Bachelor |
| 35 | NaN | Dewas | Master |
| NaN | 70000 | Bhopal | PhD |

Hume chahiye:

```text
Age, Salary
→ missing values fill
→ scaling

City
→ missing value fill
→ OneHotEncoder

Education
→ OrdinalEncoder
```

Isi problem ko `ColumnTransformer` solve karta hai.

Scikit-learn ke current stable docs ke according `ColumnTransformer` different columns/subsets par different transformers apply karta hai aur sabke generated features ko combine karke ek single feature space banata hai. :chatgpt-content-reference{index="0"}

---

## 1. `ColumnTransformer` kya hai?

Simple definition:

> **`ColumnTransformer` hume dataset ke different columns par different preprocessing transformations apply karne deta hai.**

Mental model:

```text
                    Dataset
                       ↓
             ┌─────────┴─────────┐
             ↓                   ↓
      Numerical Columns    Categorical Columns
             ↓                   ↓
      Imputer + Scaler     Imputer + Encoder
             ↓                   ↓
             └─────────┬─────────┘
                       ↓
              Combined Features
```

---

# 2. Import

```python
from sklearn.compose import ColumnTransformer
```

Basic syntax:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("name1", transformer1, columns1),
        ("name2", transformer2, columns2)
    ]
)
```

Each tuple has 3 parts:

```text
(
    name,
    transformer,
    columns
)
```

Example:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Age", "Salary"]),
        ("cat", OneHotEncoder(), ["City"])
    ]
)
```

Meaning:

```text
"num"
→ transformer ka naam

StandardScaler()
→ kya transformation apply hogi

["Age", "Salary"]
→ kin columns par
```

---

# 3. Simple example

Dataset:

```python
import pandas as pd

df = pd.DataFrame({
    "Age": [20, 30, 40, 50],
    "Salary": [20000, 40000, 60000, 80000],
    "City": ["Indore", "Dewas", "Bhopal", "Indore"]
})
```

Numerical columns:

```python
num_cols = ["Age", "Salary"]
```

Categorical:

```python
cat_cols = ["City"]
```

ColumnTransformer:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            num_cols
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            cat_cols
        )
    ]
)
```

Transform:

```python
X_new = preprocessor.fit_transform(df)
```

Flow:

```text
Age, Salary
    ↓
StandardScaler
    ↓
scaled numbers

City
    ↓
OneHotEncoder
    ↓
City_Bhopal
City_Dewas
City_Indore

        ↓

everything combined
```

---

# 4. Output conceptually kaisa hoga?

Original:

```text
Age   Salary   City
20    20000    Indore
30    40000    Dewas
40    60000    Bhopal
50    80000    Indore
```

After preprocessing roughly:

```text
Age_scaled
Salary_scaled
City_Bhopal
City_Dewas
City_Indore
```

So 3 original columns:

```text
Age
Salary
City
```

5 transformed features me convert ho sakte hain.

Important:

> ColumnTransformer output columns ki count original columns ke equal rehna necessary nahi.

OneHotEncoder new columns create kar sakta hai.

---

# 5. Missing values ke saath real problem

Ab realistic data:

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "Age": [20, 30, np.nan, 50],
    "Salary": [
        20000,
        np.nan,
        60000,
        80000
    ],
    "City": [
        "Indore",
        "Dewas",
        None,
        "Bhopal"
    ]
})
```

We want:

```text
Age + Salary
→ median
→ StandardScaler
```

City:

```text
City
→ most_frequent
→ OneHotEncoder
```

Ab ek column ko **multiple sequential transformations** chahiye.

Isliye `Pipeline` ko ColumnTransformer ke andar use karte hain.

Scikit-learn ka official mixed-type example bhi numerical branch me imputation + scaling aur categorical branch me categorical preprocessing ko `ColumnTransformer` ke through combine karta hai. :chatgpt-content-reference{index="1"}

---

# 6. Numerical Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)
```

Flow:

```text
Age / Salary
      ↓
SimpleImputer
      ↓
missing values filled
      ↓
StandardScaler
      ↓
scaled output
```

Order important hai:

```text
Impute first
↓
Scale later
```

---

# 7. Categorical Pipeline

```python
from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)
```

Flow:

```text
City
 ↓
SimpleImputer
 ↓
missing category filled
 ↓
OneHotEncoder
 ↓
binary columns
```

---

# 8. Ab dono ko ColumnTransformer me combine karo

```python
numeric_features = [
    "Age",
    "Salary"
]

categorical_features = [
    "City"
]
```

Then:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)
```

This is the core real-world architecture.

---

# 9. Full architecture visually

```text
                    X
                    ↓
           ColumnTransformer
                    ↓

      ┌─────────────┴─────────────┐
      ↓                           ↓
Numerical Columns           Categorical Columns
Age, Salary                     City
      ↓                           ↓
SimpleImputer                SimpleImputer
(median)                    (most_frequent)
      ↓                           ↓
StandardScaler              OneHotEncoder
      ↓                           ↓
Numerical Output            Categorical Output
      └─────────────┬─────────────┘
                    ↓
             Combined Matrix
```

Ye architecture tum real ML projects me bahut baar use karoge.

---

# 10. `fit_transform()` kya karega?

```python
X_train_new = preprocessor.fit_transform(
    X_train
)
```

ColumnTransformer internally:

```text
Numerical branch:
fit imputer
transform
fit scaler
transform

Categorical branch:
fit imputer
transform
fit encoder
transform

Then:
outputs concatenate
```

So one command:

```python
preprocessor.fit_transform(X_train)
```

kaafi preprocessing perform kar sakta hai.

---

# 11. Test data

Golden rule same:

```python
X_train_new = preprocessor.fit_transform(
    X_train
)

X_test_new = preprocessor.transform(
    X_test
)
```

Not:

```python
preprocessor.fit_transform(X_test)
```

Why?

Because test data se:

```text
median
mean/std
categories
```

learn nahi karni.

---

# 12. Train-test full example

```python
from sklearn.model_selection import train_test_split

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
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Preprocessing:

```python
X_train_new = preprocessor.fit_transform(
    X_train
)

X_test_new = preprocessor.transform(
    X_test
)
```

Now:

```text
X_train_new
X_test_new
```

model-ready numerical feature matrices hain.

---

# 13. `ColumnTransformer` model nahi hai

Remember:

```text
ColumnTransformer
→ Transformer
```

Isliye:

```python
preprocessor.fit()
preprocessor.transform()
preprocessor.fit_transform()
```

available hain.

But normally:

```python
preprocessor.predict()
```

nahi.

Prediction later model karega.

---

# 14. `remainder` parameter

Bahut important.

Suppose dataset:

```text
Age
Salary
City
Experience
```

ColumnTransformer me sirf:

```text
Age
Salary
City
```

mention kiye.

What happens to:

```text
Experience
```

?

Default:

```python
remainder="drop"
```

hai.

Matlab unlisted columns drop ho jati hain. Current `ColumnTransformer` docs bhi default `remainder='drop'` specify karti hain. :chatgpt-content-reference{index="2"}

---

# 15. `remainder="passthrough"`

If you want unspecified columns unchanged:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            ["Age", "Salary"]
        ),
        (
            "cat",
            OneHotEncoder(),
            ["City"]
        )
    ],
    remainder="passthrough"
)
```

Then:

```text
Experience
```

output me unchanged rahega.

Flow:

```text
Age
→ scaled

Salary
→ scaled

City
→ one-hot

Experience
→ unchanged
```

`remainder="passthrough"` non-specified columns ko output me append karta hai. :chatgpt-content-reference{index="3"}

---

# 16. `remainder="drop"` vs `"passthrough"`

```text
remainder="drop"

Unlisted columns
→ remove
```

```text
remainder="passthrough"

Unlisted columns
→ keep unchanged
```

Default:

```text
drop
```

---

# 17. Specific column drop karna

Transformer ki jagah:

```text
"drop"
```

use kar sakte ho.

Example:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "drop_id",
            "drop",
            ["CustomerID"]
        ),
        (
            "num",
            StandardScaler(),
            ["Age", "Salary"]
        )
    ],
    remainder="passthrough"
)
```

Meaning:

```text
CustomerID
→ remove
```

Current API transformers me estimator ke saath special `"drop"` aur `"passthrough"` values support karti hai. :chatgpt-content-reference{index="4"}

---

# 18. Specific columns passthrough

Example:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "scale",
            StandardScaler(),
            ["Age", "Salary"]
        ),
        (
            "keep",
            "passthrough",
            ["Experience"]
        )
    ]
)
```

Here:

```text
Age / Salary
→ scale

Experience
→ unchanged
```

---

# 19. Ordinal + Nominal + Numeric together

Ab more realistic.

Dataset:

```text
Age
Salary
City
Education
Purchased
```

Treatment:

```text
Age, Salary
→ median
→ StandardScaler

City
→ most frequent
→ OneHotEncoder

Education
→ most frequent
→ OrdinalEncoder
```

We can have **3 branches**.

---

# 20. Ordinal transformer

```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "ordinal",
            OrdinalEncoder(
                categories=[
                    [
                        "School",
                        "Bachelor",
                        "Master",
                        "PhD"
                    ]
                ],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)
```

Then:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            ["Age", "Salary"]
        ),
        (
            "nominal",
            categorical_transformer,
            ["City"]
        ),
        (
            "ordinal",
            ordinal_transformer,
            ["Education"]
        )
    ]
)
```

---

# 21. Output order

Suppose transformer list:

```python
[
    ("num", ..., ["Age", "Salary"]),
    ("cat", ..., ["City"])
]
```

Output generally transformer order me concatenate hota hai:

```text
Numerical transformed features
then
Categorical transformed features
```

Conceptually:

```text
Age_scaled
Salary_scaled
City_Bhopal
City_Dewas
City_Indore
```

Original DataFrame column order exactly preserve hona required nahi.

Always transformed feature names inspect karna better hai.

---

# 22. `get_feature_names_out()`

Bahut useful:

```python
preprocessor.get_feature_names_out()
```

Example output:

```text
num__Age
num__Salary
cat__City_Bhopal
cat__City_Dewas
cat__City_Indore
```

Default `verbose_feature_names_out=True` hone par transformer name prefix add hota hai, jaise `num__` aur `cat__`. :chatgpt-content-reference{index="5"}

This is extremely useful for:

```text
debugging
feature importance
model interpretation
```

---

# 23. Cleaner names

You can use:

```python
preprocessor = ColumnTransformer(
    transformers=[...],
    verbose_feature_names_out=False
)
```

Then output names may look like:

```text
Age
Salary
City_Bhopal
City_Dewas
City_Indore
```

But names unique honi chahiye; otherwise sklearn error raise kar sakta hai. :chatgpt-content-reference{index="6"}

---

# 24. Pandas output directly

Modern sklearn me:

```python
preprocessor.set_output(
    transform="pandas"
)
```

use kar sakte ho.

Then:

```python
X_train_new = preprocessor.fit_transform(
    X_train
)
```

DataFrame output mil sakta hai instead of unnamed NumPy/sparse output. Scikit-learn ka current `set_output` API transformers ko pandas ya polars output ke liye configure kar sakta hai. :chatgpt-content-reference{index="7"}

This makes transformed columns inspect karna easier.

---

# 25. Example

```python
preprocessor.set_output(
    transform="pandas"
)

X_new = preprocessor.fit_transform(X)

print(X_new.head())
```

Now you may see:

```text
num__Age
num__Salary
cat__City_Bhopal
cat__City_Dewas
cat__City_Indore
```

directly as DataFrame columns.

---

# 26. `named_transformers_`

After fit:

```python
preprocessor.fit(X_train)
```

You can access fitted branches:

```python
preprocessor.named_transformers_
```

Example:

```python
preprocessor.named_transformers_["num"]
```

returns fitted numerical transformer.

Categorical:

```python
preprocessor.named_transformers_["cat"]
```

Useful for inspection/debugging.

---

# 27. Nested Pipeline access

Suppose:

```python
num_pipe = preprocessor.named_transformers_[
    "num"
]
```

Then:

```python
num_pipe.named_steps["imputer"]
```

or:

```python
num_pipe.named_steps["scaler"]
```

This lets you inspect learned values:

```python
num_pipe.named_steps[
    "imputer"
].statistics_
```

and:

```python
num_pipe.named_steps[
    "scaler"
].mean_
```

This becomes very powerful in real projects.

---

# 28. Categorical categories inspect karna

If categorical branch:

```text
SimpleImputer
→ OneHotEncoder
```

then:

```python
cat_pipe = preprocessor.named_transformers_[
    "cat"
]
```

Get encoder:

```python
encoder = cat_pipe.named_steps[
    "onehot"
]
```

Then:

```python
print(
    encoder.categories_
)
```

This tells you categories learned from training data.

---

# 29. Selecting columns manually

Most beginner-friendly:

```python
numeric_features = [
    "Age",
    "Salary"
]

categorical_features = [
    "City",
    "Gender"
]
```

Then use these lists.

This gives you maximum control.

---

# 30. Automatic dtype selection

Scikit-learn also provides:

```python
from sklearn.compose import make_column_selector
```

Example:

```python
numeric_selector = make_column_selector(
    dtype_include="number"
)
```

Categorical:

```python
categorical_selector = make_column_selector(
    dtype_exclude="number"
)
```

Then:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_selector
        ),
        (
            "cat",
            categorical_transformer,
            categorical_selector
        )
    ]
)
```

Current sklearn examples show column selection both by explicit names and by data types using selectors. :chatgpt-content-reference{index="8"}

---

# 31. Automatic dtype selection me caution

Suppose:

```text
Pincode = 452001
```

Pandas says:

```text
numeric
```

but semantically:

```text
Pincode
```

may be categorical/identifier-like.

Similarly:

```text
EducationLevel = 1, 2, 3, 4
```

dtype numeric hai but semantically ordinal.

Therefore:

> Data type alone feature meaning nahi batata.

For learning and important projects, explicit feature lists often safer hain.

---

# 32. Why ColumnTransformer is better than manual preprocessing?

Without ColumnTransformer:

```python
X_num = X[
    ["Age", "Salary"]
]

X_cat = X[
    ["City"]
]

X_num = imputer.fit_transform(X_num)
X_num = scaler.fit_transform(X_num)

X_cat = cat_imputer.fit_transform(X_cat)
X_cat = encoder.fit_transform(X_cat)

# then manually combine...
```

Problems:

```text
more code
easy leakage
column order mistakes
train/test inconsistency
harder deployment
```

With ColumnTransformer:

```python
X_new = preprocessor.fit_transform(X)
```

Much cleaner.

---

# 33. Why it reduces mistakes

ColumnTransformer remembers:

```text
which transformer
↓
which columns
↓
what learned parameters
```

So later:

```python
preprocessor.transform(new_data)
```

same logic apply karta hai.

This is very important for production.

---

# 34. New production data

Suppose model training preprocessing:

```python
preprocessor.fit(X_train)
```

New customer:

```python
new_customer = pd.DataFrame({
    "Age": [30],
    "Salary": [55000],
    "City": ["Indore"]
})
```

Use:

```python
new_customer_ready = preprocessor.transform(
    new_customer
)
```

Not:

```python
preprocessor.fit_transform(
    new_customer
)
```

Same training:

```text
medians
means/std
categories
```

must be reused.

---

# 35. Unknown category handling

Suppose training:

```text
City:

Indore
Dewas
Bhopal
```

Production:

```text
Ujjain
```

With:

```python
OneHotEncoder(
    handle_unknown="ignore"
)
```

ColumnTransformer safely processes new category without changing training feature structure.

Official mixed-type examples also use `handle_unknown="ignore"` in categorical preprocessing. :chatgpt-content-reference{index="9"}

---

# 36. Sparse vs dense output

OneHotEncoder often sparse output generate kar sakta hai.

ColumnTransformer multiple transformer outputs ko combine karta hai. Current API me `sparse_threshold=0.3` default hai; if combined sparse output ki density threshold se lower ho, final combined result sparse ho sakta hai. :chatgpt-content-reference{index="10"}

Example:

```python
preprocessor = ColumnTransformer(
    transformers=[...],
    sparse_threshold=0
)
```

can force dense combination when appropriate.

Beginner stage par isko memorize karna necessary nahi.

Bas yaad rakho:

```text
ColumnTransformer output
→ NumPy array bhi ho sakta hai
→ sparse matrix bhi
→ set_output se DataFrame bhi
```

---

# 37. ColumnTransformer + model

Abhi manually:

```python
X_train_ready = preprocessor.fit_transform(
    X_train
)

X_test_ready = preprocessor.transform(
    X_test
)

model.fit(
    X_train_ready,
    y_train
)

y_pred = model.predict(
    X_test_ready
)
```

Ye correct hai.

But next chapter me hum isko aur powerful banayenge:

```python
full_pipeline.fit(
    X_train,
    y_train
)

full_pipeline.predict(
    X_test
)
```

where:

```text
ColumnTransformer
+
Model
```

one Pipeline ke andar honge.

---

# 38. Complete real-world example

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.linear_model import LogisticRegression
```

Dataset:

```python
df = pd.DataFrame({
    "Age": [
        22, 30, np.nan, 40,
        50, 35, 45, 28
    ],
    "Salary": [
        25000, 40000, 50000, np.nan,
        90000, 55000, 80000, 35000
    ],
    "City": [
        "Indore",
        "Dewas",
        "Bhopal",
        None,
        "Indore",
        "Dewas",
        "Bhopal",
        "Indore"
    ],
    "Purchased": [
        0, 0, 0, 1,
        1, 0, 1, 0
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

Columns:

```python
numeric_features = [
    "Age",
    "Salary"
]

categorical_features = [
    "City"
]
```

Numerical:

```python
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)
```

Categorical:

```python
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)
```

ColumnTransformer:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)
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

Transform:

```python
X_train_ready = preprocessor.fit_transform(
    X_train
)

X_test_ready = preprocessor.transform(
    X_test
)
```

Train:

```python
model = LogisticRegression()

model.fit(
    X_train_ready,
    y_train
)
```

Predict:

```python
y_pred = model.predict(
    X_test_ready
)
```

---

# 39. Full workflow

```text
Raw DataFrame
      ↓
Create X/y
      ↓
Train-Test Split
      ↓
             ColumnTransformer
                     ↓
       ┌─────────────┴──────────────┐
       ↓                            ↓
Numeric columns               Categorical columns
       ↓                            ↓
SimpleImputer                 SimpleImputer
       ↓                            ↓
StandardScaler                OneHotEncoder
       ↓                            ↓
       └─────────────┬──────────────┘
                     ↓
              Combined Features
                     ↓
                  Model
                     ↓
                 Predict
```

---

# 40. Common mistake #1 — Same transformation all columns

Wrong:

```python
scaler.fit_transform(
    X
)
```

when X contains:

```text
Age
Salary
City
```

City cannot be treated like regular numeric data.

Correct:

```text
Different columns
→ different preprocessing
```

Use ColumnTransformer.

---

# 41. Common mistake #2 — Categorical imputer and numeric imputer same

Don't blindly:

```python
SimpleImputer(
    strategy="mean"
)
```

on:

```text
Age
City
```

Use:

```text
Numeric
→ median/mean

Categorical
→ most_frequent/constant
```

---

# 42. Common mistake #3 — Fit before split

Wrong:

```python
X_ready = preprocessor.fit_transform(X)

train_test_split(
    X_ready,
    y
)
```

Correct:

```text
Split first
↓
fit preprocessing on train
↓
transform test
```

---

# 43. Common mistake #4 — Forget `remainder`

Suppose:

```text
Age
Salary
City
Experience
```

and only first 3 listed.

By default:

```text
Experience
→ DROP
```

If you want to keep it:

```python
remainder="passthrough"
```

---

# 44. Common mistake #5 — Feature names assume karna

After OneHotEncoder:

```text
City
```

may become:

```text
City_Bhopal
City_Dewas
City_Indore
```

So original column list transformed output ko describe nahi karegi.

Use:

```python
preprocessor.get_feature_names_out()
```

---

# 45. Common mistake #6 — Automatic dtype selection blindly

A column numerical dtype ho sakti hai but actually category:

```text
Pincode
CategoryCode
CustomerID
```

Always semantic meaning understand karo.

---

# 46. `ColumnTransformer` ka biggest advantage

Before:

```text
Preprocessing scattered everywhere
```

After:

```text
One preprocessing object
```

which knows:

```text
which columns
which imputer
which scaler
which encoder
which learned values
```

This makes:

```text
training
testing
cross-validation
hyperparameter tuning
deployment
```

much safer.

---

# 47. Chapter 13 mental model

Isko strongly yaad rakho:

```text
ColumnTransformer
=
Different columns
+
Different transformations
+
Combine outputs
```

Example:

```text
Age, Salary
      ↓
Imputer
      ↓
Scaler
      ↓
      ┐
      │
      ├──→ Combined ML Features
      │
City  ↓
Imputer
 ↓
OneHotEncoder
```

---

# Chapter 13 Summary

Import:

```python
from sklearn.compose import ColumnTransformer
```

Basic:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)
```

Training:

```python
X_train_new = preprocessor.fit_transform(
    X_train
)
```

Testing:

```python
X_test_new = preprocessor.transform(
    X_test
)
```

Important parameters:

```text
transformers
→ which transformer on which columns

remainder="drop"
→ unspecified columns remove

remainder="passthrough"
→ unspecified columns keep

verbose_feature_names_out
→ generated feature naming
```

Useful methods/attributes:

```python
preprocessor.get_feature_names_out()

preprocessor.named_transformers_
```

And the most important real-world architecture:

```text
Numerical
→ Imputer
→ Scaler

Categorical
→ Imputer
→ Encoder

        ↓

ColumnTransformer

        ↓

Model-ready features
```

### Quick practice

Suppose dataset:

```text
Age
Salary
City
Education
CustomerID
Purchased
```

And:

```text
Purchased = target
```

Think:

1. `Age` aur `Salary` par kaunsi preprocessing branch banaoge?
2. `City` nominal hai — kaunsa encoder?
3. `Education = School/Bachelor/Master/PhD` — kaunsa encoder?
4. `CustomerID` ko model me nahi chahiye — ColumnTransformer me kaise drop kar sakte ho?
5. Agar `Experience` transformer list me nahi hai aur `remainder="drop"` hai, kya hoga?
6. `preprocessor.fit_transform(X_test)` kyon wrong hai?
7. OneHotEncoder ke baad actual output feature names dekhne ke liye kaunsa method use karoge?
