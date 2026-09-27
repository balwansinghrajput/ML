# Chapter 14 — `Pipeline` & `make_pipeline()` Deep Dive

Ab hum Scikit-learn ke sabse powerful practical concepts me se ek par aa gaye hain:

```text
Pipeline
```

Ab tak hum manually ye sab kar rahe the:

```text
Missing values fill
↓
Encoding
↓
Scaling
↓
Model training
↓
Prediction
```

Pipeline in sab steps ko **ek single object** me combine kar deta hai.

Scikit-learn ki current stable docs ke according `Pipeline` sequential transformers ko apply karta hai aur optionally end me predictor/model laga sakta hai. Intermediate steps ko `fit` + `transform` support karna chahiye, aur final step predictor ho sakta hai. :chatgpt-content-reference{index="0"}

---

# 1. Pipeline kya hota hai?

Simple definition:

> **Pipeline multiple preprocessing steps aur model ko ek fixed sequence me connect karta hai.**

Example:

```text
Raw Data
   ↓
SimpleImputer
   ↓
StandardScaler
   ↓
LogisticRegression
   ↓
Prediction
```

Instead of manually:

```python
X_train = imputer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)

model.fit(X_train, y_train)
```

hum likh sakte hain:

```python
pipeline.fit(X_train, y_train)
```

Bas.

---

# 2. Basic import

```python
from sklearn.pipeline import Pipeline
```

Basic syntax:

```python
pipeline = Pipeline(
    steps=[
        ("step1", transformer1),
        ("step2", transformer2),
        ("model", model)
    ]
)
```

Each step:

```text
(
    step_name,
    estimator
)
```

format me hota hai.

---

# 3. Simple example

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
```

Pipeline:

```python
pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression()
        )
    ]
)
```

Flow:

```text
X
↓
SimpleImputer
↓
StandardScaler
↓
LogisticRegression
```

---

# 4. `pipeline.fit()` actually kya karta hai?

Suppose:

```python
pipeline.fit(
    X_train,
    y_train
)
```

Internally roughly:

```text
Step 1:
SimpleImputer.fit_transform(X_train)

            ↓

Step 2:
StandardScaler.fit_transform(...)

            ↓

Step 3:
LogisticRegression.fit(
    transformed_X,
    y_train
)
```

Pipeline docs ke according fitting ke time transformers sequentially fit + transform hote hain aur final estimator transformed data par fit hota hai. :chatgpt-content-reference{index="1"}

So:

```python
pipeline.fit(X_train, y_train)
```

ek command ke andar pura training workflow chala deta hai.

---

# 5. Prediction me kya hota hai?

Suppose:

```python
y_pred = pipeline.predict(
    X_test
)
```

Pipeline automatically:

```text
X_test
↓
imputer.transform()
↓
scaler.transform()
↓
model.predict()
↓
y_pred
```

karta hai.

Very important:

```text
Test data par imputer/scaler fit nahi hote.
```

Previously learned training statistics hi use hote hain.

Pipeline ka `predict()` intermediate transformers ke `transform()` call karta hai aur final estimator ka `predict()` run karta hai. :chatgpt-content-reference{index="2"}

---

# 6. Manual workflow vs Pipeline

## Without Pipeline

```python
imputer = SimpleImputer(
    strategy="median"
)

scaler = StandardScaler()

model = LogisticRegression()

X_train_clean = imputer.fit_transform(
    X_train
)

X_test_clean = imputer.transform(
    X_test
)

X_train_scaled = scaler.fit_transform(
    X_train_clean
)

X_test_scaled = scaler.transform(
    X_test_clean
)

model.fit(
    X_train_scaled,
    y_train
)

y_pred = model.predict(
    X_test_scaled
)
```

Quite a lot of code.

---

## With Pipeline

```python
pipeline = Pipeline(
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
        ),
        (
            "model",
            LogisticRegression()
        )
    ]
)

pipeline.fit(
    X_train,
    y_train
)

y_pred = pipeline.predict(
    X_test
)
```

Much cleaner.

---

# 7. Pipeline ka sabse bada benefit

Pipeline automatically correct sequence maintain karta hai:

```text
TRAIN

X_train
↓
fit imputer
↓
transform
↓
fit scaler
↓
transform
↓
fit model
```

Then:

```text
TEST

X_test
↓
imputer.transform
↓
scaler.transform
↓
model.predict
```

Isliye data leakage ki common mistakes kam hoti hain.

---

# 8. Intermediate steps kaun ho sakte hain?

Pipeline ke beech ke steps ko transformer hona chahiye.

Matlab normally:

```text
fit()
+
transform()
```

support karein.

Examples:

```text
SimpleImputer
StandardScaler
MinMaxScaler
OneHotEncoder
PCA
PolynomialFeatures
ColumnTransformer
```

Final step predictor/model ho sakta hai:

```text
LogisticRegression
LinearRegression
RandomForestClassifier
SVC
KNeighborsClassifier
```

Current `Pipeline` API me all non-last steps ko `transform` support karna chahiye; final estimator ko at least `fit` support karna hota hai. :chatgpt-content-reference{index="3"}

---

# 9. Final step transformer bhi ho sakta hai?

Yes.

Example:

```python
pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer()
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)
```

Then:

```python
X_new = pipeline.fit_transform(X)
```

Valid hai.

Because final step bhi transformer hai.

Lekin agar final step:

```python
LogisticRegression()
```

hai, then use:

```python
pipeline.predict(...)
```

karoge.

---

# 10. `fit_transform()` on Pipeline

If final step transformer behavior support karta hai:

```python
X_new = pipeline.fit_transform(
    X_train
)
```

Pipeline sequentially transformations apply karega.

But model pipeline:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

ke liye generally:

```python
pipeline.fit(...)
pipeline.predict(...)
```

use karoge.

---

# 11. `ColumnTransformer` + Pipeline

Chapter 13 me humne:

```text
Numerical
→ imputer
→ scaler

Categorical
→ imputer
→ encoder
```

banaya tha.

Ab full real-world workflow:

```text
Raw X
↓
ColumnTransformer
↓
preprocessed X
↓
Model
```

Isko Pipeline me combine karte hain.

---

# 12. Example preprocessor

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
            ["Age", "Salary"]
        ),
        (
            "cat",
            categorical_transformer,
            ["City"]
        )
    ]
)
```

---

# 13. Full Pipeline

```python
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression()
        )
    ]
)
```

Now training:

```python
model.fit(
    X_train,
    y_train
)
```

Prediction:

```python
y_pred = model.predict(
    X_test
)
```

That's all.

---

# 14. Full architecture

```text
                        RAW DATA
                           ↓
                    ColumnTransformer
                           ↓

            ┌──────────────┴──────────────┐
            ↓                             ↓
       Numerical                     Categorical
            ↓                             ↓
     SimpleImputer                  SimpleImputer
            ↓                             ↓
     StandardScaler                OneHotEncoder
            ↓                             ↓
            └──────────────┬──────────────┘
                           ↓
                    Combined Features
                           ↓
                  LogisticRegression
                           ↓
                       Prediction
```

And all of this:

```python
model.fit(X_train, y_train)
```

se train ho sakta hai.

---

# 15. Why this is production-friendly

Without Pipeline, API me manually yaad rakhna padega:

```text
First impute
Then encode
Then scale
Then model
```

Pipeline:

```python
model.predict(new_customer)
```

itself correct preprocessing apply karega.

This drastically reduces training-serving mismatch.

---

# 16. New production data

Suppose:

```python
new_customer = pd.DataFrame({
    "Age": [30],
    "Salary": [55000],
    "City": ["Indore"]
})
```

Prediction:

```python
prediction = model.predict(
    new_customer
)
```

Pipeline internally:

```text
Age/Salary
→ median fill if needed
→ scaling

City
→ missing fill if needed
→ one-hot encode

↓
model prediction
```

You don't manually call each preprocessing object.

---

# 17. Unknown City automatically handle

If Pipeline categorical encoder:

```python
OneHotEncoder(
    handle_unknown="ignore"
)
```

hai, then production:

```text
City = Ratlam
```

aa jaye, pipeline same training feature structure maintain kar sakta hai.

So:

```python
model.predict(new_data)
```

still possible.

---

# 18. `named_steps`

Pipeline steps inspect karne ke liye:

```python
model.named_steps
```

Current Pipeline API `named_steps` provide karti hai to access steps by their assigned names. :chatgpt-content-reference{index="4"}

Suppose:

```python
model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression()
        )
    ]
)
```

Access scaler:

```python
model.named_steps[
    "scaler"
]
```

Access classifier:

```python
model.named_steps[
    "classifier"
]
```

---

# 19. Learned parameters inspect karna

After:

```python
model.fit(
    X_train,
    y_train
)
```

Scaler mean:

```python
model.named_steps[
    "scaler"
].mean_
```

Classifier coefficients:

```python
model.named_steps[
    "classifier"
].coef_
```

So nested objects still inspect kiye ja sakte hain.

---

# 20. ColumnTransformer inside Pipeline access

Suppose:

```python
full_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression()
        )
    ]
)
```

Preprocessor:

```python
full_pipeline.named_steps[
    "preprocessor"
]
```

Model:

```python
full_pipeline.named_steps[
    "model"
]
```

Then numerical branch:

```python
full_pipeline.named_steps[
    "preprocessor"
].named_transformers_[
    "num"
]
```

This nesting real projects me common hai.

---

# 21. Deep nested learned value

Suppose numerical pipeline:

```text
imputer
↓
scaler
```

Then fitted scaler:

```python
num_pipeline = (
    full_pipeline
    .named_steps["preprocessor"]
    .named_transformers_["num"]
)
```

Then:

```python
num_pipeline.named_steps[
    "scaler"
].mean_
```

You can inspect exact training mean.

---

# 22. `steps`

Pipeline:

```python
pipeline.steps
```

returns list of step tuples.

Conceptually:

```text
[
  ("imputer", SimpleImputer(...)),
  ("scaler", StandardScaler()),
  ("model", LogisticRegression())
]
```

Useful for debugging.

---

# 23. Parameter setting in Pipeline

Suppose:

```python
pipeline = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression()
        )
    ]
)
```

How do we change LogisticRegression parameter?

Normal object:

```python
LogisticRegression(
    C=0.5
)
```

Inside Pipeline:

```python
pipeline.set_params(
    model__C=0.5
)
```

Notice:

```text
model__C
```

Double underscore.

Current docs specify nested step parameters with:

```text
step_name__parameter_name
```

syntax. :chatgpt-content-reference{index="5"}

---

# 24. Double underscore `__` rule

Very important.

```text
step__parameter
```

Example:

```python
pipeline.set_params(
    model__C=10
)
```

Meaning:

```text
model step
↓
C parameter
↓
10
```

Another:

```python
pipeline.set_params(
    scaler__with_mean=False
)
```

Meaning:

```text
scaler step
↓
with_mean=False
```

---

# 25. Deep nested parameter

Full:

```text
Pipeline
↓
preprocessor
↓
num
↓
imputer
↓
strategy
```

Parameter syntax:

```text
preprocessor__num__imputer__strategy
```

Example:

```python
full_pipeline.set_params(
    preprocessor__num__imputer__strategy="mean"
)
```

Read from left to right:

```text
preprocessor
→ num branch
→ imputer
→ strategy
```

This becomes extremely important in GridSearchCV.

---

# 26. Another nested example

Categorical OneHotEncoder:

```python
full_pipeline.set_params(
    preprocessor__cat__onehot__drop="first"
)
```

Structure:

```text
preprocessor
↓
cat
↓
onehot
↓
drop
```

Once you understand `__`, hyperparameter tuning becomes much easier.

---

# 27. `get_params()`

See all available pipeline parameters:

```python
pipeline.get_params()
```

For nested ones:

```python
pipeline.get_params(
    deep=True
)
```

Current API's `get_params(deep=True)` includes parameters of contained estimators. :chatgpt-content-reference{index="6"}

You may see:

```text
scaler__copy
scaler__with_mean
model__C
model__max_iter
...
```

---

# 28. GridSearch preview

Later hyperparameter tuning me:

```python
param_grid = {
    "model__C": [
        0.1,
        1,
        10
    ]
}
```

GridSearchCV Pipeline ke andar model ka `C` tune kar sakta hai.

Even preprocessing:

```python
param_grid = {
    "preprocessor__num__imputer__strategy": [
        "mean",
        "median"
    ],
    "model__C": [
        0.1,
        1,
        10
    ]
}
```

So Scikit-learn simultaneously decide kar sakta hai:

```text
mean vs median
+
different model settings
```

through validation.

---

# 29. Pipeline + Cross Validation

Pipeline ka major benefit:

```python
cross_val_score(
    pipeline,
    X,
    y
)
```

ke andar each fold me preprocessing correctly fit hoti hai.

Conceptually:

```text
Fold 1:

Training fold
→ fit imputer
→ fit scaler
→ fit model

Validation fold
→ transform only
→ predict
```

Then next fold me everything fresh fit hota hai.

Isi wajah se Pipeline cross-validation ke saath leakage avoid karne me very important hai. Pipeline specifically steps ko together cross-validate aur tune karne ke purpose se designed hai. :chatgpt-content-reference{index="7"}

---

# 30. Wrong cross-validation without Pipeline

Suppose:

```python
X_scaled = scaler.fit_transform(X)

cross_val_score(
    model,
    X_scaled,
    y
)
```

Scaler cross-validation se pehle **whole dataset** dekh chuka.

Validation folds ki information preprocessing me leak ho gayi.

Better:

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

Then:

```python
cross_val_score(
    pipeline,
    X,
    y
)
```

Each fold preprocessing correctly handle karega.

---

# 31. `Pipeline` vs `make_pipeline()`

Ab second major topic.

Import:

```python
from sklearn.pipeline import make_pipeline
```

`make_pipeline()` Pipeline banane ka shortcut hai. Isme tum manually step names nahi dete; names estimator class names se automatically generate hote hain. :chatgpt-content-reference{index="8"}

---

# 32. Using `Pipeline`

```python
pipeline = Pipeline(
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
        ),
        (
            "model",
            LogisticRegression()
        )
    ]
)
```

Tumne names choose kiye:

```text
imputer
scaler
model
```

---

# 33. Same using `make_pipeline`

```python
pipeline = make_pipeline(
    SimpleImputer(
        strategy="median"
    ),
    StandardScaler(),
    LogisticRegression()
)
```

No manual names.

Sklearn automatically roughly names:

```text
simpleimputer
standardscaler
logisticregression
```

`make_pipeline()` specifically shorthand hai jo estimators ko lowercase class-name based names assign karta hai. :chatgpt-content-reference{index="9"}

---

# 34. Pipeline vs make_pipeline difference

| Feature | `Pipeline()` | `make_pipeline()` |
|---|---|---|
| Manual names | ✅ | ❌ |
| Automatic names | ❌ | ✅ |
| Functionality | Same core pipeline behavior | Same core pipeline behavior |
| More control | ✅ | Less |
| Shorter syntax | Longer | ✅ |

So:

```text
Pipeline()
→ control

make_pipeline()
→ convenience
```

---

# 35. `make_pipeline` named steps

Example:

```python
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)
```

Then:

```python
pipeline.named_steps
```

could have:

```text
standardscaler
logisticregression
```

To set `C`:

```python
pipeline.set_params(
    logisticregression__C=0.1
)
```

Because auto-generated step name is used.

---

# 36. Why I recommend `Pipeline()` while learning

Because:

```python
Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ]
)
```

makes structure clear.

And later parameters:

```text
model__C
scaler__with_mean
```

easy to remember.

With:

```python
make_pipeline(...)
```

auto-generated names can get longer.

So for serious real projects:

```text
Pipeline()
```

often gives clearer explicit control.

For quick experiments:

```text
make_pipeline()
```

is convenient.

---

# 37. Can a Pipeline step be removed?

Current Pipeline supports replacing a transformer with:

```text
"passthrough"
```

or `None`. :chatgpt-content-reference{index="10"}

Example:

```python
pipeline.set_params(
    scaler="passthrough"
)
```

Now:

```text
imputer
↓
scaler skipped
↓
model
```

Useful when testing:

```text
with scaling
vs
without scaling
```

---

# 38. Example `passthrough`

```python
pipeline = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression()
        )
    ]
)
```

Disable scaler:

```python
pipeline.set_params(
    scaler="passthrough"
)
```

Then training essentially:

```text
raw input
↓
LogisticRegression
```

This can also be included in hyperparameter search.

---

# 39. `memory` parameter

Pipeline can cache fitted transformer results:

```python
Pipeline(
    steps=[...],
    memory="cache_folder"
)
```

This can help when preprocessing is computationally expensive and repeated many times during GridSearch.

Current docs note that `memory` caches fitted transformers; final step is not cached. :chatgpt-content-reference{index="11"}

Beginner stage me usually:

```text
memory=None
```

fine hai.

---

# 40. `verbose=True`

```python
pipeline = Pipeline(
    steps=[...],
    verbose=True
)
```

Then fitting ke time step timings print ho sakte hain.

Useful for debugging large pipelines.

Current Pipeline API supports `verbose` for elapsed time per fitting step. :chatgpt-content-reference{index="12"}

---

# 41. `predict_proba()` through Pipeline

Suppose final classifier supports:

```python
predict_proba()
```

Example LogisticRegression.

Then:

```python
probabilities = pipeline.predict_proba(
    X_test
)
```

Pipeline:

```text
X_test
↓
preprocessing transforms
↓
final model.predict_proba()
```

automatically karega. Current Pipeline forwards transformed data to the final estimator's `predict_proba()` when available. :chatgpt-content-reference{index="13"}

---

# 42. `decision_function()`

If final model supports:

```python
decision_function()
```

then:

```python
scores = pipeline.decision_function(
    X_test
)
```

also work kar sakta hai.

Again transformations automatically applied.

---

# 43. `score()`

You can:

```python
pipeline.score(
    X_test,
    y_test
)
```

Pipeline first transforms X, then final estimator ka score run karta hai.

But later hum metrics ke chapter me preferred explicit evaluation bhi dekhenge.

---

# 44. Saving Pipeline

Another huge benefit.

Without Pipeline, save karna pade:

```text
imputer
scaler
encoder
model
```

multiple objects.

Pipeline me:

```text
everything
↓
one object
```

Then later:

```python
joblib.dump(
    pipeline,
    "model.pkl"
)
```

and:

```python
pipeline = joblib.load(
    "model.pkl"
)
```

Then:

```python
pipeline.predict(new_data)
```

directly.

Deployment ke liye ye extremely useful hai.

---

# 45. Complete real-world example

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

---

# 46. Numerical branch

```python
numeric_features = [
    "Age",
    "Salary"
]

numeric_pipeline = Pipeline(
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

---

# 47. Categorical branch

```python
categorical_features = [
    "City"
]

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)
```

---

# 48. ColumnTransformer

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)
```

---

# 49. Final Pipeline

```python
pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression()
        )
    ]
)
```

---

# 50. Split + train

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

Then simply:

```python
pipeline.fit(
    X_train,
    y_train
)
```

No manual transform.

---

# 51. Predict

```python
y_pred = pipeline.predict(
    X_test
)
```

That's a complete real-world sklearn architecture.

---

# 52. New customer

```python
new_customer = pd.DataFrame({
    "Age": [33],
    "Salary": [60000],
    "City": ["Ujjain"]
})
```

Prediction:

```python
prediction = pipeline.predict(
    new_customer
)
```

Pipeline itself:

```text
Age/Salary missing?
↓
impute
↓
scale

City
↓
impute
↓
encode
↓
model prediction
```

handles everything.

---

# 53. Common mistake #1 — Preprocess manually before Pipeline

Wrong idea:

```python
X_scaled = scaler.fit_transform(X)

pipeline.fit(
    X_scaled,
    y
)
```

when Pipeline already contains scaler.

You'd scale twice.

If preprocessing is inside Pipeline:

```text
give raw X
```

to the pipeline.

---

# 54. Common mistake #2 — Fit preprocessing separately

If full pipeline already contains:

```text
preprocessor
+
model
```

don't do:

```python
preprocessor.fit(X_train)

pipeline.fit(X_train, y_train)
```

unnecessarily.

Just:

```python
pipeline.fit(
    X_train,
    y_train
)
```

Pipeline handles it.

---

# 55. Common mistake #3 — Call transformed data on predict

Suppose Pipeline includes scaler.

Wrong:

```python
X_test_scaled = scaler.transform(
    X_test
)

pipeline.predict(
    X_test_scaled
)
```

Pipeline will try to scale again.

Correct:

```python
pipeline.predict(
    X_test
)
```

raw data in.

---

# 56. Common mistake #4 — Model not last

Wrong:

```python
Pipeline([
    ("model", LogisticRegression()),
    ("scaler", StandardScaler())
])
```

LogisticRegression doesn't generally transform X for the next step.

Correct:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

Transformers first.

Predictor last.

---

# 57. Common mistake #5 — Duplicate step names

Avoid:

```python
Pipeline([
    ("step", StandardScaler()),
    ("step", LogisticRegression())
])
```

Step names should be unique because:

```text
named_steps
parameter tuning
```

depend on them.

---

# 58. Common mistake #6 — Forget double underscore

Wrong:

```python
pipeline.set_params(
    model_C=10
)
```

Correct:

```python
pipeline.set_params(
    model__C=10
)
```

Remember:

```text
step__parameter
```

---

# 59. Pipeline vs ColumnTransformer

Don't confuse.

### Pipeline

Sequential:

```text
Step 1
↓
Step 2
↓
Step 3
```

Example:

```text
Imputer
↓
Scaler
↓
Model
```

### ColumnTransformer

Parallel branches on different columns:

```text
            Dataset
            ↓
     ┌──────┴──────┐
     ↓             ↓
Numeric        Categorical
     ↓             ↓
Scaler         Encoder
```

Then combine.

Real world often:

```text
Pipeline
contains
ColumnTransformer
which contains
smaller Pipelines
```

---

# 60. The architecture hierarchy

This is worth memorizing:

```text
FINAL PIPELINE
│
├── PREPROCESSOR (ColumnTransformer)
│
│   ├── NUMERICAL PIPELINE
│   │   ├── Imputer
│   │   └── Scaler
│   │
│   └── CATEGORICAL PIPELINE
│       ├── Imputer
│       └── OneHotEncoder
│
└── MODEL
    └── LogisticRegression
```

This pattern is one of the most useful sklearn architectures you can learn.

---

# 61. `Pipeline()` vs `make_pipeline()` final example

Explicit:

```python
pipe1 = Pipeline(
    steps=[
        (
            "scale",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression()
        )
    ]
)
```

Shortcut:

```python
pipe2 = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)
```

Both return Pipeline-like workflows. `make_pipeline` simply auto-generates names. :chatgpt-content-reference{index="14"}

For `pipe1`:

```python
pipe1.named_steps[
    "classifier"
]
```

For `pipe2`:

```python
pipe2.named_steps[
    "logisticregression"
]
```

---

# Chapter 14 Final Summary

Core idea:

```text
Pipeline
=
preprocessing
+
model
+
one reusable workflow
```

Basic:

```python
pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer()
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression()
        )
    ]
)
```

Train:

```python
pipeline.fit(
    X_train,
    y_train
)
```

Predict:

```python
pipeline.predict(
    X_test
)
```

No manual preprocessing needed.

Most important concepts:

```text
Intermediate steps
→ transformers

Final step
→ model/predictor
```

Nested parameters:

```text
step__parameter
```

Example:

```python
model__C
```

Deep:

```python
preprocessor__num__imputer__strategy
```

Inspect:

```python
pipeline.named_steps
```

Quick shortcut:

```python
make_pipeline(
    StandardScaler(),
    LogisticRegression()
)
```

And the most important real-world pattern:

```text
Pipeline
↓
ColumnTransformer
↓
Numerical Pipeline + Categorical Pipeline
↓
Model
```

### Quick practice

Suppose:

```python
pipeline = Pipeline(
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
        ),
        (
            "model",
            LogisticRegression(
                C=1
            )
        )
    ]
)
```

Think:

1. `pipeline.fit(X_train, y_train)` ke time exact sequence kya chalega?
2. `pipeline.predict(X_test)` me scaler `fit()` karega ya sirf `transform()`?
3. LogisticRegression ka `C` `10` karna ho to parameter name kya hoga?
4. `pipeline.named_steps["scaler"]` kya return karega?
5. `make_pipeline()` aur `Pipeline()` ka biggest difference kya hai?
6. Pipeline cross-validation me leakage kam kyon karta hai?
7. Agar preprocessing Pipeline ke andar hai, to `predict()` ko raw X dena hai ya manually transformed X?
