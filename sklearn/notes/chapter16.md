# Chapter 16 — Polynomial Regression + `PolynomialFeatures` Deep Dive

Chapter 15 me humne padha:

```text
Linear Regression
→ straight-line relationship
```

Example:

\[
y = b_0 + b_1x
\]

Lekin real-world data hamesha straight line follow nahi karta.

Kabhi relation curved hota hai:

```text
x increases
→ y slowly increases
→ phir rapidly increases
```

Aise case me **Polynomial Regression** useful ho sakta hai.

---

## 1. Polynomial Regression kya hai?

Simple definition:

> Polynomial Regression ek regression technique hai jisme original features ke polynomial terms jaise \(x^2\), \(x^3\), interaction terms etc. create karke Linear Regression ko non-linear patterns fit karne diya jata hai.

Example linear equation:

\[
y = b_0 + b_1x
\]

Polynomial degree 2:

\[
y = b_0 + b_1x + b_2x^2
\]

Degree 3:

\[
y = b_0 + b_1x + b_2x^2 + b_3x^3
\]

So:

```text
Degree 1
→ straight line

Degree 2
→ curve

Degree 3
→ more flexible curve
```

---

# 2. Polynomial Regression ki zarurat kyon?

Suppose data:

| Experience | Salary |
|---:|---:|
| 1 | 20 |
| 2 | 30 |
| 3 | 50 |
| 4 | 80 |
| 5 | 120 |

Salary constant rate se increase nahi kar rahi.

Difference:

```text
20 → 30 = +10
30 → 50 = +20
50 → 80 = +30
80 → 120 = +40
```

Relationship curved lag raha hai.

Straight line:

```text
Linear Regression
```

may underfit.

Polynomial Regression:

```text
x
x²
```

use karke curve represent kar sakta hai.

---

# 3. Simple mathematical example

Suppose actual relation:

\[
y = x^2
\]

Data:

```text
x   y
1   1
2   4
3   9
4   16
5   25
```

Linear Regression tries:

\[
y = b_0+b_1x
\]

But exact relation:

\[
y=x^2
\]

hai.

If we create:

```text
x²
```

as new feature, then LinearRegression can fit:

\[
y=b_0+b_1x+b_2x^2
\]

and learn approximately:

```text
b0 = 0
b1 = 0
b2 = 1
```

---

# 4. Important confusion: Polynomial Regression still Linear Regression hai?

Yes — coefficients ke respect me model **linear** hi hai.

Equation:

\[
y=b_0+b_1x+b_2x^2
\]

Here:

```text
x²
```

non-linear transformation hai.

But coefficients:

```text
b0
b1
b2
```

linearly combine ho rahe hain.

So architecture:

```text
Original X
↓
PolynomialFeatures
↓
LinearRegression
```

That's why Polynomial Regression often Linear Regression + feature engineering hota hai.

---

# 5. `PolynomialFeatures`

Scikit-learn:

```python
from sklearn.preprocessing import PolynomialFeatures
```

Basic:

```python
poly = PolynomialFeatures(
    degree=2
)
```

Then:

```python
X_poly = poly.fit_transform(X)
```

---

# 6. Degree 2 with one feature

Suppose:

```python
X = [
    [1],
    [2],
    [3]
]
```

Using:

```python
poly = PolynomialFeatures(
    degree=2
)

X_poly = poly.fit_transform(X)
```

Conceptually output:

```text
1   x   x²

1   1   1
1   2   4
1   3   9
```

First column:

```text
1
```

bias/intercept-like feature hai.

---

# 7. `include_bias`

Default:

```python
PolynomialFeatures(
    degree=2,
    include_bias=True
)
```

means constant column:

```text
1
```

include hoti hai.

But LinearRegression already by default:

```python
fit_intercept=True
```

use karta hai.

So practical pipeline me commonly:

```python
PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

use karna cleaner hai.

Then output:

```text
x
x²
```

instead of:

```text
1
x
x²
```

---

# 8. Basic complete example

```python
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([
    1,
    4,
    9,
    16,
    25
])
```

Create polynomial features:

```python
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly.fit_transform(X)
```

Check:

```python
print(X_poly)
```

Conceptually:

```text
[[ 1.  1.]
 [ 2.  4.]
 [ 3.  9.]
 [ 4. 16.]
 [ 5. 25.]]
```

Columns:

```text
x
x²
```

---

# 9. Train model

```python
model = LinearRegression()

model.fit(
    X_poly,
    y
)
```

Prediction:

```python
new_x = np.array([
    [6]
])

new_x_poly = poly.transform(
    new_x
)

prediction = model.predict(
    new_x_poly
)

print(prediction)
```

Expected approximately:

```text
36
```

because:

\[
6^2=36
\]

---

# 10. Important: New data par `fit_transform()` nahi

Training:

```python
X_train_poly = poly.fit_transform(
    X_train
)
```

Test:

```python
X_test_poly = poly.transform(
    X_test
)
```

New data:

```python
X_new_poly = poly.transform(
    X_new
)
```

Same general rule:

```text
TRAIN
→ fit_transform()

TEST/NEW
→ transform()
```

---

# 11. Multiple features me PolynomialFeatures kya karta hai?

Suppose two features:

```text
x1
x2
```

Degree 2:

```python
PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

can create:

```text
x1
x2
x1²
x1*x2
x2²
```

This is extremely important.

PolynomialFeatures sirf squares nahi banata.

It can also create:

```text
interaction terms
```

---

# 12. Interaction term kya hota hai?

Suppose:

```text
House Price
```

depends on:

```text
Area
Bedrooms
```

Maybe effect:

```text
Large area + many bedrooms
```

together alag impact create kare.

Interaction:

\[
Area \times Bedrooms
\]

is captured using:

```text
x1*x2
```

So degree 2:

\[
y =
b_0
+b_1x_1
+b_2x_2
+b_3x_1^2
+b_4x_1x_2
+b_5x_2^2
\]

---

# 13. Two-feature example

Input:

```python
X = np.array([
    [2, 3]
])
```

With:

```python
PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

output conceptually:

```text
x1   x2   x1²   x1*x2   x2²

2    3     4       6       9
```

So one row from 2 features becomes 5 features.

---

# 14. `get_feature_names_out()`

Useful:

```python
poly.get_feature_names_out(
    ["Area", "Bedrooms"]
)
```

Could return names like:

```text
Area
Bedrooms
Area^2
Area Bedrooms
Bedrooms^2
```

This is very useful because polynomial transformations rapidly increase feature count.

---

# 15. Degree ka meaning

### Degree 1

```text
x
```

Essentially ordinary linear regression.

### Degree 2

```text
x
x²
```

### Degree 3

```text
x
x²
x³
```

### Degree 4

```text
x
x²
x³
x⁴
```

Higher degree:

```text
more flexibility
```

But:

```text
more overfitting risk
```

---

# 16. Underfitting vs Overfitting

Suppose real data curved hai.

Degree 1:

```text
too simple
→ underfitting
```

Degree 2:

```text
may capture pattern well
```

Degree 15:

```text
may twist around every training point
→ overfitting
```

Mental picture:

```text
Degree too low
→ too simple

Degree reasonable
→ pattern capture

Degree too high
→ memorization/noise capture
```

---

# 17. Overfitting example

Suppose:

```text
10 training points
```

and degree:

```text
9
```

very flexible polynomial could nearly pass through every point.

Training score:

```text
99.9%
```

Test score:

```text
poor
```

because it learned noise.

So:

```text
higher degree
≠ better model
```

---

# 18. How to choose degree?

Don't choose:

```text
degree=10
```

just because score on training data is high.

Better:

```text
degree=1
degree=2
degree=3
degree=4
```

compare using:

```text
validation
cross-validation
```

Later GridSearchCV:

```python
param_grid = {
    "poly__degree": [1, 2, 3, 4, 5]
}
```

can automatically test degrees.

---

# 19. Pipeline is perfect for Polynomial Regression

Without Pipeline:

```python
poly.fit_transform(X_train)
model.fit(...)
poly.transform(X_test)
model.predict(...)
```

With Pipeline:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    steps=[
        (
            "poly",
            PolynomialFeatures(
                degree=2,
                include_bias=False
            )
        ),
        (
            "model",
            LinearRegression()
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
y_pred = pipeline.predict(
    X_test
)
```

Much cleaner.

---

# 20. What happens inside Pipeline?

Training:

```text
X_train
↓
PolynomialFeatures.fit_transform()
↓
[x, x², ...]
↓
LinearRegression.fit()
```

Prediction:

```text
X_test
↓
PolynomialFeatures.transform()
↓
same polynomial structure
↓
LinearRegression.predict()
```

---

# 21. Complete example with train/test split

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

y = np.array([
    3,
    8,
    15,
    24,
    35,
    48,
    63,
    80,
    99,
    120
])
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

Pipeline:

```python
pipeline = Pipeline(
    steps=[
        (
            "poly",
            PolynomialFeatures(
                degree=2,
                include_bias=False
            )
        ),
        (
            "model",
            LinearRegression()
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
y_pred = pipeline.predict(
    X_test
)
```

---

# 22. Linear vs Polynomial Regression

Suppose true relationship:

\[
y=2x^2+3x+5
\]

Linear Regression tries:

\[
y=b_0+b_1x
\]

Polynomial Degree 2:

\[
y=b_0+b_1x+b_2x^2
\]

So degree 2 has correct functional form available.

---

# 23. Visual intuition

Linear:

```text
y
↑
|        •
|      •
|    •
|  •
|•
+------------→ x
```

Straight line.

Polynomial:

```text
y
↑
|          •
|       •
|     •
|   •
| •
+------------→ x
```

curve.

---

# 24. Does Polynomial Regression mean every relation is polynomial?

No.

Polynomial regression is one way to approximate non-linear relationships.

Some relationships may be better modeled by:

```text
Decision Trees
Random Forest
Gradient Boosting
Splines
Kernel methods
Neural networks
```

So don't force polynomial features everywhere.

---

# 25. Feature explosion

Very important.

Suppose:

```text
10 original features
degree=2
```

PolynomialFeatures creates many:

```text
original terms
squared terms
pairwise interactions
```

With degree 3 even more.

Feature count can grow very quickly.

This causes:

```text
more memory
slower training
overfitting
multicollinearity
```

---

# 26. Why multicollinearity increases?

If features:

```text
x
x²
x³
```

are related, especially if x range limited.

And with multiple original features:

```text
x1
x2
x1*x2
x1²
x2²
```

many transformed features can correlate.

This can make ordinary LinearRegression coefficients unstable.

That's one reason Polynomial Regression is often paired with:

```text
Ridge
```

instead of plain LinearRegression.

---

# 27. Polynomial Regression + Ridge preview

Architecture:

```text
PolynomialFeatures
↓
StandardScaler
↓
Ridge
```

Why?

```text
Polynomial feature expansion
→ many correlated/large-scale features

Ridge
→ coefficient penalty
→ helps control overfitting
```

We'll study Ridge in Chapter 17.

---

# 28. Should polynomial features be scaled?

Often yes, especially when using regularization.

Example original x:

```text
1–100
```

Then:

```text
x²
→ up to 10,000

x³
→ up to 1,000,000
```

Feature scales differ hugely.

Plain LinearRegression can still mathematically fit without mandatory scaling, but numerical conditioning can become worse.

For Ridge/Lasso:

```text
scaling is especially important
```

because penalties depend on coefficient magnitudes.

Common pipeline:

```python
Pipeline(
    steps=[
        (
            "poly",
            PolynomialFeatures(
                degree=2,
                include_bias=False
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            Ridge()
        )
    ]
)
```

---

# 29. `interaction_only=True`

Suppose:

```python
PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)
```

For:

```text
x1
x2
```

instead of:

```text
x1
x2
x1²
x1*x2
x2²
```

you may get:

```text
x1
x2
x1*x2
```

No squared powers.

Useful when you want:

```text
feature interactions
```

but not powers like \(x^2\).

---

# 30. Interaction-only intuition

Suppose sales depends on:

```text
Advertising Budget
Season Score
```

Maybe:

```text
High advertising
+
high season demand
```

together has strong effect.

Term:

\[
Advertising \times Season
\]

captures that interaction.

But maybe:

```text
Advertising²
```

not needed.

Then:

```python
interaction_only=True
```

can be useful.

---

# 31. `include_bias=False` kyon recommend kiya?

`PolynomialFeatures` default bias column:

```text
1
```

create karta hai.

LinearRegression already:

```text
intercept_
```

learn karta hai.

Therefore often:

```python
PolynomialFeatures(
    include_bias=False
)
```

cleaner.

Alternative:

```text
include_bias=True
+
LinearRegression(fit_intercept=False)
```

possible hai, but beginner workflow me unnecessary complexity.

---

# 32. `fit()` PolynomialFeatures me kya learn karta hai?

Interesting question.

PolynomialFeatures dataset mean/std nahi learn karta.

Instead `fit()` mainly determines transformation structure based on:

```text
number of input features
degree
interaction settings
```

Then:

```python
transform()
```

same feature combinations generate karta hai.

You can inspect:

```python
poly.n_features_in_
poly.n_output_features_
poly.powers_
```

---

# 33. `powers_` intuition

For:

```text
x1
x2
```

degree 2, powers might represent:

```text
x1
x2
x1²
x1*x2
x2²
```

Internally exponents like:

```text
[1,0]
[0,1]
[2,0]
[1,1]
[0,2]
```

represent kar sakte hain.

You don't need to memorize this, but useful for advanced inspection.

---

# 34. New value prediction

Suppose pipeline trained:

```python
pipeline.fit(
    X_train,
    y_train
)
```

Then:

```python
prediction = pipeline.predict(
    [[12]]
)
```

Pipeline automatically:

```text
12
↓
12²
↓
model equation
↓
prediction
```

No manual polynomial transformation needed.

---

# 35. Extrapolation danger

Very important.

Suppose training X:

```text
1 to 10
```

and you predict:

```text
100
```

Polynomial degree 3:

\[
x^3
\]

at 10:

```text
1000
```

at 100:

```text
1,000,000
```

So predictions can explode dramatically outside training range.

Polynomial Regression is especially risky for extrapolation.

Rule:

> Polynomial fit can look good inside training range but behave wildly outside it.

---

# 36. Example extrapolation issue

Suppose fitted:

\[
y=2x^3-3x^2+x+10
\]

For:

```text
x=10
```

value manageable.

For:

```text
x=100
```

cubic term dominates:

\[
2(100^3)=2,000,000
\]

Prediction huge ho sakti hai.

So be cautious with high-degree polynomials and future values far outside training range.

---

# 37. PolynomialFeatures categorical data par?

Raw categorical:

```text
City = Indore
```

direct PolynomialFeatures ke liye suitable nahi.

Categorical feature pehle encode hoti hai.

But one-hot encoded categories par blindly polynomial expansion bahut unnecessary interactions create kar sakta hai.

In mixed datasets, polynomial terms often selected numerical features par apply karna better hota hai.

---

# 38. ColumnTransformer + PolynomialFeatures

Suppose:

```text
Age
Experience
City
```

You may want:

```text
Age + Experience
→ polynomial terms

City
→ OneHotEncoder
```

Use ColumnTransformer:

```python
numeric_poly = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "poly",
            PolynomialFeatures(
                degree=2,
                include_bias=False
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
categorical_pipe = Pipeline(
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

Then combine using ColumnTransformer.

---

# 39. Full mixed-data regression pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

numeric_features = [
    "Age",
    "Experience"
]

categorical_features = [
    "City"
]
```

Numeric:

```python
numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "poly",
            PolynomialFeatures(
                degree=2,
                include_bias=False
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
categorical_pipeline = Pipeline(
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

Preprocessor:

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

Model:

```python
pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)
```

Then:

```python
pipeline.fit(
    X_train,
    y_train
)

y_pred = pipeline.predict(
    X_test
)
```

---

# 40. Linear vs Polynomial comparison

| Property | Linear | Polynomial |
|---|---|---|
| Features | Original | Original + powers/interactions |
| Shape | Straight plane/line | Curved surface possible |
| Complexity | Lower | Higher |
| Overfitting risk | Lower | Higher |
| Interpretability | Easier | Harder |
| Extrapolation | Still risky | Often much riskier |
| Feature count | Same | Can explode |

---

# 41. Polynomial Regression is feature engineering

This is a key insight.

Model itself:

```python
LinearRegression()
```

same hi hai.

What changes?

```text
X
```

Before:

```text
x
```

After:

```text
x
x²
x³
```

So:

```text
Polynomial Regression
=
Polynomial Feature Engineering
+
Linear Regression
```

---

# 42. Example with 2 features

Suppose:

```text
Area
Bedrooms
```

Degree 2 produces:

```text
Area
Bedrooms
Area²
Area*Bedrooms
Bedrooms²
```

Then model:

\[
Price =
b_0
+b_1Area
+b_2Bedrooms
+b_3Area^2
+b_4(Area\times Bedrooms)
+b_5Bedrooms^2
\]

This allows much richer relationship.

---

# 43. Coefficient interpretation becomes harder

Simple linear model:

```text
Area coefficient = 3000
```

easy to interpret.

Polynomial:

```text
Area
Area²
Area*Bedrooms
```

all contribute simultaneously.

Now saying:

> Area increase by 1 gives exactly ₹3000 increase

is no longer correct in a simple way.

Effect depends on current:

```text
Area
Bedrooms
```

values.

So polynomial models lose some interpretability.

---

# 44. Common mistake #1 — High degree blindly

Wrong mindset:

```text
degree=10
→ more powerful
→ therefore better
```

Actually:

```text
higher degree
→ higher flexibility
→ higher variance
→ overfitting risk
```

Use validation.

---

# 45. Common mistake #2 — Fit PolynomialFeatures before split unnecessarily

Better workflow:

```text
Split
↓
Pipeline
↓
fit PolynomialFeatures on training
↓
transform validation/test
```

Even though PolynomialFeatures doesn't learn target statistics like StandardScaler, keeping transformations inside Pipeline prevents workflow inconsistencies and makes cross-validation correct.

---

# 46. Common mistake #3 — Scale before polynomial expansion when using regularization without thought

Suppose:

```text
x standardized
↓
x²
```

versus:

```text
x
↓
x²
↓
scale resulting polynomial terms
```

These are different transformations.

A common regularized polynomial pipeline is:

```text
PolynomialFeatures
↓
StandardScaler
↓
Ridge/Lasso
```

because generated terms themselves need comparable scale.

---

# 47. Common mistake #4 — Use polynomial regression only because target is nonlinear-looking

First check:

```text
scatterplots
residual patterns
cross-validation
domain meaning
```

Maybe:

```text
log transform
tree model
gradient boosting
```

works better.

PolynomialFeatures is one option, not universal solution.

---

# 48. Common mistake #5 — Ignore data range

Polynomial extrapolation can become unstable.

If training:

```text
0–10
```

don't confidently predict:

```text
1000
```

without strong domain justification.

---

# 49. Common mistake #6 — One-hot everything then degree 5

Suppose one-hot creates:

```text
100 columns
```

then PolynomialFeatures degree 2/3 can create enormous feature space.

Memory and computation can blow up.

So feature count always inspect.

---

# 50. Inspect output feature count

```python
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly.fit_transform(X_train)

print(
    poly.n_features_in_
)

print(
    poly.n_output_features_
)
```

Example:

```text
Input features = 3
Output features = 9
```

depending on degree/settings.

Very useful.

---

# 51. How degree affects output size

For 1 feature:

```text
degree 1 → x
degree 2 → x, x²
degree 3 → x, x², x³
```

For multiple features, interaction combinations dramatically increase.

So:

```text
10 features + degree 3
```

can generate many dozens/hundreds of features.

---

# 52. Training score vs test score

Suppose:

```text
Degree 1:
Train R² = 0.70
Test R²  = 0.68
```

Degree 2:

```text
Train = 0.90
Test = 0.88
```

Degree 10:

```text
Train = 0.999
Test = 0.20
```

Degree 10 clearly overfit.

So:

```text
highest training score
≠ best model
```

---

# 53. Cross-validation is better

Later:

```python
cross_val_score(
    pipeline,
    X,
    y,
    cv=5
)
```

can compare degree more reliably.

Example:

```python
for degree in [1, 2, 3, 4]:
    ...
```

Eventually GridSearch:

```python
param_grid = {
    "poly__degree": [
        1,
        2,
        3,
        4
    ]
}
```

will automate.

---

# 54. PolynomialFeatures vs manual feature creation

Manual:

```python
df["Experience2"] = (
    df["Experience"] ** 2
)

df["Experience3"] = (
    df["Experience"] ** 3
)
```

Works.

But `PolynomialFeatures`:

```text
automatic
consistent
pipeline-compatible
creates interactions
easy tuning
```

is usually better for systematic sklearn workflows.

---

# 55. Complete compact example

```python
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([
    3,
    7,
    13,
    21,
    31
])

model = Pipeline(
    steps=[
        (
            "poly",
            PolynomialFeatures(
                degree=2,
                include_bias=False
            )
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)

model.fit(X, y)

print(
    model.predict([[6]])
)
```

If data follows:

\[
y=x^2+x+1
\]

approximately, prediction for 6:

\[
36+6+1=43
\]

---

# 56. Chapter 16 mental model

Remember:

```text
Linear Regression

X
↓
LinearRegression
↓
straight relationship
```

Polynomial Regression:

```text
X
↓
PolynomialFeatures
↓
x, x², x³, interactions
↓
LinearRegression
↓
curved relationship possible
```

Most important formula:

Degree 2:

\[
y=b_0+b_1x+b_2x^2
\]

Multiple features:

\[
y=
b_0+
b_1x_1+
b_2x_2+
b_3x_1^2+
b_4x_1x_2+
b_5x_2^2
\]

---

# Chapter 16 Summary

Import:

```python
from sklearn.preprocessing import PolynomialFeatures
```

Basic:

```python
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

Transform:

```python
X_train_poly = poly.fit_transform(
    X_train
)

X_test_poly = poly.transform(
    X_test
)
```

Model:

```python
model = LinearRegression()

model.fit(
    X_train_poly,
    y_train
)
```

Better:

```python
pipeline = Pipeline(
    steps=[
        (
            "poly",
            PolynomialFeatures(
                degree=2,
                include_bias=False
            )
        ),
        (
            "model",
            LinearRegression()
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

Most important concepts:

```text
Polynomial Regression
=
PolynomialFeatures
+
LinearRegression
```

```text
Higher degree
→ more flexibility
→ more overfitting risk
```

```text
Degree 2 with x1,x2
→ x1
→ x2
→ x1²
→ x1*x2
→ x2²
```

And:

```text
Use validation/cross-validation
to choose degree.
```

### Quick practice

Suppose:

```text
x1 = Area
x2 = Bedrooms
```

and:

```python
PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

1. Kaun-kaun se transformed features create honge?
2. `interaction term` kaunsa hoga?
3. `degree=1` ka Polynomial Regression kis model ke equivalent hai?
4. Degree bahut high rakhne par sabse bada risk kya hai?
5. `include_bias=False` kyon useful hai with `LinearRegression()`?
6. New test data par `poly.fit_transform()` ya `poly.transform()`?
7. Polynomial Regression me model LinearRegression hi hota hai to curve kaise learn hota hai?
8. Why can polynomial extrapolation be dangerous?

