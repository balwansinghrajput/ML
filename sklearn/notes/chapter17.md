# Chapter 17 — Ridge Regression Deep Dive (`L2 Regularization`)

Chapter 15 me humne padha:

```text
LinearRegression
→ Ordinary Least Squares
→ Squared error minimize
```

Chapter 16 me:

```text
PolynomialFeatures
→ feature count badh sakta hai
→ model more flexible
→ overfitting risk badh sakta hai
```

Ab question:

> Agar Linear Regression ke coefficients bahut large ho rahe hain aur model training data ko overfit kar raha hai, to hum model ko thoda control kaise karein?

Ek important solution:

```text
Ridge Regression
```

Ridge basically:

```text
Linear Regression
+
L2 Regularization
```

hai.

Current Scikit-learn `Ridge` least-squares loss ke saath L2 penalty minimize karta hai. :chatgpt-content-reference{index="0"}

---

# 1. Regularization kya hoti hai?

Simple definition:

> **Regularization model ke loss function me ek extra penalty add karti hai taaki model unnecessarily large coefficients na learn kare.**

Normal Linear Regression:

```text
Goal:
Prediction error ko minimum karo
```

Ridge:

```text
Goal:
Prediction error ko minimum karo
+
large coefficients ko penalize karo
```

Mental model:

```text
Fit data well
        +
Keep coefficients controlled
        ↓
Better generalization
```

---

# 2. Linear Regression ka loss

Ordinary Linear Regression roughly minimize karta hai:

\[
\sum (y_i-\hat y_i)^2
\]

Meaning:

```text
Actual - Predicted
↓
square
↓
sum
↓
minimum
```

---

# 3. Ridge ka loss function

Ridge objective:

\[
\underbrace{\sum (y_i-\hat y_i)^2}_{Prediction\ Error}
+
\alpha
\underbrace{\sum_{j=1}^{p} w_j^2}_{L2\ Penalty}
\]

Scikit-learn docs ise:

\[
||y-Xw||_2^2 + \alpha ||w||_2^2
\]

ke form me define karti hain. :chatgpt-content-reference{index="1"}

Yahan:

```text
w
→ model coefficients

α / alpha
→ regularization strength
```

---

# 4. L2 Regularization kya hai?

Suppose coefficients:

```text
w1 = 10
w2 = 20
w3 = 30
```

L2 penalty:

\[
10^2 + 20^2 + 30^2
\]

\[
=100+400+900
\]

\[
=1400
\]

So large coefficients:

```text
large square
→ large penalty
```

Model ko incentive milta hai coefficients ko smaller rakhne ka.

---

# 5. Ridge ka main idea

Without Ridge:

```text
Model:
"I only care about minimizing training error."
```

With Ridge:

```text
Model:
"I want low prediction error,
but very large coefficients bhi avoid karunga."
```

Ye usually model ko:

```text
less sensitive
more stable
less high-variance
```

bana sakta hai.

Official sklearn example bhi dikhata hai ki Ridge coefficient shrinkage ke through noisy/sparse situations me OLS ke comparison me variance reduce kar sakta hai. :chatgpt-content-reference{index="2"}

---

# 6. Coefficient shrinkage kya hai?

Suppose LinearRegression learns:

```text
Feature 1 → 120
Feature 2 → -95
Feature 3 → 80
Feature 4 → 0.5
```

Ridge might learn something like:

```text
Feature 1 → 60
Feature 2 → -45
Feature 3 → 35
Feature 4 → 0.3
```

Exact values depend on data and `alpha`.

Process:

```text
Large coefficients
↓
L2 penalty
↓
coefficients shrink toward zero
```

Important:

> Ridge coefficients ko usually **exactly zero** nahi karta.

That's an important difference from Lasso, jo next chapter me padhenge.

---

# 7. `alpha` kya hai?

`alpha` controls regularization strength.

Import:

```python
from sklearn.linear_model import Ridge
```

Example:

```python
model = Ridge(
    alpha=1.0
)
```

Current stable sklearn me default:

```text
alpha = 1.0
```

hai. Larger alpha = stronger regularization. :chatgpt-content-reference{index="3"}

---

# 8. Small alpha

Example:

```python
Ridge(alpha=0.01)
```

Penalty weak hai.

So model behavior:

```text
Ridge
≈
LinearRegression
```

Coefficients ko little shrinkage milegi.

---

# 9. Large alpha

Example:

```python
Ridge(alpha=100)
```

Penalty strong hai.

Model coefficients ko aggressively smaller karne ki koshish karega.

So:

```text
alpha ↑
↓
regularization ↑
↓
coefficients generally shrink more
```

Scikit-learn docs bhi larger alpha ko stronger regularization ke roop me define karti hain. :chatgpt-content-reference{index="4"}

---

# 10. `alpha = 0`

Mathematically:

```text
alpha = 0
```

means L2 penalty disappear.

Then:

\[
Ridge \approx OLS
\]

Scikit-learn docs ke according `alpha=0` objective ko ordinary least squares ke equivalent banata hai, lekin numerical reasons ke liye actual `LinearRegression` use karna recommended hai rather than `Ridge(alpha=0)`. :chatgpt-content-reference{index="5"}

So:

```python
LinearRegression()
```

use karo if no regularization.

---

# 11. Alpha too large ho to?

Suppose:

```python
Ridge(alpha=1000000)
```

Coefficients bahut zyada shrink ho sakte hain.

Then model:

```text
too simple
```

ban sakta hai.

Result:

```text
Underfitting
```

So alpha trade-off hai.

---

# 12. Bias-Variance tradeoff

Important ML concept.

### LinearRegression

Could have:

```text
low bias
higher variance
```

especially noisy/high-dimensional correlated data me.

### Ridge

Regularization adds some bias:

```text
slightly less flexible
```

but can reduce variance:

```text
more stable
```

So:

```text
Regularization
→ small bias increase
→ potentially large variance decrease
```

which can improve test performance.

---

# 13. Easy example

Suppose training data change hone par LinearRegression coefficients:

```text
Run 1:
[100, -90]

Run 2:
[500, -480]

Run 3:
[-200, 220]
```

Predictions similar ho sakti hain because features highly correlated hain, but coefficients unstable.

Ridge might produce:

```text
Run 1:
[20, 18]

Run 2:
[22, 17]

Run 3:
[19, 20]
```

More stable.

---

# 14. Multicollinearity me Ridge useful kyon?

Suppose:

```text
MonthlySalary
AnnualSalary
```

where:

\[
AnnualSalary \approx 12\times MonthlySalary
\]

Both features highly correlated.

Ordinary Linear Regression confused ho sakta hai:

```text
Which feature should get how much coefficient?
```

Small data changes coefficients dramatically change kar sakte hain.

Ridge penalty encourages more controlled coefficient values and improves numerical conditioning/stability. Scikit-learn specifically notes ki Ridge regularization conditioning improve aur estimate variance reduce kar sakti hai. :chatgpt-content-reference{index="6"}

---

# 15. Basic sklearn syntax

```python
from sklearn.linear_model import Ridge

model = Ridge(
    alpha=1.0
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)
```

Exactly LinearRegression jaisa API.

---

# 16. Learned attributes

After:

```python
model.fit(X_train, y_train)
```

you can inspect:

```python
model.coef_
model.intercept_
```

Current Ridge exposes `coef_` as its learned weight vector and `intercept_` as the independent term. :chatgpt-content-reference{index="7"}

---

# 17. LinearRegression vs Ridge code

Linear:

```python
from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(
    X_train,
    y_train
)
```

Ridge:

```python
from sklearn.linear_model import Ridge

ridge = Ridge(
    alpha=1.0
)

ridge.fit(
    X_train,
    y_train
)
```

Then:

```python
print(linear.coef_)
print(ridge.coef_)
```

You may observe:

```text
Ridge coefficient magnitudes
<
OLS coefficient magnitudes
```

depending on data.

---

# 18. Does Ridge change intercept?

Regularization primarily applies to feature coefficients in the Ridge objective.

Intercept is handled separately when:

```python
fit_intercept=True
```

which is the current default. :chatgpt-content-reference{index="8"}

For beginner understanding:

```text
L2 penalty
→ model weights / coefficients
```

focus karo.

---

# 19. Scaling Ridge me extremely important kyon hai?

Suppose features:

```text
Age:
20–60

Salary:
20,000–5,00,000
```

Ridge penalty:

\[
w_1^2+w_2^2
\]

penalizes coefficient magnitude.

But coefficient magnitude depends on feature units.

For example:

```text
Salary in rupees
```

requires very small coefficient.

Salary in lakhs:

```text
1, 2, 3...
```

requires much larger coefficient.

Without scaling:

```text
L2 penalty
```

different features ko unfairly penalize kar sakti hai simply because units differ.

Therefore Ridge ke saath normally:

```text
StandardScaler
↓
Ridge
```

strongly recommended workflow hai.

---

# 20. Correct Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipeline = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=1.0
            )
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

# 21. Why Pipeline especially important here?

Because scaling must be learned from training data.

Wrong:

```python
X_scaled = scaler.fit_transform(X)

train_test_split(
    X_scaled,
    y
)
```

Leakage.

Better:

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])
```

Then:

```python
pipeline.fit(
    X_train,
    y_train
)
```

and:

```python
pipeline.predict(
    X_test
)
```

Pipeline automatically correct training/test behavior maintain karta hai.

---

# 22. Numerical + categorical real dataset

Suppose:

```text
Age
Experience
City
Salary ← target
```

Numerical:

```text
Age, Experience
→ impute
→ StandardScaler
```

Categorical:

```text
City
→ impute
→ OneHotEncoder
```

Then:

```text
Ridge
```

Use Chapter 13 + 14 architecture.

---

# 23. Full real-world preprocessing

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.linear_model import Ridge
```

Numerical:

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

---

# 24. ColumnTransformer

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            ["Age", "Experience"]
        ),
        (
            "cat",
            categorical_pipeline,
            ["City"]
        )
    ]
)
```

Final:

```python
model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "ridge",
            Ridge(
                alpha=1.0
            )
        )
    ]
)
```

Train:

```python
model.fit(
    X_train,
    y_train
)
```

Predict:

```python
y_pred = model.predict(
    X_test
)
```

---

# 25. Ridge + Polynomial Regression

Chapter 16 ka biggest practical connection.

PolynomialFeatures:

```text
x
x²
x³
x1*x2
...
```

many features generate karta hai.

This can cause:

```text
Overfitting
Multicollinearity
Large coefficients
```

So very common architecture:

```text
PolynomialFeatures
↓
StandardScaler
↓
Ridge
```

---

# 26. Polynomial Ridge Pipeline

```python
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

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
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=1.0
            )
        )
    ]
)
```

Train:

```python
model.fit(
    X_train,
    y_train
)
```

This is much safer than:

```text
High-degree polynomial
+
unregularized LinearRegression
```

in many situations.

---

# 27. Why scale after PolynomialFeatures?

Suppose:

```text
x = 1–100
```

Polynomial:

```text
x      → up to 100
x²     → up to 10,000
x³     → up to 1,000,000
```

If Ridge directly applied:

```text
different terms
→ wildly different scales
```

Penalty becomes unit-sensitive.

Better:

```text
PolynomialFeatures
↓
StandardScaler
↓
Ridge
```

Generated polynomial terms get comparable scale before L2 penalty.

---

# 28. Alpha impact visually

Think:

```text
alpha = 0.001
|
| coefficients mostly free
|
alpha = 1
|
| moderate shrinkage
|
alpha = 100
|
| strong shrinkage
|
alpha = 100000
|
| coefficients near zero
| possible underfitting
```

So:

```text
alpha small
→ complexity high

alpha large
→ complexity low
```

---

# 29. Alpha manually guess nahi karna

Don't think:

```text
alpha=1
always best
```

No.

Best alpha dataset dependent hai.

Try values across orders of magnitude:

```python
[
    0.001,
    0.01,
    0.1,
    1,
    10,
    100,
    1000
]
```

Why logarithmic values?

Because regularization strength ka useful range often multiple orders of magnitude span karta hai.

---

# 30. GridSearchCV with Ridge

Later GridSearch detail me padhenge, but preview:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "ridge__alpha": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ]
}
```

If pipeline:

```python
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ]
)
```

Then:

```python
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5
)
```

Train:

```python
grid.fit(
    X_train,
    y_train
)
```

Best:

```python
print(
    grid.best_params_
)
```

Could give:

```text
{'ridge__alpha': 10}
```

---

# 31. Why double underscore?

Chapter 14:

```text
step__parameter
```

Pipeline step:

```text
ridge
```

Parameter:

```text
alpha
```

Therefore:

```text
ridge__alpha
```

---

# 32. `RidgeCV`

Scikit-learn also provides:

```python
from sklearn.linear_model import RidgeCV
```

It is Ridge regression with built-in cross-validation support. :chatgpt-content-reference{index="9"}

Example:

```python
model = RidgeCV(
    alphas=[
        0.1,
        1,
        10,
        100
    ]
)
```

After fit:

```python
model.alpha_
```

gives selected regularization strength.

We will understand cross-validation properly in its dedicated chapter.

---

# 33. LinearRegression vs Ridge

| Property | LinearRegression | Ridge |
|---|---|---|
| Squared-error loss | ✅ | ✅ |
| Regularization | ❌ | ✅ |
| Penalty | None | L2 |
| Shrinks coefficients | ❌ | ✅ |
| Handles multicollinearity more stably | Less | Better |
| Alpha parameter | ❌ | ✅ |
| Scaling important | Less critical | Very important |
| Exact zero coefficients | Not by penalty | Usually no |

---

# 34. Ridge vs Lasso preview

Ridge:

\[
\alpha\sum w_j^2
\]

Lasso:

\[
\alpha\sum |w_j|
\]

Main practical difference:

```text
Ridge
→ coefficients shrink
→ usually remain non-zero
```

```text
Lasso
→ coefficients shrink
→ some can become exactly zero
```

So Lasso can perform feature selection.

Next chapter isi par hoga.

---

# 35. Why Ridge doesn't usually perform feature selection

Suppose coefficients:

```text
Linear:
[100, 20, 0.5, -50]
```

Ridge might become:

```text
[30, 10, 0.2, -15]
```

Not:

```text
[30, 0, 0, -15]
```

typically.

It reduces importance but usually keeps all features.

For automatic sparse coefficients:

```text
Lasso
```

more suitable.

---

# 36. When Ridge useful hai?

Strong candidate when:

```text
Many numerical features
Features correlated hain
Overfitting ho raha hai
Polynomial features hain
Coefficients unstable hain
You want all features retained
```

Especially:

```text
many correlated predictors
```

Ridge ka classic use case hai.

---

# 37. When Ridge unnecessary ho sakta hai?

Suppose:

```text
few features
huge clean dataset
no overfitting
OLS already stable
```

Then LinearRegression enough ho sakta hai.

Regularization use karna:

```text
automatic requirement
```

nahi.

Always validation performance compare karo.

---

# 38. Ridge can't fix everything

Ridge is not magic.

If:

```text
wrong features
severe data leakage
poor data quality
strong non-linearity not modeled
wrong target
```

then:

```text
Ridge(alpha=...)
```

problem solve nahi karega.

Regularization primarily model complexity/coefficient stability control karta hai.

---

# 39. Outliers

Ridge still uses squared-error loss.

So extreme target outliers:

```text
large residual
↓
square
↓
very large loss
```

can strongly affect model.

L2 regularization coefficients ko control karta hai, but Ridge automatically robust-to-outliers regression nahi hai.

Don't confuse:

```text
Ridge
≠ Robust regression
```

---

# 40. Ridge does not remove features

Again:

```text
Ridge
→ shrink
```

not:

```text
Ridge
→ delete features
```

If feature:

```text
very weak
```

its coefficient may get very small, but usually not exactly zero.

---

# 41. Scaling example

Suppose model has:

```text
Age
Salary
```

Without scaling:

```text
Age = 30
Salary = 50000
```

Coefficients might need:

```text
Age coefficient = 1000
Salary coefficient = 0.8
```

Penalty:

\[
1000^2 + 0.8^2
\]

would heavily punish Age coefficient mostly due to units.

After StandardScaler:

```text
Age ≈ standardized
Salary ≈ standardized
```

coefficient magnitudes become much more comparable.

This makes L2 penalty much more meaningful.

---

# 42. Should target `y` also be scaled?

Usually not required for basic Ridge workflows.

Typical:

```text
X
→ StandardScaler

y
→ original target
```

Ridge predicts target in original units.

Example:

```text
Salary
→ ₹ values
```

Prediction still:

```text
₹65,000
```

Target transformation can be useful in some special problems, but that's separate.

---

# 43. `solver`

Current Ridge supports several solvers such as:

```text
auto
svd
cholesky
lsqr
sparse_cg
sag
saga
lbfgs
```

and default:

```python
solver="auto"
```

lets sklearn choose based on the data. :chatgpt-content-reference{index="10"}

Beginner recommendation:

```python
Ridge(
    alpha=1.0
)
```

and leave:

```text
solver="auto"
```

unless you have a specific reason.

---

# 44. Scaling especially with `sag` / `saga`

Current sklearn docs specifically note that `sag` and `saga` converge quickly only when features are approximately on the same scale, and recommend preprocessing with a scaler. :chatgpt-content-reference{index="11"}

But even beyond these solvers, regularized-model interpretation makes feature scaling good practice.

---

# 45. `positive=True`

Current Ridge supports:

```python
Ridge(
    positive=True
)
```

which forces coefficients non-negative; in the current API this configuration uses the supported `lbfgs` solver. :chatgpt-content-reference{index="12"}

Use only when domain genuinely requires:

```text
all coefficients >= 0
```

Don't force positive just because negative coefficient looks surprising.

---

# 46. Complete practical example

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression,
    Ridge
)
from sklearn.metrics import r2_score
```

Dataset:

```python
df = pd.DataFrame({
    "Experience": [1,2,3,4,5,6,7,8,9,10],
    "Projects": [1,2,2,4,5,5,7,8,9,10],
    "Salary": [
        25000,
        30000,
        34000,
        43000,
        49000,
        54000,
        62000,
        68000,
        76000,
        82000
    ]
})
```

Create:

```python
X = df[
    ["Experience", "Projects"]
]

y = df["Salary"]
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

---

# 47. Linear model

```python
linear = LinearRegression()

linear.fit(
    X_train,
    y_train
)

linear_pred = linear.predict(
    X_test
)
```

---

# 48. Ridge model

```python
ridge = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=1.0
            )
        )
    ]
)

ridge.fit(
    X_train,
    y_train
)

ridge_pred = ridge.predict(
    X_test
)
```

Compare:

```python
print(
    r2_score(
        y_test,
        linear_pred
    )
)

print(
    r2_score(
        y_test,
        ridge_pred
    )
)
```

Important:

> Ridge ka score automatically LinearRegression se higher hona guaranteed nahi.

Regularization tab useful hai jab variance/overfitting/stability problem exist karti ho.

---

# 49. Inspect scaled Ridge coefficients

Because Ridge is inside Pipeline:

```python
ridge_model = ridge.named_steps[
    "ridge"
]

print(
    ridge_model.coef_
)
```

Scaler:

```python
ridge.named_steps[
    "scaler"
].mean_
```

Again Chapter 14 ka `named_steps`.

---

# 50. Polynomial + Ridge complete example

```python
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

model = Pipeline(
    steps=[
        (
            "poly",
            PolynomialFeatures(
                degree=3,
                include_bias=False
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=1.0
            )
        )
    ]
)
```

Then:

```python
model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)
```

Here:

```text
PolynomialFeatures
→ flexibility increases

Ridge
→ complexity control
```

Nice combination.

---

# 51. Alpha + polynomial degree together tune kar sakte ho

Example GridSearch preview:

```python
param_grid = {
    "poly__degree": [
        1,
        2,
        3,
        4
    ],
    "ridge__alpha": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}
```

Now model selection simultaneously asks:

```text
How curved/flexible?
+
How much regularization?
```

This is a powerful ML workflow.

---

# 52. Common mistake #1 — Ridge without scaling

Technically Ridge run kar sakta hai.

But if:

```text
Age = 20–60
Salary = 20,000–5,00,000
```

then penalty comparison becomes unit-dependent.

So common best practice:

```text
StandardScaler
↓
Ridge
```

---

# 53. Common mistake #2 — Think larger alpha always better

Wrong:

```text
alpha 1 < alpha 100
therefore 100 better
```

No.

Too much regularization:

```text
coefficients overly shrink
↓
underfit
```

Alpha should be validated.

---

# 54. Common mistake #3 — Use training score to choose alpha

Suppose:

```text
alpha=0.001
Train R² = 0.99
```

and:

```text
alpha=10
Train R² = 0.95
```

Don't automatically choose `0.001`.

Test/CV may be:

```text
alpha=0.001
Test = 0.70

alpha=10
Test = 0.90
```

Regularization often intentionally sacrifices some training fit for better generalization.

---

# 55. Common mistake #4 — Think Ridge sets coefficients zero

No.

Ridge:

```text
shrink toward 0
```

Lasso:

```text
can set exactly 0
```

Very important distinction.

---

# 56. Common mistake #5 — Ridge solves non-linearity automatically

Ridge model itself is still linear in its input features.

If true relation is:

\[
y=x^2
\]

then plain:

```python
Ridge()
```

still only gets:

```text
x
```

unless you provide:

```text
x²
```

via:

```python
PolynomialFeatures
```

So:

```text
Ridge
≠ nonlinear feature generator
```

---

# 57. Common mistake #6 — Apply Ridge to classification target

`Ridge` is a regressor.

For classification sklearn separately provides:

```python
RidgeClassifier
```

Don't use ordinary:

```python
Ridge()
```

as your standard classification model.

---

# 58. Chapter 17 mental model

Remember this:

```text
LinearRegression
↓
Minimize prediction error
```

Ridge:

```text
Minimize prediction error
+
Penalize large coefficients
```

Formula:

\[
Loss =
RSS
+
\alpha\sum w_j^2
\]

Where:

```text
alpha ↓
→ weak regularization
→ more like LinearRegression
```

```text
alpha ↑
→ strong regularization
→ smaller coefficients
```

---

# Chapter 17 Summary

Import:

```python
from sklearn.linear_model import Ridge
```

Basic:

```python
model = Ridge(
    alpha=1.0
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)
```

Current sklearn Ridge minimizes least-squares error plus an L2 penalty, and `alpha` controls regularization strength. :chatgpt-content-reference{index="13"}

Preferred practical pattern:

```python
pipeline = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=1.0
            )
        )
    ]
)
```

Most important:

```text
Ridge
=
Linear Regression
+
L2 Regularization
```

```text
L2
=
sum of squared coefficients
```

```text
alpha small
→ weak penalty

alpha large
→ strong penalty
```

```text
Ridge
→ shrinks coefficients
→ usually doesn't make them exactly zero
```

And:

```text
Multicollinearity
Polynomial Features
High-dimensional/noisy data
Overfitting
↓
Ridge can be very useful
```

### Quick practice

Suppose:

\[
Loss =
RSS
+
\alpha(w_1^2+w_2^2+w_3^2)
\]

Answer mentally:

1. `alpha` kya control karta hai?
2. `alpha` increase karoge to coefficients par generally kya effect hoga?
3. Ridge ka penalty L1 hai ya L2?
4. Ridge coefficients ko normally exactly `0` karta hai?
5. Ridge ke saath feature scaling important kyon hai?
6. `alpha=0` mathematically kis model ke close hai?
7. PolynomialFeatures ke saath Ridge useful kyon ho sakta hai?
8. Agar training performance slightly decrease aur test performance improve ho, kya Ridge useful ho sakta hai?
9. Pipeline step `"ridge"` ke `alpha` ko GridSearch me kaise refer karoge?

```python
"ridge__alpha"
```