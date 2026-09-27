# Chapter 15 — Regression Introduction + `LinearRegression` Deep Dive

Ab hum actual Machine Learning models start kar rahe hain.

Sabse pehla regression model:

```text
Linear Regression
```

Ye ML ka foundational algorithm hai. Agar Linear Regression achhe se samajh aa gaya, to Ridge, Lasso, ElasticNet aur even Logistic Regression ke kuch concepts bhi easy lagenge.

Scikit-learn ka `LinearRegression` **Ordinary Least Squares (OLS)** linear regression implement karta hai: ye coefficients aise choose karta hai ki actual targets aur predicted targets ke beech squared errors ka sum minimum ho. :chatgpt-content-reference{index="0"}

---

## 1. Regression kya hota hai?

Regression tab use hota hai jab target `y` ek **continuous numerical value** ho.

Examples:

```text
House Price
Salary
Temperature
Sales
Weight
Revenue
Demand
```

Example:

```text
Experience → Salary
```

Dataset:

| Experience | Salary |
|---:|---:|
| 1 | 25000 |
| 2 | 30000 |
| 3 | 35000 |
| 4 | 40000 |

Question:

```text
5 years experience wale employee ki salary kitni hogi?
```

Output:

```text
45000
```

jaisa continuous number hai.

Therefore:

```text
Regression Problem
```

---

# 2. Regression vs Classification

Classification:

```text
Purchased?
→ Yes / No

Spam?
→ Spam / Not Spam
```

Output category hai.

Regression:

```text
House Price?
→ ₹42,50,000

Salary?
→ ₹65,000
```

Output numerical continuous value hai.

So:

```text
Classification
→ Class predict

Regression
→ Number predict
```

---

# 3. Linear Regression kya karta hai?

Suppose:

```text
Hours Studied → Marks
```

Data:

```text
Hours    Marks

1        20
2        30
3        40
4        50
```

Linear Regression ek straight line find karne ki koshish karta hai:

\[
y = mx + c
\]

Where:

```text
x = input feature
y = predicted target
m = slope / coefficient
c = intercept
```

For this data:

\[
Marks = 10 \times Hours + 10
\]

So:

```text
Hours = 5
```

Then:

\[
Marks = 10(5)+10
\]

\[
=60
\]

Prediction:

```text
60
```

---

# 4. `y = mx + c` ko deeply samjho

Equation:

\[
y = mx+c
\]

Suppose:

\[
Salary = 5000 \times Experience + 20000
\]

Here:

```text
m = 5000
c = 20000
```

If:

```text
Experience = 0
```

then:

\[
Salary=20000
\]

If:

```text
Experience = 1
```

then:

\[
Salary=25000
\]

If:

```text
Experience = 2
```

then:

\[
Salary=30000
\]

---

# 5. Slope / Coefficient kya batata hai?

Suppose:

\[
Salary = 5000 \times Experience + 20000
\]

Coefficient:

```text
5000
```

Interpretation:

> Experience me 1 unit increase hone par predicted Salary approximately ₹5000 increase hoti hai, assuming model relationship linear hai.

So:

```text
Coefficient positive
→ x increase
→ predicted y increase
```

Example:

```text
m = +10
```

positive relationship.

---

# 6. Negative coefficient

Suppose:

\[
Price = -2000 \times Age + 500000
\]

Coefficient:

```text
-2000
```

Interpretation:

> Age me 1 unit increase ke saath predicted price 2000 decrease hoti hai.

So:

```text
Positive coefficient
→ positive relationship

Negative coefficient
→ negative relationship
```

---

# 7. Intercept kya hota hai?

Equation:

\[
y=mx+c
\]

`c` = intercept.

Example:

\[
Salary=5000x+20000
\]

Intercept:

```text
20000
```

Mathematically:

> Jab x = 0 ho, predicted y kya hai?

But practical interpretation me careful rehna.

Suppose:

```text
House price vs area
```

Area `0 sq ft` real-world meaningful house nahi hai.

To intercept mathematically useful ho sakta hai, but har problem me real-world interpretation meaningful nahi hoti.

---

# 8. Scikit-learn import

```python
from sklearn.linear_model import LinearRegression
```

Create model:

```python
model = LinearRegression()
```

Current stable sklearn `LinearRegression` OLS model hai; by default `fit_intercept=True`, so model intercept estimate karta hai. :chatgpt-content-reference{index="1"}

---

# 9. Basic training

```python
model.fit(X_train, y_train)
```

Chapter 2 yaad karo:

```text
fit()
→ learn from data
```

LinearRegression specifically learns:

```text
coefficient(s)
+
intercept
```

After fit:

```python
print(model.coef_)
print(model.intercept_)
```

Current sklearn exposes learned coefficients through `coef_` and independent/intercept term through `intercept_`. :chatgpt-content-reference{index="2"}

---

# 10. Simple complete example

```python
import numpy as np

from sklearn.linear_model import LinearRegression

X = np.array([
    [1],
    [2],
    [3],
    [4]
])

y = np.array([
    20,
    30,
    40,
    50
])

model = LinearRegression()

model.fit(X, y)
```

Check:

```python
print(model.coef_)
print(model.intercept_)
```

Approximately:

```text
coef_ = [10.]
intercept_ = 10
```

So model learned:

\[
y=10x+10
\]

---

# 11. Prediction

```python
prediction = model.predict(
    [[5]]
)

print(prediction)
```

Output approximately:

```text
[60.]
```

Because:

\[
10(5)+10=60
\]

---

# 12. Why `[[5]]`, not `[5]`?

Remember:

```text
X
→ 2D
→ samples × features
```

Correct:

```python
model.predict([[5]])
```

Meaning:

```text
1 sample
1 feature
```

Shape:

```text
(1, 1)
```

Wrong:

```python
model.predict([5])
```

because that's 1D.

---

# 13. `coef_` array kyon hota hai?

Even one feature hone par:

```python
model.coef_
```

returns:

```text
[10.]
```

because sklearn model multiple features support karta hai.

If 3 features:

```text
Age
Salary
Experience
```

then:

```text
coef_
```

could be:

```text
[2.5, 0.4, 8.2]
```

Current docs ke according single-target regression me `coef_` ka length number of features ke equal hota hai. :chatgpt-content-reference{index="3"}

---

# 14. Simple Linear Regression

Jab only **one input feature** ho:

```text
Experience
↓
Salary
```

Equation:

\[
y=b_0+b_1x
\]

Where:

```text
b0 = intercept
b1 = coefficient
```

This is:

```text
Simple Linear Regression
```

---

# 15. Multiple Linear Regression

Real datasets me usually multiple features hote hain.

Example:

```text
House Price
```

predict using:

```text
Area
Bedrooms
Age
DistanceToCity
```

Equation:

\[
y =
b_0
+
b_1x_1
+
b_2x_2
+
b_3x_3
+
b_4x_4
\]

Example:

\[
Price =
500000
+
3000(Area)
+
200000(Bedrooms)
-
15000(Age)
\]

This is:

```text
Multiple Linear Regression
```

---

# 16. Multiple Linear Regression code

```python
import pandas as pd

df = pd.DataFrame({
    "Area": [1000, 1200, 1500, 1800, 2000],
    "Bedrooms": [2, 2, 3, 3, 4],
    "Age": [10, 8, 5, 4, 2],
    "Price": [
        3000000,
        3500000,
        4500000,
        5200000,
        6200000
    ]
})
```

Create X/y:

```python
X = df[
    ["Area", "Bedrooms", "Age"]
]

y = df["Price"]
```

Train:

```python
model = LinearRegression()

model.fit(X, y)
```

Check:

```python
print(model.coef_)
print(model.intercept_)
```

Now:

```text
coef_[0]
→ Area coefficient

coef_[1]
→ Bedrooms coefficient

coef_[2]
→ Age coefficient
```

---

# 17. Coefficient interpretation in multiple regression

Suppose:

```text
Area coefficient = 3000
Bedrooms coefficient = 150000
Age coefficient = -20000
```

Interpretation:

### Area

> Baaki features constant rakhte hue, 1 sq ft increase se predicted price ~₹3000 increase.

### Bedrooms

> Baaki variables same hone par 1 additional bedroom predicted price ~₹150000 increase.

### Age

> Baaki variables same hone par 1 year additional age predicted price ~₹20000 decrease.

Very important phrase:

```text
holding other variables constant
```

Multiple regression coefficients interpret karte waqt.

---

# 18. Linear Regression line kaise choose karta hai?

Suppose actual points perfectly line par nahi hain:

```text
Hours    Marks

1        25
2        28
3        43
4        47
5        62
```

No single line exactly every point ko touch karegi.

Linear Regression tries to find:

```text
best-fitting line
```

But "best" ka meaning kya?

It minimizes:

```text
sum of squared residuals/errors
```

Scikit-learn `LinearRegression` coefficients ko residual sum of squares minimize karne ke liye fit karta hai. :chatgpt-content-reference{index="4"}

---

# 19. Residual kya hota hai?

Actual:

```text
y
```

Predicted:

```text
ŷ
```

Residual:

\[
e=y-\hat y
\]

Suppose:

```text
Actual Salary = 50000
Predicted = 47000
```

Residual:

\[
50000-47000=3000
\]

Another:

```text
Actual = 50000
Prediction = 53000
```

Residual:

\[
-3000
\]

---

# 20. Why squared errors?

OLS minimizes:

\[
\sum(y_i-\hat y_i)^2
\]

Suppose errors:

```text
+10
-10
```

If simply add:

```text
10 + (-10) = 0
```

which falsely looks perfect.

Square:

```text
10² + (-10)²
= 100 + 100
= 200
```

Now errors cancel nahi kar sakte.

Squaring larger errors ko bhi more heavily penalize karta hai.

---

# 21. Best-fit line intuition

Imagine many possible lines:

```text
Line A
Line B
Line C
Line D
```

For each line:

```text
actual - predicted
↓
square errors
↓
sum them
```

Whichever line gives minimum squared-error sum:

```text
OLS best-fit line
```

---

# 22. `fit_intercept`

Default:

```python
LinearRegression(
    fit_intercept=True
)
```

Meaning:

```text
intercept learn karo
```

Current stable docs ke according `fit_intercept=False` hone par model intercept calculate nahi karta and data expected to be appropriately centered. :chatgpt-content-reference{index="5"}

Example:

```python
model = LinearRegression(
    fit_intercept=False
)
```

Then model equation becomes roughly:

\[
y=b_1x_1+b_2x_2+\cdots
\]

No independent intercept term.

Beginner rule:

```text
Usually default True hi rakho.
```

Don't disable without a reason.

---

# 23. Does Linear Regression need feature scaling?

Ordinary unregularized `LinearRegression` ke predictions ke liye scaling generally **required nahi** hoti in the same way as KNN/SVM.

If you rescale a feature:

```text
Salary rupees
→ Salary lakhs
```

coefficient adjust ho sakta hai while predictions can remain equivalent.

But scaling can still be useful for:

```text
numerical conditioning
coefficient comparability in some contexts
Pipeline consistency
```

And later:

```text
Ridge
Lasso
ElasticNet
```

me scaling much more important hai because regularization directly coefficient sizes ko penalize karti hai.

---

# 24. Categorical data directly use kar sakte hain?

No, raw:

```text
City = Indore
```

LinearRegression ko directly numeric equation me use nahi kar sakte.

Need encoding:

```text
City
↓
OneHotEncoder
```

Example:

```text
City_Indore
City_Bhopal
City_Dewas
```

Then Linear Regression numeric columns use kar sakta hai.

---

# 25. Missing values?

Raw missing values ke case me preprocessing required ho sakti hai:

```text
SimpleImputer
↓
LinearRegression
```

Best architecture:

```text
Pipeline
↓
ColumnTransformer
↓
LinearRegression
```

Exactly Chapters 13–14 ka use yahan hoga.

---

# 26. Complete train-test example

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8],
    "Salary": [
        25000,
        30000,
        36000,
        41000,
        47000,
        52000,
        59000,
        65000
    ]
})
```

X/y:

```python
X = df[["Experience"]]
y = df["Salary"]
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
y_pred = model.predict(
    X_test
)
```

Inspect:

```python
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", y_pred)
```

---

# 27. Prediction for completely new value

```python
new_employee = pd.DataFrame({
    "Experience": [10]
})

salary = model.predict(
    new_employee
)

print(salary)
```

This gives model's estimated salary for 10 years experience.

---

# 28. Actual vs predicted

Suppose:

```text
y_test:
[41000, 65000]

y_pred:
[42000, 62000]
```

Compare:

```text
Actual   Predicted   Error

41000    42000      -1000
65000    62000       3000
```

Metrics later calculate karenge:

```text
MAE
MSE
RMSE
R²
```

---

# 29. `score()` quick preview

You can:

```python
model.score(
    X_test,
    y_test
)
```

For regressors, sklearn's default `.score()` commonly returns:

```text
R²
```

But next chapter dedicated regression metrics me hum R² ko properly understand karenge rather than blindly relying on `.score()`.

---

# 30. Linear relationship ka meaning

Linear Regression assumes model relation roughly:

```text
straight-line / linear combination
```

Simple case:

```text
X increases
↓
Y approximately constant rate se increases/decreases
```

Example:

```text
Experience
↓
Salary
```

might roughly linear be.

But:

```text
Age
↓
Human height
```

entire lifespan me linear nahi.

Childhood:

```text
height increases
```

Adult:

```text
stabilizes
```

Old age:

```text
may decrease
```

One straight line poor fit ho sakti hai.

---

# 31. Non-linear data example

Suppose:

```text
x   y

1   1
2   4
3   9
4   16
5   25
```

Relationship:

\[
y=x^2
\]

This is curved.

Simple:

\[
y=mx+c
\]

perfectly represent nahi karega.

Later:

```text
Polynomial Regression
```

padhenge.

---

# 32. Outliers ka effect

Linear Regression squared errors minimize karta hai.

So extreme outlier:

```text
Normal salaries:
30000
40000
50000

Outlier:
5000000
```

line ko strongly pull kar sakta hai.

Because huge residual gets squared.

So Linear Regression:

```text
outlier-sensitive
```

ho sakta hai.

Later robust regressors:

```text
HuberRegressor
RANSACRegressor
TheilSenRegressor
```

jaise options bhi exist karte hain.

---

# 33. Correlation ≠ causation

Suppose model learns:

```text
Feature X
strongly associated with
Target Y
```

This does **not automatically mean**:

```text
X causes Y
```

Linear Regression prediction/association model hai.

Causal conclusion ke liye separate study design and assumptions required hote hain.

This is an important professional habit.

---

# 34. Multicollinearity

Suppose features:

```text
Height_cm
Height_inches
```

Both almost same information represent kar rahe hain.

Or:

```text
MonthlySalary
AnnualSalary
```

with:

\[
AnnualSalary = 12 \times MonthlySalary
\]

Features strongly dependent hain.

This can make coefficients unstable/hard to interpret.

This problem:

```text
Multicollinearity
```

Later Ridge Regression is one important way to improve coefficient stability in related situations.

---

# 35. Linear Regression assumptions

For prediction, assumptions ko rigid checklist ki tarah nahi use karna chahiye, but traditional statistical Linear Regression inference me important assumptions hain.

Main ideas:

### 1. Linearity

Relationship can be reasonably represented as linear combination.

### 2. Independent observations/errors

Observations/residual structure shouldn't violate independence assumptions when doing inference.

### 3. Constant error variance

Residual spread roughly consistent:

```text
Homoscedasticity
```

### 4. Residual normality

Especially classical confidence intervals/hypothesis testing ke context me.

### 5. Severe multicollinearity avoid

Input features extremely linearly dependent na hon.

Important:

> Machine-learning prediction ke liye residual normality generally prediction validity ki mandatory condition nahi hai in the simplistic sense beginners often hear. Assumptions matter differently depending on whether goal prediction hai ya statistical inference.

---

# 36. Residual plot intuition

Good-ish pattern:

```text
Residuals randomly scattered around 0
```

Problematic pattern:

```text
Residual
  ↑

  •
    •
      •
        •
          •

────────────→ prediction
```

Structured pattern indicate kar sakta hai model important non-linearity miss kar raha hai.

Later model diagnostics me more detail.

---

# 37. Multiple Linear Regression does not mean multiple models

Important confusion:

```text
Multiple Linear Regression
```

means:

```text
one target
+
multiple input features
```

Not:

```text
multiple LinearRegression models
```

Example:

```text
Area
Bedrooms
Age
↓
ONE LinearRegression
↓
Price
```

---

# 38. Multiple targets bhi possible hain

Advanced case:

```text
X:
Area
Bedrooms

y:
Price
Rent
```

Sklearn `LinearRegression` multiple target columns support kar sakta hai; in that case `coef_` can become 2D. :chatgpt-content-reference{index="6"}

But our current learning:

```text
one target
```

par focus karega.

---

# 39. Current important LinearRegression parameters

Current stable sklearn includes parameters such as:

```python
LinearRegression(
    fit_intercept=True,
    copy_X=True,
    tol=1e-6,
    n_jobs=None,
    positive=False
)
```

The current stable API includes these options; `positive=True` forces non-negative coefficients for dense inputs. :chatgpt-content-reference{index="7"}

Beginner stage me mainly:

```text
fit_intercept
```

understand karna enough hai.

Defaults generally fine.

---

# 40. `positive=True`

Suppose domain says coefficients **must not be negative**.

Then:

```python
model = LinearRegression(
    positive=True
)
```

can force:

```text
coef_ >= 0
```

for supported dense input. :chatgpt-content-reference{index="8"}

But don't use it just because negative coefficient looks strange.

A negative learned coefficient may reflect real conditional relationships or multicollinearity.

---

# 41. Full Pipeline example

Now previous chapters combine karte hain.

Dataset:

```text
Age
Experience
City
Salary ← target
```

Numeric:

```text
Age
Experience
→ impute
→ scale
```

Categorical:

```text
City
→ impute
→ OneHot
```

Model:

```text
LinearRegression
```

Code:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

numeric_features = [
    "Age",
    "Experience"
]

categorical_features = [
    "City"
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

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
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

This is proper end-to-end sklearn regression architecture.

---

# 42. Common mistake #1 — Classification target with LinearRegression

Suppose target:

```text
Purchased:
0
1
```

You technically can fit numbers, but LinearRegression is not the proper standard model for binary classification.

Why?

It can predict:

```text
-0.3
1.4
0.72
```

while we need class probabilities/decision logic.

Use:

```text
LogisticRegression
```

for binary classification.

Despite its name, LogisticRegression is a classification model.

We'll study it later.

---

# 43. Common mistake #2 — Assume coefficient means causality

If:

```text
coef_ = 5000
```

don't automatically conclude:

> Increasing X by manipulating it will cause Y to rise by 5000.

Regression coefficient describes fitted relationship conditional on included features; causal interpretation needs much stronger assumptions.

---

# 44. Common mistake #3 — Ignore non-linearity

If scatterplot clearly curved:

```text
      •
    •
  •
•
```

one straight line may underfit.

Don't force LinearRegression simply because target is numeric.

---

# 45. Common mistake #4 — Evaluate on training data only

Wrong:

```python
model.fit(X, y)

model.score(X, y)
```

and conclude model is excellent.

Better:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)
```

Then evaluate unseen data.

---

# 46. Common mistake #5 — Ignore feature shape

For one feature:

Wrong:

```python
X = df["Experience"]
```

Better:

```python
X = df[["Experience"]]
```

Because:

```text
X
→ 2D feature matrix
```

---

# 47. Common mistake #6 — Interpret coefficient without units

Suppose:

```text
Area measured in square feet
```

Coefficient:

```text
3000
```

Meaning:

```text
₹3000 per 1 square foot
```

If Area converted to:

```text
100 square feet units
```

coefficient changes.

Therefore coefficient magnitude depends on feature units.

---

# 48. LinearRegression vs Ridge/Lasso preview

Current sklearn documentation points to `Ridge`, `Lasso`, and `ElasticNet` as regularized linear alternatives. Ridge adds L2 regularization, Lasso uses L1, and ElasticNet combines both. :chatgpt-content-reference{index="9"}

Mental preview:

```text
LinearRegression
→ no coefficient penalty

Ridge
→ L2 penalty

Lasso
→ L1 penalty

ElasticNet
→ L1 + L2
```

We will learn them after Polynomial Regression.

---

# Chapter 15 Summary

Regression:

```text
Continuous number predict
```

Examples:

```text
Price
Salary
Sales
Temperature
```

Simple Linear Regression:

\[
y=mx+c
\]

Where:

```text
m
→ coefficient / slope

c
→ intercept
```

Scikit-learn:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)
```

Learned values:

```python
model.coef_
model.intercept_
```

`LinearRegression` OLS use karta hai, meaning it fits coefficients to minimize the sum of squared residuals between actual and predicted target values. :chatgpt-content-reference{index="10"}

Multiple Linear Regression:

\[
y =
b_0+b_1x_1+b_2x_2+\dots+b_px_p
\]

Most important concepts:

```text
Actual - Predicted
= Residual

OLS
→ squared residuals ka sum minimize

Positive coefficient
→ positive fitted relationship

Negative coefficient
→ negative fitted relationship
```

And:

```text
Linear Regression
→ regression

Logistic Regression
→ classification
```

### Quick practice

Suppose model learned:

\[
Salary =
20000
+
5000(Experience)
\]

1. `coef_` kya hoga?
2. `intercept_` kya hoga?
3. Experience `6` ho to predicted salary?
4. Agar coefficient `-3000` ho to uska meaning kya hoga?
5. `model.fit(X_train, y_train)` LinearRegression me kya learn karta hai?
6. Residual ka formula kya hai?
7. OLS squared residuals ko kyon minimize karta hai?
8. Agar `X.shape = (1000, 4)` hai, to `coef_` me single-target regression ke liye approximately kitne coefficients honge?
