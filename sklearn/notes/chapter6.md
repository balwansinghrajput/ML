# Chapter 6 — Features (`X`) aur Target (`y`) Deep Dive

Ab hum Machine Learning ka ek bahut fundamental concept samjhenge:

```text
X = Features / Inputs
y = Target / Output
```

Agar `X` aur `y` clear nahi hain, to sklearn ka almost har model confusing lagega. Is chapter ke baad tum dataset ko dekhkar easily identify kar paoge ki **kaunsi columns features hain aur kaunsi target**.

---

## 1. Feature kya hota hai?

Simple definition:

> **Feature wo information hoti hai jo model ko prediction karne ke liye di jati hai.**

Example:

| Age | Salary | Experience | Purchased |
| --: | -----: | ---------: | --------: |
|  22 |  25000 |          1 |         0 |
|  30 |  45000 |          4 |         0 |
|  40 |  70000 |         10 |         1 |
|  50 |  90000 |         18 |         1 |

Suppose hume predict karna hai:

```text
Customer product purchase karega ya nahi?
```

Then:

```text
Age
Salary
Experience
```

are features.

And:

```text
Purchased
```

target hai.

So:

```text
Features:
Age
Salary
Experience

Target:
Purchased
```

---

# 2. Target kya hota hai?

> **Target wo value hai jise model predict karna seekhta hai.**

Example:

```text
Age = 30
Salary = 50000
Experience = 5

              ↓

Model

              ↓

Purchased = ?
```

Here:

```text
Age, Salary, Experience
= Features

Purchased
= Target
```

---

# 3. `X` aur `y`

Machine Learning me convention hai:

```python
X = features
y = target
```

Example:

```python
X = df[["Age", "Salary", "Experience"]]

y = df["Purchased"]
```

Notice:

```text
X → capital X
y → small y
```

Ye mathematical convention hai.

---

# 4. `X` ko capital kyon likhte hain?

Generally `X` ek **2-dimensional feature matrix** hota hai.

Example:

```python
X = [
    [22, 25000, 1],
    [30, 45000, 4],
    [40, 70000, 10],
    [50, 90000, 18]
]
```

Structure:

```text
Rows
↓

[22, 25000, 1]
[30, 45000, 4]
[40, 70000, 10]
[50, 90000, 18]

     ↑
   Columns
```

Here:

```text
4 samples
3 features
```

So shape:

```python
X.shape
```

would be:

```text
(4, 3)
```

Meaning:

```text
4 rows
3 feature columns
```

---

# 5. `y` generally 1D hota hai

Target:

```python
y = [0, 0, 1, 1]
```

Shape:

```text
(4,)
```

Meaning:

```text
4 target values
```

So:

```text
X.shape = (4, 3)

y.shape = (4,)
```

Number of rows same hona chahiye.

---

# 6. Samples kya hote hain?

Dataset ki har row ko generally:

```text
Sample
Observation
Instance
Record
```

kaha ja sakta hai.

Example:

```text
Age = 22
Salary = 25000
Experience = 1
Purchased = 0
```

Ye ek **sample** hai.

Complete dataset:

```text
1000 rows
```

means:

```text
1000 samples
```

---

# 7. Feature ka dusra naam

Tum alag-alag books/videos me ye terms dekhoge:

```text
Feature
Input variable
Independent variable
Predictor variable
Attribute
Column
```

Ye often similar concept refer karte hain.

Target ke liye:

```text
Target
Output
Dependent variable
Response variable
Label
```

Classification me `label` word especially common hai.

---

# 8. Example — House Price Prediction

Dataset:

| Area | Bedrooms | Age |   Price |
| ---: | -------: | --: | ------: |
| 1000 |        2 |  10 | 3000000 |
| 1500 |        3 |   5 | 4500000 |
| 2000 |        4 |   2 | 6000000 |

Question:

```text
House ki price predict karni hai.
```

Therefore:

```text
Features:
Area
Bedrooms
Age

Target:
Price
```

Code:

```python
X = df[["Area", "Bedrooms", "Age"]]

y = df["Price"]
```

---

# 9. Example — Student Pass/Fail

Dataset:

| Study Hours | Attendance | Previous Marks | Result |
| ----------: | ---------: | -------------: | ------ |
|           2 |         50 |             45 | Fail   |
|           5 |         75 |             65 | Pass   |
|           8 |         90 |             85 | Pass   |

Prediction:

```text
Result
```

Then:

```text
X:
Study Hours
Attendance
Previous Marks

y:
Result
```

Code:

```python
X = df[
    ["Study Hours", "Attendance", "Previous Marks"]
]

y = df["Result"]
```

---

# 10. Classification vs Regression target

Target ke type se problem ka type decide hota hai.

### Classification

Target categories/classes:

```text
Yes / No
Spam / Not Spam
Fraud / Normal
Cat / Dog
```

Example:

```python
y = [0, 1, 1, 0]
```

### Regression

Target continuous numbers:

```text
Salary
Price
Temperature
Sales
```

Example:

```python
y = [
    25000,
    40000,
    60000,
    90000
]
```

So:

```text
Categorical target
→ Classification

Continuous numerical target
→ Regression
```

---

# 11. Pandas me X aur y kaise banate hain?

Suppose:

```python
import pandas as pd

df = pd.DataFrame({
    "Age": [22, 30, 40, 50],
    "Salary": [25000, 45000, 70000, 90000],
    "Experience": [1, 4, 10, 18],
    "Purchased": [0, 0, 1, 1]
})
```

Features:

```python
X = df[
    ["Age", "Salary", "Experience"]
]
```

Target:

```python
y = df["Purchased"]
```

Print:

```python
print(X)
print(y)
```

---

# 12. Target remove karke X banana

Agar almost sari columns features hain except target:

```python
X = df.drop(
    "Purchased",
    axis=1
)
```

Then:

```python
y = df["Purchased"]
```

Ye common approach hai.

---

# 13. `axis=1` ka meaning

```python
df.drop("Purchased", axis=1)
```

means:

```text
axis=1
→ column remove
```

Whereas:

```text
axis=0
→ row direction
```

So:

```python
X = df.drop("Purchased", axis=1)
```

means Purchased column hata do.

---

# 14. Better modern syntax

Pandas me readable syntax:

```python
X = df.drop(
    columns=["Purchased"]
)
```

This is often easier to understand than:

```python
axis=1
```

So:

```python
X = df.drop(columns=["Purchased"])

y = df["Purchased"]
```

---

# 15. Multiple columns remove karna

Suppose dataset:

```text
ID
Name
Age
Salary
Experience
Purchased
```

Maybe:

```text
ID
Name
```

prediction ke liye useful nahi.

Then:

```python
X = df.drop(
    columns=[
        "Purchased",
        "ID",
        "Name"
    ]
)

y = df["Purchased"]
```

But feature remove karne ka decision blindly nahi karna. Us feature ka meaning samajhna hota hai.

---

# 16. Target X me nahi hona chahiye

Bahut important.

Wrong:

```python
X = df[
    ["Age", "Salary", "Purchased"]
]

y = df["Purchased"]
```

Problem:

Model ko answer input me hi mil gaya.

This is severe:

```text
Target Leakage
```

Model unrealistic performance de sakta hai.

Correct:

```python
X = df[
    ["Age", "Salary"]
]

y = df["Purchased"]
```

Rule:

```text
Target ko features me include mat karo.
```

---

# 17. Data leakage ka easy example

Suppose target:

```text
Loan_Default
```

Features me accidentally:

```text
Default_Status_After_6_Months
```

include kar diya.

Ye information prediction time par available hi nahi hogi.

Model excellent score de sakta hai, but real world me useless ho jayega.

So feature selection me ye question poochna chahiye:

> Kya ye feature actual prediction ke waqt available hoga?

---

# 18. `X` 2D kyon hona chahiye?

Scikit-learn generally feature input ko:

```text
(samples, features)
```

form me expect karta hai.

Suppose:

```python
X = [1, 2, 3, 4]
```

Shape conceptually:

```text
(4,)
```

1-dimensional.

LinearRegression often expects:

```text
2D array
```

Correct:

```python
X = [
    [1],
    [2],
    [3],
    [4]
]
```

Shape:

```text
(4, 1)
```

Meaning:

```text
4 samples
1 feature
```

---

# 19. `(4,)` vs `(4,1)`

Ye distinction very important hai.

```text
(4,)
```

means:

```text
1-dimensional array
```

While:

```text
(4, 1)
```

means:

```text
4 rows × 1 column
```

Machine learning feature matrix normally:

```text
(n_samples, n_features)
```

honi chahiye.

---

# 20. Single feature with Pandas

Suppose:

```python
X = df["Age"]
```

This gives a Pandas Series.

Shape:

```text
(100,)
```

But sklearn feature matrix ke liye better:

```python
X = df[["Age"]]
```

Notice double brackets.

Shape:

```text
(100, 1)
```

This is one of the most common beginner mistakes.

---

# 21. Single bracket vs double bracket

### Single bracket

```python
df["Age"]
```

returns:

```text
Series
```

Usually 1D.

### Double bracket

```python
df[["Age"]]
```

returns:

```text
DataFrame
```

2D.

For X with one feature:

```python
X = df[["Age"]]
```

usually appropriate.

For y:

```python
y = df["Purchased"]
```

1D Series is usually fine.

---

# 22. Example

```python
print(df["Age"].shape)
```

Could output:

```text
(100,)
```

But:

```python
print(df[["Age"]].shape)
```

returns:

```text
(100, 1)
```

So:

```text
df["Age"]
→ Series

df[["Age"]]
→ DataFrame
```

---

# 23. `X.shape` aur `y.shape` always check karo

ML workflow me useful habit:

```python
print(X.shape)
print(y.shape)
```

Suppose:

```text
X shape:
(1000, 5)

y shape:
(1000,)
```

Perfectly reasonable.

Meaning:

```text
1000 samples
5 features
1000 targets
```

---

# 24. X aur y rows same honi chahiye

Wrong:

```text
X.shape
(1000, 5)

y.shape
(900,)
```

Model training nahi ho payegi because:

```text
1000 input samples
but
900 answers
```

Every training sample ko corresponding target chahiye.

Correct:

```text
X rows = y values
```

---

# 25. Feature matrix ko visually samjho

Suppose:

```python
X = [
    [22, 25000, 1],
    [30, 45000, 4],
    [40, 70000, 10],
]
```

Think:

```text
             Features
         ↓       ↓       ↓

       Age    Salary   Experience

Row 1   22     25000       1
Row 2   30     45000       4
Row 3   40     70000      10

 ↑
Samples
```

Shape:

```text
3 × 3
```

---

# 26. Corresponding target

```python
y = [
    0,
    0,
    1
]
```

Relation:

```text
X row 1 → y[0]
X row 2 → y[1]
X row 3 → y[2]
```

Example:

```text
22,25000,1
→ 0

30,45000,4
→ 0

40,70000,10
→ 1
```

Model ye mapping learn karta hai.

---

# 27. `fit(X, y)` ab aur clear hona chahiye

When:

```python
model.fit(X, y)
```

tum actually keh rahe ho:

> In input features aur corresponding target values ke beech relationship learn karo.

Example:

```text
X
Age Salary Experience

        ↓

model.fit(X, y)

        ↓

y
Purchased
```

---

# 28. `predict(X_new)` me y kyon nahi dete?

Training:

```python
model.fit(X_train, y_train)
```

because correct answers available hain.

Prediction:

```python
model.predict(X_new)
```

because target unknown hai.

Agar target already pata hota:

```text
predict karne ki zarurat hi nahi hoti.
```

So:

```text
Training:
X + y

Prediction:
X only
```

---

# 29. Supervised Learning

Jab training ke time:

```text
X + y
```

dono available hote hain, it's supervised learning.

Example:

```python
model.fit(X, y)
```

Typical algorithms:

```text
Linear Regression
Logistic Regression
Decision Tree
Random Forest
SVM
KNN
```

---

# 30. Unsupervised Learning

Unsupervised learning me target:

```text
y
```

nahi hota.

Example dataset:

```text
Age
Salary
Spending Score
```

Hume customers ko groups me divide karna hai.

KMeans:

```python
kmeans.fit(X)
```

Notice:

```text
No y
```

Because model khud patterns/groups discover karta hai.

---

# 31. Supervised vs Unsupervised

```text
Supervised:

X + y
 ↓
model
 ↓
learn relationship
```

```text
Unsupervised:

X
 ↓
model
 ↓
discover structure
```

Example:

```python
LinearRegression().fit(X, y)
```

vs:

```python
KMeans().fit(X)
```

---

# 32. Feature types

Features alag types ke ho sakte hain.

### Numerical

```text
Age
Salary
Height
Weight
Temperature
```

### Categorical

```text
City
Gender
Color
Education
```

### Binary

```text
Yes/No
0/1
True/False
```

### Ordinal

```text
Low
Medium
High
```

Different feature types ko different preprocessing chahiye.

Later:

```text
Numerical
→ scaling

Categorical
→ encoding
```

---

# 33. Example mixed dataset

```text
Age        Numerical
Salary     Numerical
City       Categorical
Education  Ordinal
Purchased  Target
```

X:

```text
Age
Salary
City
Education
```

y:

```text
Purchased
```

Later ColumnTransformer ke through:

```text
Age / Salary
→ scaling

City
→ OneHotEncoding

Education
→ OrdinalEncoding
```

kar sakte hain.

---

# 34. Raw feature aur engineered feature

Raw features:

```text
Date of Birth
Monthly Salary
Total Purchases
```

Inse new features create kar sakte hain:

```text
Age
Annual Salary
Average Purchase Value
```

Is process ko:

```text
Feature Engineering
```

kehte hain.

Example:

```python
df["AnnualSalary"] = df["MonthlySalary"] * 12
```

Then `AnnualSalary` X ka part ban sakta hai.

---

# 35. Feature name aur feature value

Don't confuse.

Example:

```text
Age = 25
```

Here:

```text
Age
= feature name

25
= feature value
```

Similarly:

```text
City = Indore
```

Feature:

```text
City
```

Value:

```text
Indore
```

---

# 36. Number of features

Suppose:

```python
X.shape
```

returns:

```text
(5000, 12)
```

Then:

```text
5000 samples
12 features
```

So:

```python
X.shape[0]
```

gives samples:

```text
5000
```

and:

```python
X.shape[1]
```

gives features:

```text
12
```

---

# 37. Feature names check karna

Pandas:

```python
print(X.columns)
```

Output:

```text
Index([
    'Age',
    'Salary',
    'Experience'
])
```

Useful before model training.

---

# 38. Target distribution check karna

Classification:

```python
y.value_counts()
```

Example:

```text
0    900
1    100
```

This tells:

```text
Class 0 = 900
Class 1 = 100
```

Could indicate imbalance.

Later imbalanced dataset chapter me detail me karenge.

---

# 39. Target me missing values?

Features me missing values commonly impute ki ja sakti hain.

But target missing hona more serious hai.

Example:

```text
Age Salary Purchased
22  30000   0
30  50000   NaN
40  70000   1
```

Second row ka correct training answer hi missing hai.

Usually supervised learning me target missing rows ko separately handle karna padta hai; simply ordinary feature imputation jaisa treat nahi karte.

---

# 40. ID feature ka issue

Suppose:

```text
Customer_ID
1001
1002
1003
1004
```

ID often sirf identification ke liye hoti hai.

Model ko:

```text
Customer_ID = 1004
```

ka meaningful predictive relation necessarily nahi hota.

So IDs frequently X se remove ki jati hain.

But always context samjho. Har integer column automatically useless ID nahi hoti.

---

# 41. Date features

Suppose:

```text
PurchaseDate
2026-01-05
2026-07-10
```

Raw date ko model direct useful form me samajh nahi sakta depending on estimator.

Hum extract kar sakte hain:

```text
Year
Month
Day
DayOfWeek
Weekend
```

Example:

```python
df["Month"] = df["PurchaseDate"].dt.month
```

Ye feature engineering hai.

---

# 42. Text features

Suppose:

```text
Review
"This product is excellent"
```

Ye raw text hai.

Classic sklearn ML model ko numbers chahiye.

So transform:

```text
Text
 ↓
Vectorizer
 ↓
Numbers
```

Using later concepts:

```text
CountVectorizer
TfidfVectorizer
```

Then numerical representation X ban sakta hai.

---

# 43. Model numbers ke saath kaam karta hai

Most sklearn estimators finally numerical matrix expect karte hain.

Raw:

```text
City = Indore
Education = Graduate
```

needs encoding.

After preprocessing:

```text
City_Indore = 1
City_Bhopal = 0
Education code = ...
```

Hence:

```text
Raw X
 ↓
Preprocessing
 ↓
Numerical X
 ↓
Model
```

---

# 44. Features ki scale different ho sakti hai

Example:

```text
Age
20 – 60

Salary
20,000 – 200,000
```

Range bahut different.

Some algorithms ke liye scaling important hoti hai.

Later:

```text
StandardScaler
MinMaxScaler
RobustScaler
```

use karenge.

But abhi basic:

```text
Features = inputs
Scaling = inputs ko suitable representation dena
```

---

# 45. One feature prediction

Suppose:

```text
Hours Studied
→ Marks
```

X:

```python
X = df[["Hours"]]
```

y:

```python
y = df["Marks"]
```

Even though only one feature hai, X ko 2D rakhte hain:

```text
(n_samples, 1)
```

---

# 46. Multiple feature prediction

Suppose:

```text
Hours
Attendance
PreviousMarks
→ FinalMarks
```

Then:

```python
X = df[
    ["Hours", "Attendance", "PreviousMarks"]
]

y = df["FinalMarks"]
```

Shape could be:

```text
X = (1000, 3)
y = (1000,)
```

---

# 47. Multi-output target

Advanced case me `y` bhi multiple columns ho sakta hai.

Suppose inputs:

```text
Temperature
Humidity
Wind
```

Predict:

```text
Tomorrow Temperature
Tomorrow Humidity
```

Then y could be 2D:

```text
(n_samples, 2)
```

Scikit-learn me multi-output models possible hain.

But beginner level par mostly:

```text
y = one target column
```

use karenge.

---

# 48. Feature-target relationship visual

```text
            X
   ┌─────────────────┐
   │ Age             │
   │ Salary          │
   │ Experience      │
   └─────────────────┘
            ↓
       ML Model
            ↓
            y
   ┌─────────────────┐
   │ Purchased       │
   └─────────────────┘
```

Training:

```text
Known X
+
Known y
```

Prediction:

```text
Known X
+
Unknown y
```

---

# 49. Complete example

```python
import pandas as pd

df = pd.DataFrame({
    "Age": [22, 30, 40, 50, 28, 45],
    "Salary": [
        25000,
        45000,
        70000,
        90000,
        35000,
        80000
    ],
    "Experience": [1, 4, 10, 18, 3, 14],
    "Purchased": [0, 0, 1, 1, 0, 1]
})
```

Create X:

```python
X = df.drop(
    columns=["Purchased"]
)
```

Create y:

```python
y = df["Purchased"]
```

Check:

```python
print(X)
print(y)
```

Shapes:

```python
print(X.shape)
print(y.shape)
```

Output:

```text
X.shape
(6, 3)

y.shape
(6,)
```

Meaning:

```text
6 samples
3 features
6 target values
```

---

# 50. Train model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
```

Train:

```python
model.fit(X, y)
```

Now this statement ka meaning clearly samjho:

```text
Age, Salary aur Experience ko use karke
Purchased target ka relationship learn karo.
```

Then:

```python
prediction = model.predict(
    [[35, 55000, 6]]
)
```

Conceptually:

```text
Age = 35
Salary = 55000
Experience = 6

          ↓

Model

          ↓

Purchased = 0 or 1
```

---

# 51. New DataFrame prediction better ho sakta hai

Since training features had names, better:

```python
new_customer = pd.DataFrame({
    "Age": [35],
    "Salary": [55000],
    "Experience": [6]
})

prediction = model.predict(new_customer)
```

This makes feature meaning clearer.

---

# 52. Feature order rule

If X:

```python
X = df[
    ["Age", "Salary", "Experience"]
]
```

then raw array prediction should follow:

```text
Age
Salary
Experience
```

order.

Correct:

```python
[[35, 55000, 6]]
```

Wrong:

```python
[[55000, 6, 35]]
```

Same values, wrong meaning.

---

# 53. Target column detect kaise karein?

Simple question:

> Model se exactly kya predict karwana hai?

Us question ka answer target hai.

Example:

### Question

```text
Employee ki salary predict karni hai.
```

Target:

```text
Salary
```

### Question

```text
Customer churn karega?
```

Target:

```text
Churn
```

### Question

```text
House ki price kya hogi?
```

Target:

```text
Price
```

Baaki useful information features ban sakti hai.

---

# 54. Kya har non-target column feature hona chahiye?

No.

Suppose:

```text
CustomerID
Name
Phone
Age
Salary
Purchased
```

Target:

```text
Purchased
```

Potential features:

```text
Age
Salary
```

But:

```text
CustomerID
Name
Phone
```

usually directly useful ML features nahi honge.

So:

```text
Target decide karna
≠
Baaki sab blindly X me daal dena
```

Feature relevance, availability, leakage aur data type check karna hota hai.

---

# 55. Feature selection aur feature engineering difference

### Feature Selection

Existing features me se useful choose karna.

```text
Age
Salary
Experience
ID

↓ select

Age
Salary
Experience
```

### Feature Engineering

Existing data se new features banana.

```text
DateOfBirth
↓
Age
```

Both later detail me karenge.

---

# 56. X aur y ke types

Pandas workflow:

```python
type(X)
```

usually:

```text
pandas.DataFrame
```

And:

```python
type(y)
```

usually:

```text
pandas.Series
```

This is perfectly normal.

Scikit-learn NumPy arrays bhi accept karta hai.

---

# 57. NumPy example

```python
import numpy as np

X = np.array([
    [20, 20000],
    [30, 40000],
    [40, 60000]
])

y = np.array([
    0,
    0,
    1
])
```

Then:

```python
model.fit(X, y)
```

Same concept.

---

# 58. DataFrame vs NumPy

Sklearn dono ke saath commonly kaam karta hai.

Pandas advantage:

```text
Column names
Easy data manipulation
Readable
```

NumPy advantage:

```text
Efficient numerical arrays
```

Real ML workflow me often:

```text
Pandas
↓
Scikit-learn
↓
NumPy-like transformations
```

dekhoge.

---

# 59. Dataset ka structure inspect karna

Before X/y:

```python
df.head()
```

Then:

```python
df.shape
```

Then:

```python
df.columns
```

Then:

```python
df.info()
```

Then target identify karo.

Finally:

```python
X = ...
y = ...
```

Ye achhi workflow habit hai.

---

# 60. Chapter 6 ka core mental model

```text
Dataset

Age   Salary   Experience   Purchased
 ↓       ↓          ↓           ↓
Feature Feature    Feature      Target

             ↓

X = Age + Salary + Experience

y = Purchased
```

Then:

```python
model.fit(X, y)
```

means:

```text
X se y ka relation learn karo
```

Then:

```python
model.predict(X_new)
```

means:

```text
New X ke liye y estimate karo
```

---

# Chapter 6 Summary

Sabse important:

```text
X
= Features
= Inputs
= Independent/Predictor variables

y
= Target
= Output
= Dependent/Response variable
```

Typical shapes:

```text
X.shape
= (n_samples, n_features)

y.shape
= (n_samples,)
```

Example:

```python
X = df.drop(columns=["Purchased"])

y = df["Purchased"]
```

For one feature:

```python
X = df[["Age"]]
```

not usually:

```python
X = df["Age"]
```

because:

```text
df[["Age"]]
→ 2D DataFrame
```

Most important training relationship:

```text
X_train
+
y_train
↓
fit()
↓
model learns
```

Prediction:

```text
X_new
↓
predict()
↓
predicted y
```

And target must **not** accidentally be included inside `X`.

### Quick test

Suppose dataset:

```text
Experience   Projects   SkillScore   Salary
2            4          65           30000
5            8          80           55000
10           15         92           90000
```

Agar hume **Salary predict** karni hai, to:

```text
X = ?
y = ?
```

Aur agar:

```python
X.shape = (5000, 8)
```

to batao `5000` aur `8` ka kya meaning hai.
