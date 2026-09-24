# Chapter 9 — Advanced Imputation: `KNNImputer`, `IterativeImputer` & `MissingIndicator`

Chapter 8 me humne `SimpleImputer` padha:

```text
SimpleImputer
↓
Ek column ki missing value
↓
usi column ka
mean / median / mode / constant
↓
use karke fill
```

Ab problem ye hai ki kabhi missing value ko sirf usi column se estimate karna enough nahi hota.

Suppose:

| Age | Experience | Salary |
| --: | ---------: | -----: |
|  24 |          2 |  30000 |
|  26 |          3 |  35000 |
|  25 |          2 |    NaN |
|  50 |         25 | 150000 |

Missing salary wale person ki:

```text
Age = 25
Experience = 2
```

hai.

Obviously wo first two people ke zyada similar lag raha hai.

`SimpleImputer(strategy="mean")` poori Salary column ka mean use karega. Lekin `KNNImputer` similar rows dhundh kar missing salary estimate kar sakta hai.

Scikit-learn advanced missing-value handling ke liye `KNNImputer`, `IterativeImputer`, aur `MissingIndicator` provide karta hai. ([scikit-learn][1])

---

# 1. Univariate vs Multivariate Imputation

Ye difference sabse pehle samjho.

### Univariate Imputation

`SimpleImputer`:

```text
Age column
↓
Age column ko hi dekho
↓
median calculate
↓
missing Age fill
```

Example:

```text
Age

20
30
NaN
40
```

Median:

```text
30
```

Missing value:

```text
30
```

ban jayegi.

---

### Multivariate Imputation

Advanced imputers dusre features ko bhi dekh sakte hain.

Example:

```text
Age   Experience   Salary

24       2         30000
26       3         35000
25       2         NaN
50      25        150000
```

Missing Salary estimate karte waqt:

```text
Age
Experience
```

ki information bhi use ho sakti hai.

This is:

```text
Multivariate Imputation
```

`KNNImputer` aur `IterativeImputer` dono multivariate approaches hain, lekin dono ka mechanism different hai. ([scikit-learn][2])

---

# 2. `KNNImputer` kya hai?

Definition:

> **`KNNImputer` missing value wale sample ke similar/nearest training samples find karta hai aur un neighboring samples ki values se missing value estimate karta hai.**

Import:

```python
from sklearn.impute import KNNImputer
```

Basic:

```python
imputer = KNNImputer(
    n_neighbors=5
)
```

Current stable sklearn me default `n_neighbors=5`, `weights="uniform"` aur distance metric `nan_euclidean` hai. Missing value ko nearest training samples ke corresponding values ke mean se estimate kiya jata hai. ([scikit-learn][3])

---

# 3. KNN ka basic idea

`KNN` means:

```text
K
Nearest
Neighbors
```

Suppose ek student:

```text
Age = 20
StudyHours = 6
Marks = NaN
```

Dataset:

```text
Age  StudyHours  Marks

19       6        70
20       7        75
21       6        72
45       2        30
```

Missing student:

```text
Age = 20
StudyHours = 6
```

first 3 students ke close hai.

So instead of:

```text
Entire Marks column mean
```

KNNImputer use kar sakta hai:

```text
70
75
72
```

ka average.

$$
(70+75+72)/3 = 72.33
$$

Missing Marks approximately:

```text
72.33
```

ho sakte hain.

---

# 4. Basic `KNNImputer` code

```python
import numpy as np

from sklearn.impute import KNNImputer

X = np.array([
    [24, 2, 30000],
    [26, 3, 35000],
    [25, 2, np.nan],
    [50, 25, 150000]
])

imputer = KNNImputer(
    n_neighbors=2
)

X_new = imputer.fit_transform(X)

print(X_new)
```

Conceptually third row ki salary nearest similar rows se estimate hogi.

Scikit-learn ke official example me bhi `n_neighbors=2` ke saath neighboring samples ki values use karke missing fields fill ki jati hain. ([scikit-learn][3])

---

# 5. KNNImputer ka `fit()` kya karta hai?

Chapter 2 yaad karo:

```text
fit()
= learn/store information
```

KNNImputer ke case me:

```python
imputer.fit(X_train)
```

training data ko store/prepare karta hai taki later nearest training samples find kiye ja sakein.

Then:

```python
imputer.transform(X_test)
```

test row ke nearest **training samples** use karke missing values impute karega.

Very important:

```text
X_train
↓
fit

X_test
↓
transform
```

---

# 6. Train-Test rule yahan bhi same hai

Correct:

```python
X_train_new = imputer.fit_transform(X_train)

X_test_new = imputer.transform(X_test)
```

Wrong:

```python
X_train_new = imputer.fit_transform(X_train)

X_test_new = imputer.fit_transform(X_test)
```

Kyon?

Because:

```text
Test data se learning nahi karni.
```

Same rule jo `SimpleImputer` ke saath tha.

---

# 7. `n_neighbors`

Most important parameter:

```python
KNNImputer(
    n_neighbors=5
)
```

Means:

```text
5 nearest samples ko use karo.
```

Current default:

```text
5
```

hai. ([scikit-learn][3])

For example:

```python
imputer = KNNImputer(
    n_neighbors=3
)
```

Missing value:

```text
3 nearest neighbours
↓
corresponding observed values
↓
combine
↓
missing value estimate
```

---

# 8. K bahut small ya large ho to?

Suppose:

```python
n_neighbors=1
```

Sirf closest row par heavy dependence hogi.

Example closest Salary:

```text
30000
```

Then missing:

```text
30000
```

ho sakti hai.

But:

```python
n_neighbors=10
```

more neighbors involve honge.

Possible result more smooth/generic ho sakta hai.

So:

```text
K too small
→ local/noisy estimate ho sakta hai

K larger
→ smoother estimate
```

Best `n_neighbors` automatically universally `5` nahi hota.

Eventually Cross Validation ke through different values compare kar sakte hain.

---

# 9. `weights="uniform"`

Default:

```python
KNNImputer(
    weights="uniform"
)
```

Means:

> Sab selected neighbors ko equal importance do.

Suppose 3 neighbors Salary:

```text
30000
32000
40000
```

Then roughly:

$$
\frac{30000+32000+40000}{3}
$$

use hoga.

---

# 10. `weights="distance"`

We can use:

```python
imputer = KNNImputer(
    n_neighbors=3,
    weights="distance"
)
```

Then:

```text
Very close neighbor
→ more importance

Farther neighbor
→ less importance
```

Current sklearn docs ke according `distance` weighting inverse distance use karta hai, so closer points ka influence zyada hota hai. ([scikit-learn][3])

Easy example:

```text
Missing person Age = 25

Neighbor A Age = 24
Salary = 30000

Neighbor B Age = 40
Salary = 80000
```

Neighbor A zyada similar hai.

Distance weighting me:

```text
30000
```

ka influence zyada ho sakta hai.

---

# 11. KNN distance kaise calculate karta hai jab NaN present hai?

Normal Euclidean distance:

$$
d(x,y)
=
\sqrt{\sum(x_i-y_i)^2}
$$

Lekin agar features me `NaN` hai, normal distance calculation problematic ho jati hai.

KNNImputer by default:

```python
metric="nan_euclidean"
```

use karta hai, jo shared non-missing features ko use karke distance determine karne ke liye designed hai. ([scikit-learn][3])

Abhi exact mathematical formula memorize karne ki zarurat nahi.

Mental model:

```text
Available common features
↓
distance calculate
↓
similar rows find
```

---

# 12. Scaling KNNImputer me important kyon hai?

Bahut important.

Suppose features:

```text
Age:
20 – 60

Salary:
20,000 – 10,00,000
```

Distance calculate karte waqt Salary ka numerical range Age se huge hai.

Then:

```text
Salary
```

distance ko dominate kar sakti hai.

Official sklearn example bhi note karta hai ki features ke scales bahut different hon to KNN imputation se pehle rescaling potentially performance improve kar sakti hai. ([scikit-learn][4])

Isliye distance-based methods me scaling ko seriously consider karna chahiye.

Feature Scaling chapter me hum is concept ko bahut detail me padhenge.

---

# 13. KNNImputer ke advantages

KNN ka main benefit:

```text
Missing value
↓
poore column ka one constant
nahi
↓
similar samples ke according
different estimates
```

Compare:

SimpleImputer:

```text
Person A missing Salary → 50000
Person B missing Salary → 50000
Person C missing Salary → 50000
```

KNNImputer:

```text
Person A → 32000
Person B → 75000
Person C → 110000
```

depending on their neighbors.

Isse feature relationships ka kuch information preserve ho sakta hai.

---

# 14. KNNImputer disadvantages

KNN always better nahi hai.

Large dataset me neighbors calculate karna expensive ho sakta hai.

Feature scaling important ho sakti hai.

Irrelevant features:

```text
similarity calculation
```

ko hurt kar sakte hain.

High-dimensional data me "nearest" neighbor ka concept less useful ho sakta hai.

Aur missingness bahut high ho to reliable neighbor find karna difficult ho sakta hai.

So:

```text
Advanced
≠
Automatically better
```

---

# 15. `add_indicator=True` with KNNImputer

Just like SimpleImputer:

```python
imputer = KNNImputer(
    n_neighbors=5,
    add_indicator=True
)
```

Then model ko:

```text
Imputed value
+
Originally missing tha ya nahi
```

dono information mil sakti hai.

Current KNNImputer API directly `add_indicator` support karti hai. ([scikit-learn][3])

---

# 16. Ab `IterativeImputer`

`IterativeImputer` ka idea KNN se different hai.

Definition:

> **Har feature jisme missing values hain, us feature ko temporarily target maan kar baaki features se uski missing values predict ki jati hain, aur ye process multiple rounds me repeat hota hai.**

Current stable sklearn docs ise multivariate imputer describe karti hain jo each missing feature ko other features ke function ke roop me model karta hai in a round-robin fashion. ([scikit-learn][2])

---

# 17. Example

Dataset:

```text
Age   Experience   Salary

25       2         30000
30       5         50000
35       9         NaN
40      NaN        80000
NaN     20        120000
```

Three columns me missing values hain.

IterativeImputer roughly relations learn kar sakta hai:

```text
Salary
← Age + Experience

Experience
← Age + Salary

Age
← Experience + Salary
```

Then missing values predict karta hai.

---

# 18. `IterativeImputer` ka step-by-step intuition

Suppose:

```text
Age       Salary       Experience

20        30000        1
30        50000        5
40        NaN          10
50        100000       NaN
```

Salary missing hai.

Temporarily:

```text
Salary = target
Age + Experience = features
```

Model learn karega:

```text
Age + Experience
↓
Salary
```

Then missing Salary predict karega.

Next Experience missing:

```text
Experience = target
Age + Salary = features
```

Then missing Experience predict karega.

Process multiple rounds repeat ho sakta hai:

```text
Round 1
↓
Round 2
↓
Round 3
...
```

until stopping condition or `max_iter`.

---

# 19. Import important hai

Current stable sklearn docs me `IterativeImputer` abhi experimental hai, isliye stable release me pehle experimental enable import karna padta hai. ([scikit-learn][2])

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

Then:

```python
imputer = IterativeImputer(
    random_state=42
)
```

Important version note: development docs show this requirement is planned to change in sklearn 1.10, but current stable docs still require the experimental enable import. ([scikit-learn][5])

---

# 20. Basic example

```python
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

X = np.array([
    [20, 1, 30000],
    [30, 5, 50000],
    [40, 10, np.nan],
    [50, np.nan, 100000]
])

imputer = IterativeImputer(
    random_state=42
)

X_new = imputer.fit_transform(X)
```

Now missing values dusre features ke relationships ki help se estimate hongi.

---

# 21. IterativeImputer internally kaunsa model use karta hai?

Default estimator current stable sklearn me:

```text
BayesianRidge
```

hai. ([scikit-learn][2])

Conceptually:

```text
Missing feature
↓
temporarily target
↓
Regression model
↓
other features se missing value predict
```

---

# 22. `max_iter`

Default:

```python
IterativeImputer(
    max_iter=10
)
```

Current stable default maximum rounds `10` hai. ([scikit-learn][2])

Think:

```text
Iteration 1
→ estimates

Iteration 2
→ estimates update

Iteration 3
→ estimates update

...
```

It may stop earlier depending on convergence settings.

---

# 23. Initial values kahan se aati hain?

Before iterative modeling starts, missing values ko initial values deni padti hain.

Default:

```python
initial_strategy="mean"
```

Other options current stable API me:

```text
mean
median
most_frequent
constant
```

hain. ([scikit-learn][2])

Example:

```python
imputer = IterativeImputer(
    initial_strategy="median",
    random_state=42
)
```

---

# 24. KNN vs Iterative ka main difference

KNN:

```text
Missing row
↓
similar rows find
↓
neighbors ki values
↓
missing value estimate
```

Iterative:

```text
Missing column
↓
baaki columns ko features banao
↓
predictive model train
↓
missing values predict
↓
multiple rounds
```

So both multivariate hain, but logic entirely different hai.

---

# 25. SimpleImputer vs KNNImputer vs IterativeImputer

| Method             | Main idea                           | Other features use? |  Complexity |
| ------------------ | ----------------------------------- | ------------------: | ----------: |
| `SimpleImputer`    | Mean/median/mode                    |                   ❌ |         Low |
| `KNNImputer`       | Similar rows                        |                   ✅ | Medium/high |
| `IterativeImputer` | Other features se predictive models |                   ✅ |      Higher |

Important:

> Complex imputer automatically better prediction nahi deta.

Scikit-learn docs specifically note karti hain ki missing-value pattern ke depending on simple imputers prediction tasks me preferable ho sakte hain. ([scikit-learn][2])

---

# 26. `IterativeImputer` computationally expensive ho sakta hai

Because imagine:

```text
20 columns with missing values
```

Har column ke liye:

```text
model fit
```

and this multiple iterations me repeat.

Current documentation bhi warn karti hai ki feature count badhne par default IterativeImputer computationally expensive ho sakta hai. ([scikit-learn][2])

So large dataset me blindly use mat karo.

---

# 27. Train-test rule IterativeImputer me bhi

Correct:

```python
imputer = IterativeImputer(
    random_state=42
)

X_train_new = imputer.fit_transform(X_train)

X_test_new = imputer.transform(X_test)
```

`transform(X_test)` test data par imputation apply karta hai without refitting; sklearn fitted feature estimators store karta hai for this inductive behavior. ([scikit-learn][2])

Wrong:

```python
X_test_new = imputer.fit_transform(X_test)
```

Again data leakage.

---

# 28. Ab `MissingIndicator`

Suppose:

```text
Salary

30000
NaN
50000
NaN
```

After median imputation:

```text
30000
40000
50000
40000
```

Problem:

Model ko nahi pata:

```text
40000 original tha?
```

ya:

```text
NaN ko 40000 banaya gaya?
```

Sometimes this difference itself useful hota hai.

---

# 29. MissingIndicator kya karta hai?

> **MissingIndicator missing values fill nahi karta. Ye binary features create karta hai jo batate hain value missing thi ya nahi.**

Import:

```python
from sklearn.impute import MissingIndicator
```

Example:

```python
indicator = MissingIndicator()

missing_flags = indicator.fit_transform(X_train)
```

Output type:

```text
True / False
```

ya equivalent binary representation.

Current sklearn docs MissingIndicator ko binary indicators for missing values define karti hain. ([scikit-learn][6])

---

# 30. Easy example

Original:

```text
Age     Salary

20      30000
NaN     40000
30      NaN
```

Missing indicator conceptually:

```text
Age_missing   Salary_missing

0             0
1             0
0             1
```

So:

```text
1
= originally missing

0
= available
```

---

# 31. MissingIndicator value fill nahi karta

Important.

This:

```python
indicator.fit_transform(X)
```

does **not** turn:

```text
NaN → median
```

It only tells:

```text
NaN tha ya nahi.
```

Usually you want:

```text
Imputation
+
Missing indicator
```

---

# 32. Easiest method: `add_indicator=True`

Often separate `MissingIndicator` banane ki zarurat nahi.

Example:

```python
imputer = SimpleImputer(
    strategy="median",
    add_indicator=True
)
```

Or:

```python
imputer = KNNImputer(
    add_indicator=True
)
```

Or current `IterativeImputer` also supports:

```python
add_indicator=True
```

Then:

```text
imputed features
+
missing indicator features
```

automatically append ho sakte hain. ([scikit-learn][3])

---

# 33. Kab MissingIndicator useful ho sakta hai?

Suppose Loan dataset me:

```text
Income = missing
```

random nahi hai.

Maybe some type ke customers income disclose nahi karte.

Then:

```text
Income_missing = 1
```

itself model ke liye useful information ho sakta hai.

Another example:

```text
Medical test result missing
```

could potentially reflect ki test karaya hi nahi gaya.

So missingness itself predictive signal ho sakti hai.

---

# 34. Important MissingIndicator behavior

Default:

```python
MissingIndicator(
    features="missing-only"
)
```

means only un columns ke indicator create honge jahan **fit time** par missing values dekhi gayi.

Current API me `features="all"` bhi available hai, jahan all input features ke missingness indicators output hote hain. ([scikit-learn][6])

Example:

```python
indicator = MissingIndicator(
    features="all"
)
```

---

# 35. `error_on_new`

Current default:

```python
error_on_new=True
```

when using:

```text
features="missing-only"
```

If test/transform data me kisi aise feature me missing value aa gayi jisme fit time par missing values nahi thi, MissingIndicator error raise kar sakta hai. ([scikit-learn][6])

Example:

Training:

```text
Age has missing
Salary has NO missing
```

Test:

```text
Salary suddenly has NaN
```

This behavior ko production pipeline design karte waqt understand karna important hai.

---

# 36. KNNImputer categorical data ke liye?

KNNImputer distance-based numerical calculations karta hai.

Therefore raw:

```text
City = Indore
Gender = Male
```

directly suitable nahi.

You'd need numerical representation.

Lekin blindly categorical values:

```text
Indore = 1
Bhopal = 2
Dewas = 3
```

kar dena bhi problematic ho sakta hai because KNN distance interpret karega:

```text
Dewas and Bhopal have numeric distance 1
```

which may have no semantic meaning.

So practical beginner approach:

```text
Numerical features
→ KNNImputer consider

Categorical features
→ SimpleImputer(most_frequent / constant)
```

Later `ColumnTransformer` isko cleanly handle karega.

---

# 37. Scaling + KNN: practical architecture

Suppose numerical data:

```text
Age
Salary
Experience
```

Different ranges.

Conceptually you may want:

```text
Numerical columns
↓
appropriate scaling / imputation design
↓
KNN distance becomes more meaningful
```

Pipeline design thoda subtle hota hai because missing values and scaling interact karte hain.

Isliye real project me blindly fixed order copy karne ki jagah validation karna chahiye.

Official sklearn example explicitly scaling + KNN imputation pipeline demonstrate karta hai for differently scaled housing features. ([scikit-learn][4])

---

# 38. Simple vs KNN example

Suppose:

```text
Age  Experience  Salary

20       1        25000
22       2        30000
45      20       100000
48      23       120000
21       2        NaN
47      21        NaN
```

Simple median might give both:

```text
65000
```

approximately.

So:

```text
Age 21 → Salary 65000
Age 47 → Salary 65000
```

Not very personalized.

KNN could produce something more like:

```text
Age 21 → around 27500
Age 47 → around 110000
```

because their neighbors differ.

That's the intuition behind multivariate imputation.

---

# 39. Iterative example

Same dataset:

```text
Age + Experience
```

strongly related to:

```text
Salary
```

If relationship can be modeled:

```text
Age
Experience
↓
regression model
↓
Salary
```

then IterativeImputer can exploit that relationship.

It isn't necessarily using nearest samples like KNN.

---

# 40. Which one should you choose?

A useful starting framework:

| Situation                                      | Starting choice                         |
| ---------------------------------------------- | --------------------------------------- |
| Need simple reliable baseline                  | `SimpleImputer`                         |
| Numerical features and similar rows meaningful | `KNNImputer`                            |
| Strong relationships among numerical features  | `IterativeImputer`                      |
| Missingness itself may matter                  | `add_indicator=True`                    |
| Categorical values missing                     | `SimpleImputer(most_frequent/constant)` |

But final choice ideally model performance through proper validation se karni chahiye.

---

# 41. Don't choose based only on imputed values

Suppose:

```text
SimpleImputer
KNNImputer
IterativeImputer
```

teen methods available hain.

You might think:

> Kaunsi missing value sabse realistic dikhti hai?

That's useful but final ML goal usually downstream prediction performance hai.

Better:

```text
Imputer A + model
↓
Cross-validation score

Imputer B + model
↓
Cross-validation score

Imputer C + model
↓
Cross-validation score
```

Then compare.

Sklearn ka own example multiple imputation strategies ko downstream estimator performance ke through compare karta hai. ([scikit-learn][4])

---

# 42. Data leakage rule again

Never:

```python
imputer.fit_transform(X)
```

then split.

Correct:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Then:

```python
imputer.fit(X_train)

X_train_new = imputer.transform(X_train)

X_test_new = imputer.transform(X_test)
```

or shortcut:

```python
X_train_new = imputer.fit_transform(X_train)

X_test_new = imputer.transform(X_test)
```

Applies to:

```text
SimpleImputer
KNNImputer
IterativeImputer
```

all.

---

# 43. Complete KNN example

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

df = pd.DataFrame({
    "Age": [
        20, 22, 25, 30,
        35, 40, 45, 50
    ],
    "Experience": [
        1, 2, 3, 5,
        np.nan, 12, 18, 22
    ],
    "Salary": [
        25000, 30000, np.nan, 45000,
        55000, 70000, 90000, np.nan
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
imputer = KNNImputer(
    n_neighbors=3,
    weights="distance"
)
```

Training:

```python
X_train_new = imputer.fit_transform(
    X_train
)
```

Testing:

```python
X_test_new = imputer.transform(
    X_test
)
```

Then these transformed values can go into your model.

---

# 44. Complete Iterative example

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

Create:

```python
imputer = IterativeImputer(
    max_iter=10,
    random_state=42
)
```

Training:

```python
X_train_new = imputer.fit_transform(
    X_train
)
```

Testing:

```python
X_test_new = imputer.transform(
    X_test
)
```

Then:

```python
model.fit(
    X_train_new,
    y_train
)
```

and:

```python
y_pred = model.predict(
    X_test_new
)
```

---

# 45. `add_indicator=True` complete example

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy="median",
    add_indicator=True
)

X_train_new = imputer.fit_transform(
    X_train
)

X_test_new = imputer.transform(
    X_test
)
```

Suppose original X has:

```text
3 columns
```

and 2 columns had missing values during training.

Output may have:

```text
3 imputed features
+
2 indicator features
=
5 features
```

So output shape can increase.

---

# 46. Advanced imputation always use karni chahiye?

No.

Suppose:

```text
Age
```

only 1% missing hai.

Median imputation may work perfectly fine.

Using:

```text
IterativeImputer
```

could introduce:

```text
more complexity
more computation
more tuning
```

without meaningful improvement.

Best learning habit:

```text
Start simple
↓
Create baseline
↓
Try advanced approach
↓
Validate whether it actually improves
```

---

# 47. Chapter 9 ka core mental model

```text
SimpleImputer
↓
One column's statistic
↓
mean / median / mode
```

```text
KNNImputer
↓
Similar rows
↓
nearest neighbors
↓
missing value estimate
```

```text
IterativeImputer
↓
Other features
↓
predictive models
↓
missing value estimate
↓
repeat iteratively
```

```text
MissingIndicator
↓
Missing value fill nahi
↓
"was missing?" information
```

---

# 48. Most important comparison

| Feature                           | Simple |      KNN |    Iterative |
| --------------------------------- | -----: | -------: | -----------: |
| Easy                              |      ✅ |   Medium | More complex |
| Fast                              |      ✅ |     Less |         Less |
| Uses same column statistics       |      ✅ |        — |            — |
| Uses other features               |      ❌ |        ✅ |            ✅ |
| Uses neighbor similarity          |      ❌ |        ✅ |            ❌ |
| Uses prediction models internally |      ❌ |        ❌ |            ✅ |
| Good baseline                     |      ✅ | Possible |     Possible |
| Scaling may matter strongly       |   Less |        ✅ |   Can matter |

---

# Chapter 9 Summary

`KNNImputer`:

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(
    n_neighbors=5
)
```

Concept:

```text
Nearest similar samples
↓
missing value estimate
```

`IterativeImputer` in current stable sklearn:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(
    random_state=42
)
```

Concept:

```text
One missing feature
↓
other features se predict
↓
repeat for features
```

`MissingIndicator`:

```python
from sklearn.impute import MissingIndicator
```

Concept:

```text
Missing?
↓
True / False
```

And all imputers ke liye golden rule:

```text
X_train
↓
fit_transform()

X_test
↓
transform()
```

**Do not assume KNN/Iterative is automatically better than SimpleImputer.** Start with a simple baseline, then compare advanced methods through proper validation.

### Quick test

Suppose:

```text
Age   Experience   Salary

22       2         30000
24       3         35000
23       2         NaN
50      25        150000
```

Think:

**Q1.** `SimpleImputer(strategy="mean")` Salary fill karte waqt Age aur Experience dekhega?

**Q2.** `KNNImputer` Age/Experience ki similarity use kar sakta hai?

**Q3.** `IterativeImputer` Salary ko temporary target bana kar other features se predict kar sakta hai?

**Q4.** `MissingIndicator` missing Salary ko fill karega ya sirf batayega ki value missing thi?

**Q5.** KNN me Age range `20–50` aur Salary range `20,000–2,00,000` ho, to scaling important kyon ho sakti hai?

