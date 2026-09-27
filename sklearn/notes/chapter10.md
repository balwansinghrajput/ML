# Chapter 10 — Categorical Encoding Deep Dive  
## `LabelEncoder`, `OrdinalEncoder`, `OneHotEncoder`

Ab hum Machine Learning preprocessing ke ek bahut important topic par aa gaye hain:

```text
Categorical Data
      ↓
Encoding
      ↓
Numbers
      ↓
ML Model
```

Is chapter ka sabse important goal hai ye confusion permanently clear karna:

```text
LabelEncoder  → mainly y / target
OrdinalEncoder → ordered categorical X/features
OneHotEncoder → nominal categorical X/features
```

Scikit-learn ki current documentation bhi `LabelEncoder` ko prediction target `y` ke labels ke liye describe karti hai, jabki `OrdinalEncoder` aur `OneHotEncoder` categorical input features ko encode karte hain. :chatgpt-content-reference{index="0"}

---

# 1. Categorical data kya hota hai?

Categorical data me values kisi **category/group** ko represent karti hain.

Example:

```text
City
Indore
Bhopal
Dewas
Indore
```

Yahan:

```text
Indore
Bhopal
Dewas
```

numbers nahi, categories hain.

Another example:

```text
Education

School
Bachelor
Master
PhD
```

Ye bhi categorical data hai.

---

# 2. ML model ko categorical data se problem kya hai?

Bahut saare ML algorithms numerical input ke saath kaam karte hain.

Example:

```text
Age = 25
Salary = 50000
City = "Indore"
```

`Age` aur `Salary` numbers hain.

Lekin:

```text
"Indore"
```

ko mathematical operations me directly use karna difficult hai.

Isliye:

```text
Indore
Bhopal
Dewas
```

ko numerical representation me convert karna padta hai.

Is process ko:

> **Categorical Encoding**

kehte hain.

---

# 3. Categorical data ke main types

Sabse pehle categorical data ke 2 major types samjho:

```text
Categorical Data
      ↓
 ┌──────────────┐
 ↓              ↓
Nominal       Ordinal
```

---

# 4. Nominal Data

Nominal categories me **natural order/ranking nahi hoti**.

Examples:

```text
City
→ Indore
→ Bhopal
→ Dewas
```

Kya:

```text
Indore > Bhopal?
```

No.

Kya:

```text
Dewas < Indore?
```

No meaningful ordering.

Other examples:

```text
Color
Red
Blue
Green

Blood Group
A
B
AB
O

Country
India
Japan
USA
```

These are **Nominal Categories**.

Usually:

```text
Nominal
→ OneHotEncoder
```

ek strong default hota hai.

---

# 5. Ordinal Data

Ordinal categories me meaningful order hota hai.

Example:

```text
Education Level

School
Bachelor
Master
PhD
```

There is an order:

```text
School
<
Bachelor
<
Master
<
PhD
```

Another:

```text
Size

Small
Medium
Large
```

Order:

```text
Small < Medium < Large
```

Another:

```text
Rating

Poor
Average
Good
Excellent
```

This is **Ordinal Data**.

Usually:

```text
Ordinal
→ OrdinalEncoder
```

---

# 6. Nominal vs Ordinal

| Data | Order? | Example | Common encoder |
|---|---:|---|---|
| Nominal | ❌ | City | OneHotEncoder |
| Ordinal | ✅ | Low/Medium/High | OrdinalEncoder |

Ye distinction bahut important hai.

---

# 7. Sabse pehle `LabelEncoder`

Import:

```python
from sklearn.preprocessing import LabelEncoder
```

Basic example:

```python
y = [
    "Fail",
    "Pass",
    "Pass",
    "Fail"
]
```

Create:

```python
encoder = LabelEncoder()
```

Fit + transform:

```python
y_encoded = encoder.fit_transform(y)

print(y_encoded)
```

Output could be:

```text
[0 1 1 0]
```

Scikit-learn `LabelEncoder` labels ko `0` se `n_classes - 1` range me normalize karta hai. :chatgpt-content-reference{index="1"}

---

# 8. `LabelEncoder` ne kya learn kiya?

Check:

```python
print(encoder.classes_)
```

Output:

```text
['Fail' 'Pass']
```

Mapping conceptually:

```text
Fail → 0
Pass → 1
```

Important:

```text
classes_
```

fit ke time learned attribute hai.

Chapter 2 yaad karo:

```text
underscore "_"
→ fitted/learned attribute
```

---

# 9. `inverse_transform()`

Encoded:

```text
0
1
1
0
```

wapas labels me:

```python
original = encoder.inverse_transform(
    [0, 1, 1, 0]
)

print(original)
```

Output:

```text
Fail
Pass
Pass
Fail
```

So:

```text
transform()
→ label → number

inverse_transform()
→ number → label
```

---

# 10. `LabelEncoder` ka main use

Sabse important rule:

> **`LabelEncoder` ko mainly target `y` ke labels encode karne ke liye use karo, input features `X` ke liye nahi.**

Scikit-learn ki docs explicitly target transformation section me `LabelEncoder` rakhti hain aur batati hain ki ye prediction target ke labels ke liye hai, features ke liye intended nahi. :chatgpt-content-reference{index="2"}

Example:

```python
X = df[
    ["Age", "Salary"]
]

y = df["Result"]
```

If:

```text
Result:
Pass
Fail
```

Then:

```python
encoder = LabelEncoder()

y = encoder.fit_transform(y)
```

reasonable hai.

---

# 11. Features par `LabelEncoder` kyon nahi?

Suppose:

```text
City

Indore
Bhopal
Dewas
```

Agar LabelEncoder lagaya:

```text
Bhopal → 0
Dewas  → 1
Indore → 2
```

Ab model numerical relation assume kar sakta hai:

```text
Indore (2) > Dewas (1) > Bhopal (0)
```

But actual City me:

```text
order hi nahi hai.
```

Worse:

```text
Indore = 2
Bhopal = 0
```

model kuch algorithms me numerical distance interpret kar sakta hai.

But city values ke beech:

```text
2 - 0 = 2
```

ka koi meaningful categorical meaning nahi.

Isi wajah se nominal feature ke liye OneHotEncoder better hota hai.

---

# 12. Another wrong example

Suppose:

```text
Color

Red
Green
Blue
```

Numbers assign:

```text
Blue  → 0
Green → 1
Red   → 2
```

Does this mean:

```text
Red > Green > Blue?
```

No.

Therefore:

```text
Nominal Feature
→ arbitrary integer codes se caution
```

---

# 13. Ab `OrdinalEncoder`

Import:

```python
from sklearn.preprocessing import OrdinalEncoder
```

`OrdinalEncoder` categorical input features ko integer representation me encode karta hai. Current sklearn docs me har feature ki categories integer codes me map hoti hain. :chatgpt-content-reference{index="3"}

Example:

```text
Size

Small
Medium
Large
```

We want:

```text
Small  → 0
Medium → 1
Large  → 2
```

---

# 14. OrdinalEncoder me order explicitly dena

Best practice jab real order pata hai:

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[
        ["Small", "Medium", "Large"]
    ]
)
```

Then:

```python
X_encoded = encoder.fit_transform(
    [["Small"], ["Large"], ["Medium"]]
)

print(X_encoded)
```

Output:

```text
[[0.]
 [2.]
 [1.]]
```

Mapping:

```text
Small  = 0
Medium = 1
Large  = 2
```

---

# 15. Order manually dena kyon important?

Suppose tum simply:

```python
encoder = OrdinalEncoder()
```

use karte ho.

Encoder categories training data se detect karega.

Lekin:

```text
Low
Medium
High
```

ka semantic order necessarily automatically nahi samjhega.

Isliye real ordinal variable ke liye:

```python
categories=[
    ["Low", "Medium", "High"]
]
```

explicitly dena safer hai.

---

# 16. Example — Education

Dataset:

```text
Education

School
Bachelor
Master
PhD
```

Code:

```python
encoder = OrdinalEncoder(
    categories=[
        [
            "School",
            "Bachelor",
            "Master",
            "PhD"
        ]
    ]
)
```

Then:

```text
School   → 0
Bachelor → 1
Master   → 2
PhD      → 3
```

Here numbers genuinely represent order.

---

# 17. Multiple ordinal columns

Suppose:

```text
Education
Size
```

Then:

```python
encoder = OrdinalEncoder(
    categories=[
        [
            "School",
            "Bachelor",
            "Master",
            "PhD"
        ],
        [
            "Small",
            "Medium",
            "Large"
        ]
    ]
)
```

Important:

```text
categories[0]
→ first feature

categories[1]
→ second feature
```

---

# 18. `categories_`

After fitting:

```python
encoder.fit(X_train)
```

Check:

```python
print(encoder.categories_)
```

This stores categories learned/used for each input feature. Current sklearn docs expose `categories_` for both OrdinalEncoder and OneHotEncoder. :chatgpt-content-reference{index="4"}

---

# 19. Unknown category problem

Suppose training:

```text
Small
Medium
Large
```

But test data:

```text
Extra Large
```

appears.

Default behavior may error because:

```text
"Extra Large"
```

training me seen nahi tha.

For OrdinalEncoder, we can use:

```python
encoder = OrdinalEncoder(
    categories=[
        ["Small", "Medium", "Large"]
    ],
    handle_unknown="use_encoded_value",
    unknown_value=-1
)
```

Then unseen:

```text
Extra Large
→ -1
```

Current OrdinalEncoder supports `handle_unknown="use_encoded_value"` with a configured `unknown_value`. :chatgpt-content-reference{index="5"}

---

# 20. Train/test rule yahan bhi same

Correct:

```python
X_train_encoded = encoder.fit_transform(
    X_train
)

X_test_encoded = encoder.transform(
    X_test
)
```

Wrong:

```python
X_test_encoded = encoder.fit_transform(
    X_test
)
```

Because:

```text
test data se categories learn nahi karni.
```

Golden rule still:

```text
TRAIN
→ fit_transform()

TEST
→ transform()
```

---

# 21. Ab sabse common encoder: `OneHotEncoder`

Import:

```python
from sklearn.preprocessing import OneHotEncoder
```

OneHotEncoder categorical feature ki har category ke liye separate binary column bana sakta hai. Current sklearn API categorical string/integer values ko one-hot numeric array me encode karta hai. :chatgpt-content-reference{index="6"}

Example:

```text
City

Indore
Dewas
Bhopal
```

One-hot:

```text
City_Bhopal   City_Dewas   City_Indore

0             0            1
0             1            0
1             0            0
```

---

# 22. OneHotEncoder ka main idea

Original:

```text
City = Indore
```

Instead of:

```text
Indore = 2
```

we create:

```text
Bhopal = 0
Dewas  = 0
Indore = 1
```

No artificial ranking.

This is why nominal categories ke liye OneHotEncoder useful hai.

---

# 23. Basic code

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
```

Training:

```python
encoder.fit(X_train)
```

Transform:

```python
X_train_encoded = encoder.transform(
    X_train
)
```

Shortcut:

```python
X_train_encoded = encoder.fit_transform(
    X_train
)
```

Test:

```python
X_test_encoded = encoder.transform(
    X_test
)
```

---

# 24. Dense output kaise mile?

Current sklearn me parameter:

```python
sparse_output=False
```

use kar sakte ho. Current stable `OneHotEncoder` ka default `sparse_output=True` hai. :chatgpt-content-reference{index="7"}

Example:

```python
encoder = OneHotEncoder(
    sparse_output=False
)
```

Then:

```python
X_encoded = encoder.fit_transform(X)

print(X_encoded)
```

Output easy-to-read dense array:

```text
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]
```

---

# 25. Sparse matrix kya hoti hai?

Suppose City has:

```text
1000 categories
```

OneHotEncoder 1000 columns bana sakta hai.

Har row me mostly:

```text
0 0 0 0 0 1 0 0 0...
```

bahut zeros honge.

Sparse representation:

```text
sirf important/non-zero values efficiently store karta hai
```

Isliye memory save ho sakti hai.

Current OneHotEncoder default sparse output return karta hai. :chatgpt-content-reference{index="8"}

Learning examples me:

```python
sparse_output=False
```

convenient hai.

Large real datasets me sparse representation useful ho sakti hai.

---

# 26. `get_feature_names_out()`

OneHotEncoder ke baad column names dekh sakte ho:

```python
encoder.get_feature_names_out(
    ["City"]
)
```

Could give:

```text
City_Bhopal
City_Dewas
City_Indore
```

Current API `get_feature_names_out()` ke through generated feature names expose karti hai. :chatgpt-content-reference{index="9"}

---

# 27. Complete OneHot example

```python
import pandas as pd

df = pd.DataFrame({
    "City": [
        "Indore",
        "Dewas",
        "Bhopal",
        "Indore"
    ]
})
```

Encoder:

```python
encoder = OneHotEncoder(
    sparse_output=False
)

encoded = encoder.fit_transform(
    df[["City"]]
)
```

Names:

```python
columns = encoder.get_feature_names_out(
    ["City"]
)
```

Convert DataFrame:

```python
encoded_df = pd.DataFrame(
    encoded,
    columns=columns
)

print(encoded_df)
```

Conceptually:

```text
City_Bhopal  City_Dewas  City_Indore

0            0           1
0            1           0
1            0           0
0            0           1
```

---

# 28. Unknown categories with OneHotEncoder

Suppose training:

```text
Indore
Dewas
Bhopal
```

Test:

```text
Ujjain
```

Default `OneHotEncoder` currently:

```text
handle_unknown="error"
```

use karta hai. :chatgpt-content-reference{index="10"}

Production me common configuration:

```python
encoder = OneHotEncoder(
    handle_unknown="ignore"
)
```

Now unseen category transform karte waqt error nahi karegi. Current docs ke example me unknown values `handle_unknown="ignore"` ke saath corresponding one-hot columns me zeros produce karti hain. :chatgpt-content-reference{index="11"}

---

# 29. `handle_unknown="ignore"`

Example training categories:

```text
Bhopal
Dewas
Indore
```

Columns:

```text
Bhopal
Dewas
Indore
```

New:

```text
Ujjain
```

None of existing categories match.

So representation:

```text
0 0 0
```

ho sakti hai.

That's useful because model input column count stable rehta hai.

---

# 30. Why not refit encoder on test?

Wrong:

```python
train_encoded = encoder.fit_transform(
    X_train
)

test_encoded = encoder.fit_transform(
    X_test
)
```

Suppose train categories:

```text
Indore
Dewas
Bhopal
```

Train columns:

```text
Bhopal
Dewas
Indore
```

Test:

```text
Indore
Ujjain
```

Refit test gives:

```text
Indore
Ujjain
```

columns.

Now model expects 3 columns but test has 2 totally different category meanings.

Correct:

```python
train_encoded = encoder.fit_transform(
    X_train
)

test_encoded = encoder.transform(
    X_test
)
```

---

# 31. `drop="first"`

OneHotEncoder can one category per feature drop kar sakta hai:

```python
encoder = OneHotEncoder(
    drop="first"
)
```

Suppose:

```text
City:
Bhopal
Dewas
Indore
```

Normally:

```text
Bhopal
Dewas
Indore
```

3 columns.

With:

```text
drop="first"
```

maybe:

```text
Dewas
Indore
```

2 columns.

If both:

```text
0 0
```

then dropped category:

```text
Bhopal
```

implied hai.

Current docs note karti hain ki dropping can help in settings where perfectly collinear features cause problems, though category dropping representation ko asymmetric bhi bana sakta hai. :chatgpt-content-reference{index="12"}

---

# 32. Dummy Variable Trap ka basic intuition

Suppose:

```text
Male
Female
```

One-hot:

```text
Male Female

1    0
0    1
```

Notice:

```text
Female = 1 - Male
```

One column doosre se completely predictable hai.

Some linear-model settings me perfect multicollinearity issue ban sakti hai.

Then:

```python
OneHotEncoder(
    drop="first"
)
```

one column remove kar sakta hai.

But blindly every model ke liye drop karna mandatory nahi.

---

# 33. `drop="if_binary"`

Useful:

```python
encoder = OneHotEncoder(
    drop="if_binary"
)
```

Agar feature ke only 2 categories hain:

```text
Yes
No
```

one category drop ho jayegi.

Agar 4 categories hain:

```text
A
B
C
D
```

all retain ho sakti hain.

Current API `drop='if_binary'` support karti hai. :chatgpt-content-reference{index="13"}

---

# 34. OneHotEncoder ka drawback — Feature explosion

Suppose:

```text
Country
→ 200 categories
```

One-hot approximately:

```text
200 columns
```

Suppose:

```text
Product_ID
→ 100,000 categories
```

One-hot could create enormous number of columns.

This is called:

```text
High Cardinality
```

problem.

---

# 35. Cardinality kya hoti hai?

> **Ek categorical feature me kitni unique categories hain.**

Example:

```text
Gender
Male
Female
```

Cardinality:

```text
2
```

City:

```text
100 different cities
```

Cardinality:

```text
100
```

Product ID:

```text
1,000,000 unique values
```

Very high cardinality.

---

# 36. Rare categories

Current OneHotEncoder has:

```python
min_frequency
max_categories
```

parameters jo infrequent categories ko manage/group karne me help kar sakte hain. :chatgpt-content-reference{index="14"}

Example:

```python
encoder = OneHotEncoder(
    min_frequency=10
)
```

Categories jo fewer than configured frequency appear karti hain unhe infrequent handling me group kiya ja sakta hai.

Ye advanced topic hai, but high-cardinality datasets me useful.

---

# 37. `LabelEncoder` vs `OrdinalEncoder`

Bahut important difference.

### LabelEncoder

Input usually:

```text
y
```

Example:

```text
Target:

Dog
Cat
Dog
```

Output:

```text
1
0
1
```

### OrdinalEncoder

Input:

```text
X
```

Example:

```text
Education feature:

School
Bachelor
Master
PhD
```

Output:

```text
0
1
2
3
```

Main distinction:

```text
LabelEncoder
→ target labels

OrdinalEncoder
→ categorical features
```

---

# 38. `OrdinalEncoder` vs `OneHotEncoder`

Suppose feature:

```text
Size:
Small
Medium
Large
```

There is an order.

Use:

```text
OrdinalEncoder
```

Representation:

```text
Small = 0
Medium = 1
Large = 2
```

But City:

```text
Indore
Dewas
Bhopal
```

No order.

Use:

```text
OneHotEncoder
```

Representation:

```text
Bhopal Dewas Indore
0      0     1
```

---

# 39. Most important table

| Encoder | Usually applied to | Suitable for |
|---|---|---|
| `LabelEncoder` | `y` | Target labels |
| `OrdinalEncoder` | `X` | Ordered categorical features |
| `OneHotEncoder` | `X` | Unordered/nominal categorical features |

Is table ko yaad rakhna.

---

# 40. Example mixed dataset

Suppose:

| Age | City | Education | Purchased |
|---:|---|---|---:|
| 22 | Indore | School | 0 |
| 30 | Dewas | Bachelor | 0 |
| 40 | Bhopal | Master | 1 |
| 50 | Indore | PhD | 1 |

Target:

```text
Purchased
```

Features:

```text
Age
City
Education
```

Treatment:

```text
Age
→ numeric
→ no categorical encoding

City
→ nominal
→ OneHotEncoder

Education
→ ordinal
→ OrdinalEncoder

Purchased
→ target
→ maybe already 0/1
```

This is exactly the kind of dataset where later `ColumnTransformer` becomes powerful.

---

# 41. If target itself strings ho

Suppose:

```text
Purchased

Yes
No
Yes
No
```

Could use:

```python
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df["Purchased"]
)
```

Could become:

```text
No  → 0
Yes → 1
```

Many sklearn classifiers can also accept string class labels directly, so target encoding is not always mandatory. But `LabelEncoder` is the sklearn utility intended for target labels when numerical normalization is desired. :chatgpt-content-reference{index="15"}

---

# 42. OneHotEncoder target `y` par?

Normally:

```text
OneHotEncoder
→ feature encoding
```

If you specifically need one-hot style target labels, sklearn docs point to `LabelBinarizer` instead of OneHotEncoder for `y`. :chatgpt-content-reference{index="16"}

For standard sklearn classification:

```text
y = [0, 1, 2]
```

or string labels are often enough.

---

# 43. Missing values and encoding

Suppose:

```text
City

Indore
NaN
Dewas
```

A common clean workflow:

```text
Categorical column
↓
SimpleImputer
↓
OneHotEncoder
```

Example:

```text
NaN
↓
"Unknown"
↓
OneHotEncode
```

So:

```text
Imputation
before
Encoding
```

often makes sense.

Later Pipeline:

```text
SimpleImputer
↓
OneHotEncoder
```

automatically sequence karega.

---

# 44. Numerical and categorical preprocessing

Suppose:

```text
Age
Salary
City
Education
```

We want:

```text
Age
Salary
↓
Imputer
↓
Scaler
```

And:

```text
City
↓
Categorical Imputer
↓
OneHotEncoder
```

Education:

```text
Education
↓
Imputer
↓
OrdinalEncoder
```

Later:

```text
ColumnTransformer
```

ye sab ek hi system me combine karega.

---

# 45. Complete train/test OneHot example

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
```

Data:

```python
df = pd.DataFrame({
    "City": [
        "Indore",
        "Dewas",
        "Bhopal",
        "Indore",
        "Dewas",
        "Bhopal"
    ],
    "Purchased": [
        1, 0, 1, 1, 0, 0
    ]
})
```

Features/target:

```python
X = df[["City"]]

y = df["Purchased"]
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

Encoder:

```python
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

Training:

```python
X_train_encoded = encoder.fit_transform(
    X_train
)
```

Testing:

```python
X_test_encoded = encoder.transform(
    X_test
)
```

---

# 46. Check learned categories

```python
print(encoder.categories_)
```

Could output:

```text
[
    ['Bhopal', 'Dewas', 'Indore']
]
```

Then:

```python
print(
    encoder.get_feature_names_out()
)
```

Could produce names corresponding to these categories.

---

# 47. Complete Ordinal example

```python
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder

X = pd.DataFrame({
    "Size": [
        "Small",
        "Large",
        "Medium",
        "Small"
    ]
})
```

Encoder:

```python
encoder = OrdinalEncoder(
    categories=[
        ["Small", "Medium", "Large"]
    ]
)
```

Transform:

```python
X_encoded = encoder.fit_transform(X)

print(X_encoded)
```

Result:

```text
Small  → 0
Large  → 2
Medium → 1
Small  → 0
```

---

# 48. Ordinal unknown handling example

```python
encoder = OrdinalEncoder(
    categories=[
        ["Low", "Medium", "High"]
    ],
    handle_unknown="use_encoded_value",
    unknown_value=-1
)
```

Training:

```python
encoder.fit(
    [["Low"], ["Medium"], ["High"]]
)
```

New:

```python
encoder.transform(
    [["Very High"]]
)
```

Output:

```text
-1
```

because new category wasn't known.

---

# 49. Common mistake #1 — LabelEncoder on City

Avoid:

```python
le = LabelEncoder()

df["City"] = le.fit_transform(
    df["City"]
)
```

if City is nominal and the downstream model would treat those numbers as meaningful numeric order.

Prefer:

```python
OneHotEncoder()
```

for low/medium-cardinality nominal features in many standard workflows.

---

# 50. Common mistake #2 — OneHotEncoding ordinal data blindly

Suppose:

```text
Low
Medium
High
```

There is real order.

OneHot:

```text
Low Medium High
```

works mathematically but throws away explicit ordering information.

Ordinal encoding:

```text
0
1
2
```

can represent order more directly.

Which representation performs better can depend on model and problem, but semantic structure matters.

---

# 51. Common mistake #3 — Automatically assign ordinal numbers to nominal categories

Don't do:

```text
India = 1
Japan = 2
USA = 3
```

and think:

```text
USA > Japan > India
```

unless an actual meaningful ordering exists.

Numbers used for category IDs are not automatically meaningful measurements.

---

# 52. Common mistake #4 — Fit encoder before split

Wrong:

```python
X_encoded = encoder.fit_transform(X)

train_test_split(
    X_encoded,
    y
)
```

Encoder has seen categories from future test data.

Safer:

```text
Split
↓
fit encoder on train
↓
transform train
↓
transform test
```

Later Pipelines make this easier.

---

# 53. Common mistake #5 — Fit test encoder separately

Wrong:

```python
encoder.fit_transform(X_train)

encoder.fit_transform(X_test)
```

Correct:

```python
encoder.fit_transform(X_train)

encoder.transform(X_test)
```

Same fitted category mapping must be used.

---

# 54. Common mistake #6 — Unknown category ignore na karna

Production me new category aa sakti hai.

Example:

Training:

```text
Indore
Dewas
Bhopal
```

Production:

```text
Ujjain
```

If default:

```python
handle_unknown="error"
```

OneHotEncoder error de sakta hai. Current default indeed `"error"` hai. :chatgpt-content-reference{index="17"}

Often production pipelines me:

```python
handle_unknown="ignore"
```

safer hota hai.

---

# 55. Common mistake #7 — Feature explosion ignore karna

Suppose:

```text
User_ID
```

has:

```text
1,000,000 unique categories
```

OneHotEncoder:

```text
up to huge number of columns
```

create kar sakta hai.

At that point alternative encodings, grouping rare categories, hashing, target encoding, or dropping identifier-like columns may be more appropriate depending on the feature meaning.

---

# 56. Quick decision tree

Jab categorical column mile, ye questions poochho:

```text
Is this target y?
        ↓
       Yes
        ↓
LabelEncoder may be used
```

If feature X:

```text
Is there meaningful order?
      ↓           ↓
     Yes          No
      ↓            ↓
OrdinalEncoder   OneHotEncoder
```

Easy rule:

```text
TARGET
→ LabelEncoder

ORDERED FEATURE
→ OrdinalEncoder

UNORDERED FEATURE
→ OneHotEncoder
```

---

# 57. One subtle point — Tree models

Tree models kabhi integer-encoded categorical representations ke saath workable results de sakte hain because they split features differently from linear/distance-based models.

But arbitrary integer codes for nominal categories can still create artificial structure in the representation.

Isliye:

```text
Nominal feature
→ blindly ordinal encode
```

best general beginner rule nahi hai.

Choose encoding based on estimator + cardinality + meaning.

---

# 58. `fit`, `transform`, `fit_transform` recap with encoders

### OneHotEncoder

```python
encoder.fit(X_train)
```

means:

```text
categories learn
```

```python
encoder.transform(X_test)
```

means:

```text
learned categories ke according encode
```

```python
encoder.fit_transform(X_train)
```

means:

```text
categories learn
+
training data encode
```

Same concepts from Chapters 2–4.

---

# 59. `inverse_transform()`

Both OneHotEncoder and OrdinalEncoder can provide inverse transformation for supported encoded inputs. :chatgpt-content-reference{index="18"}

Example:

```python
encoded = encoder.fit_transform(
    [["Small"], ["Large"]]
)

original = encoder.inverse_transform(
    encoded
)
```

Can return:

```text
Small
Large
```

Useful for debugging/interpretation.

---

# 60. Chapter 10 final mental model

Suppose dataset:

```text
Age     City      Education      Result
25      Indore    Bachelor       Pass
30      Dewas     Master         Fail
40      Bhopal    PhD            Pass
```

Understand it like:

```text
Age
→ numerical
→ no categorical encoding

City
→ nominal categorical
→ OneHotEncoder

Education
→ ordinal categorical
→ OrdinalEncoder

Result
→ target y
→ LabelEncoder if needed
```

Full conceptual transformation:

```text
RAW DATA

Age  City    Education
25   Indore  Bachelor

      ↓

ENCODING

Age | City_Bhopal | City_Dewas | City_Indore | Education
25  |      0      |      0     |      1      |     1

      ↓

ML MODEL
```

---

# Chapter 10 Summary

### `LabelEncoder`

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)
```

Use mainly:

```text
Target y
```

---

### `OrdinalEncoder`

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[
        ["Low", "Medium", "High"]
    ]
)
```

Use:

```text
Ordered categorical X
```

---

### `OneHotEncoder`

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

Use:

```text
Nominal/unordered categorical X
```

---

## Golden table

| Example | Type | Encoder |
|---|---|---|
| Pass / Fail target | Target | `LabelEncoder` |
| Small / Medium / Large | Ordinal feature | `OrdinalEncoder` |
| Indore / Dewas / Bhopal | Nominal feature | `OneHotEncoder` |
| Low / Medium / High | Ordinal feature | `OrdinalEncoder` |
| Red / Green / Blue | Nominal feature | `OneHotEncoder` |

And same train/test rule:

```python
X_train_encoded = encoder.fit_transform(
    X_train
)

X_test_encoded = encoder.transform(
    X_test
)
```

## Quick practice

Dataset:

```text
Age   City      Education   Purchased
22    Indore    School      No
30    Dewas     Bachelor    No
40    Bhopal    Master      Yes
50    Ujjain    PhD         Yes
```

Think:

1. `Age` ko encoder chahiye?
2. `City` → LabelEncoder, OrdinalEncoder ya OneHotEncoder?
3. `Education` → kaunsa encoder?
4. `Purchased` agar target `y` hai aur values `"Yes"/"No"` hain, to kaunsa encoder use kar sakte ho?
5. Test data me new city `"Ratlam"` aa jaye to `handle_unknown="ignore"` ka kya role hoga?

