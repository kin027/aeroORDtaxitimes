# FOLD-R++ and FOLD-SE: a step-by-step code walkthrough (Adult dataset)

Two explainable rule-learning algorithms, run on the UCI **Adult** census dataset (predict whether income is `<=50K` or `>50K`). Both learn a small set of **default rules with exceptions** instead of a black-box model.

Each algorithm is split into **numbered steps**. Every step has a name, a short description and a code snippet that is one function. The intended way to use this:

1. Create an empty file (`my_foldrpp.py`) in this folder.
2. Paste the snippets **in order**, one step at a time, and talk through each one.
3. The last step (`main`) calls all the functions. Run `python my_foldrpp.py`.

The finished scripts are included for reference: [foldrpp_walkthrough.py](foldrpp_walkthrough.py) and [foldse_walkthrough.py](foldse_walkthrough.py).

## Folder contents

```
fold_walkthrough/
├── README.md                 this file
├── requirements.txt          pandas, requests, cryptography
├── data/adult.data           raw UCI Adult file (no header)
├── FOLD-R-PP/foldrpp.py      FOLD-R++ implementation (plain Python, from github.com/hwd404/FOLD-R-PP)
├── foldse_client/            client for the hosted FOLD-SE service (from github.com/hwd404/foldse_client)
├── foldrpp_walkthrough.py    finished FOLD-R++ script (all steps assembled)
├── foldse_walkthrough.py     finished FOLD-SE script (all steps assembled)
└── prepared/                 created when you run: cleaned CSV + saved rule files
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

- **FOLD-R++** needs only `pandas` (for step 1). The algorithm itself is plain Python.
- **FOLD-SE** additionally needs `requests` and `cryptography` and **internet access**, because training runs on the hosted server `https://foldse-server.onrender.com`. The free server may be asleep; step 3 wakes it up.

Run everything from inside this folder.

## The Adult dataset

| | |
|---|---|
| Source | https://archive.ics.uci.edu/dataset/2/adult |
| Rows | 32,561 (1994 US Census) |
| Features | 13 after dropping `education`: 7 categorical (`workclass`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`) and 6 numeric (`age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`) |
| Label | income `<=50K` (24,720 rows, the positive class) vs `>50K` (7,841 rows) |
| Split | first 80 % train (26,048), last 20 % test (6,513), no shuffling |

## How the two algorithms differ

| | FOLD-R++ | FOLD-SE |
|---|---|---|
| Where it runs | locally | training on a hosted server, everything else local |
| Literal selection | information gain | Gini impurity |
| Goal | accurate default rules | same accuracy with far **fewer rules and literals** (scalable explainability) |
| Task | binary | binary (`Foldse`) and multi-class (`Foldsem`) |
| Data privacy | n/a | names and values obfuscated and encrypted before upload |

Both output the same kind of model: a logic program such as `label(X,'<=50K') :- not marital_status(X,'Married-civ-spouse'), capital_gain(X,N), N=<6849.`

---

# Part A: FOLD-R++

Script name: `foldrpp_walkthrough.py`. Steps 0 to 7, paste in order.

### Step 0: Imports and settings

Imports the algorithm from the copied repo (`FOLD-R-PP/foldrpp.py`, plain Python, no install) and sets the constants used everywhere.

- `POS_VAL` is the **positive class**. The learned rules describe *when a person is `<=50K`*. The FOLD-R++ README recommends using the majority class.
- `RATIO` is the main hyper-parameter (see step 3).

```python
import os, sys, time
import pandas as pd

sys.path.insert(0, 'FOLD-R-PP')  # lets us `import foldrpp` from the copied folder
from foldrpp import Foldrpp, split_data, get_scores, num_predicates, save_model_to_file

RAW_FILE = 'otp_tables/adult.data'  # original UCI file (no header)
CSV_FILE = 'prepared/adult.csv'  # clean CSV created in step 1
LABEL = 'label'  # name of the target column
POS_VAL = '<=50K'  # the class the rules will describe
RATIO = 0.5  # FOLD-R++ hyper-parameter (step 3)
```

### Step 1: Prepare the Adult data

Both algorithms want a **headered CSV**, and every column declared as *categorical* (only `==` / `!=` tests) or *numeric* (`=<` / `>` tests too). The raw UCI `adult.data` has no header, so this step:

- adds the 15 column names,
- drops `education` (it duplicates `education_num`),
- treats text columns as categorical, number columns as numeric,
- writes `prepared/adult.csv` (categorical columns first, label last).

No one-hot encoding or scaling is needed. Both algorithms handle mixed data directly.

```python
COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
           'hours_per_week', 'native_country', 'label']

def prepare_adult():
    """Raw adult.data -> clean CSV. Returns (categorical columns, numeric columns)."""
    df = pd.read_csv(RAW_FILE, names=COLUMNS, skipinitialspace=True)
    df = df.drop(columns=['education'])                       # duplicate of education_num

    features = [c for c in df.columns if c != LABEL]
    str_attrs = [c for c in features if df[c].dtype == object]        # text  -> categorical
    num_attrs = [c for c in features if c not in str_attrs]           # numbers -> numeric

    os.makedirs('prepared', exist_ok=True)
    df[str_attrs + num_attrs + [LABEL]].to_csv(CSV_FILE, index=False)

    print(f'rows={len(df)}  categorical={len(str_attrs)}  numeric={len(num_attrs)}')
    print('label counts:', df[LABEL].value_counts().to_dict())
    return str_attrs, num_attrs
```

### Step 2: Load the CSV and split train / test

`Foldrpp(str_attrs, num_attrs, label, pos_val)` is the model object. `load_data` reads the CSV into a list of dicts, one per row, e.g. `{'age': 39.0, 'workclass': 'State-gov', ..., 'label': 1}`. The label becomes `1` for `<=50K` and `0` for `>50K`.

The split is 80 % train / 20 % test in file order (`rand=False`), so results are repeatable.

```python
def load_and_split(model):
    """Read the CSV into dicts and split 80/20."""
    data = model.load_data(CSV_FILE)                    # list of {feature: value, ..., 'label': 0/1}
    train, test = split_data(data, ratio=0.8, rand=False)
    print(f'train={len(train)}  test={len(test)}')
    print('first row:', {k: train[0][k] for k in list(train[0])[:5]}, '...')
    return train, test
```

### Step 3: Train the full model

`fit` runs the complete algorithm:

1. **Build one rule, one literal at a time.** A literal is a single condition such as `capital_gain =< 6849`. For every feature the algorithm scores all candidate literals with an information-gain style heuristic (numeric thresholds are scored in one pass with prefix sums) and adds the best one. This repeats until the rows the rule still covers are mostly positive: covered negatives <= `RATIO` x covered positives. A lower `RATIO` gives purer rules with more literals, and a higher one gives looser, shorter rules.
2. **Learn exceptions.** The negatives the rule still covers are treated as a new learning problem with the roles flipped, using the same procedure. The result becomes the rule's exceptions (`ab` predicates).
3. **Remove and repeat.** Every positive row the rule covers is removed, and the loop continues until no positives are left.

The result is a short list of default rules with exceptions, an answer-set program.

```python
def train_model(model, train):
    """Learn the rule set."""
    t = time.time()
    model.fit(train, ratio=RATIO)
    print(f'trained in {time.time() - t:.2f}s')
```

### Step 4: Print the learned rules

`asp()` flattens the rules into named predicates and prints them as a logic program. Read a rule as *"head holds if all body literals hold and no exception (`ab`) holds"*. `not` is negation as failure and `ab1`, `ab2`, ... are the exception predicates defined below the main rules.

```python
def show_rules(model):
    """Print the model as a logic program."""
    print('% learned rules')
    for rule in model.asp():
        print(rule)
    print('#rules =', len(model.flat_rules), '  #predicates =', num_predicates(model.flat_rules))
```

### Step 5: Evaluate on the test set

`predict` runs every test row through the rules (locally). `get_scores` compares against the true labels and returns accuracy, precision, recall and F1, all with respect to the positive class `<=50K`.

```python
def evaluate_model(model, test):
    """Accuracy / precision / recall / F1 on the held-out rows."""
    preds = model.predict(test)                               # list of True/False
    truth = [x['label'] for x in test]                        # 1 / 0
    acc, prec, rec, f1 = get_scores(preds, truth)
    print(f'accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}')
    return acc
```

### Step 6: Save the model

Writes the rules (and their logic-program text) to JSON, so they can be reloaded later without retraining.

```python
def save_model(model):
    save_model_to_file(model, 'prepared/adult_foldrpp_rules.json')
    print('saved prepared/adult_foldrpp_rules.json')
```

### Step 7: Run everything

Wires the steps together. Paste this last, then run `python foldrpp_walkthrough.py`.

```python
def main():
    str_attrs, num_attrs = prepare_adult()                    # step 1
    model = Foldrpp(str_attrs, num_attrs, LABEL, POS_VAL)
    train, test = load_and_split(model)                       # step 2
    train_model(model, train)                                 # step 3
    print(); show_rules(model)                                # step 4
    print('\n--- evaluation ---'); evaluate_model(model, test)  # step 5
    save_model(model)                                         # step 6

if __name__ == '__main__':
    main()
```


---

# Part B: FOLD-SE

Script name: `foldse_walkthrough.py`. Steps 0 to 7, paste in order. Steps 1, 2, 5 and 6 are the same idea as in Part A. Steps 3 and 4 are specific to the hosted service.

### Step 0: Imports and settings

Imports the client (`foldse_client/`, copied next to the script). FOLD-SE **training runs on a hosted server**, not on your machine, so the client's job is to prepare, protect and send the data, and then to use the returned rules locally.

The client exposes the same API as FOLD-R++ (`load_data`, `fit`, `predict`, `asp`), so the steps below mirror the FOLD-R++ walkthrough.

- `TAIL` is a FOLD-SE hyper-parameter, sent to the server with `fit` (step 4).
- `RATIO` has the same meaning as in FOLD-R++.

```python
import os, sys, time
import pandas as pd
import requests

from foldse_client import Foldse, split_data, get_scores

SERVER = 'https://foldse-server.onrender.com'
RAW_FILE = 'otp_tables/adult.data'  # original UCI file (no header)
CSV_FILE = 'prepared/adult.csv'  # clean CSV created in step 1
LABEL = 'label'  # name of the target column
POS_VAL = '<=50K'  # the class the rules will describe
RATIO = 0.5  # same meaning as in FOLD-R++
TAIL = 5e-3  # FOLD-SE stopping / pruning threshold (client default)
```

### Step 1: Prepare the Adult data

Both algorithms want a **headered CSV**, and every column declared as *categorical* (only `==` / `!=` tests) or *numeric* (`=<` / `>` tests too). The raw UCI `adult.data` has no header, so this step:

- adds the 15 column names,
- drops `education` (it duplicates `education_num`),
- treats text columns as categorical, number columns as numeric,
- writes `prepared/adult.csv` (categorical columns first, label last).

No one-hot encoding or scaling is needed. Both algorithms handle mixed data directly.

```python
COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
           'hours_per_week', 'native_country', 'label']

def prepare_adult():
    """Raw adult.data -> clean CSV. Returns (categorical columns, numeric columns)."""
    df = pd.read_csv(RAW_FILE, names=COLUMNS, skipinitialspace=True)
    df = df.drop(columns=['education'])                       # duplicate of education_num

    features = [c for c in df.columns if c != LABEL]
    str_attrs = [c for c in features if df[c].dtype == object]        # text  -> categorical
    num_attrs = [c for c in features if c not in str_attrs]           # numbers -> numeric

    os.makedirs('prepared', exist_ok=True)
    df[str_attrs + num_attrs + [LABEL]].to_csv(CSV_FILE, index=False)

    print(f'rows={len(df)}  categorical={len(str_attrs)}  numeric={len(num_attrs)}')
    print('label counts:', df[LABEL].value_counts().to_dict())
    return str_attrs, num_attrs
```

### Step 2: Load the CSV and split train / test

`Foldse(str_attrs, num_attrs, label, pos_val)` is the model object. `load_data` parses the CSV **locally** into a list of dicts (same format as FOLD-R++, label `1` = `<=50K`). Same 80 / 20 file-order split.

```python
def load_and_split(model):
    """Parse the CSV locally and split 80/20."""
    data = model.load_data(CSV_FILE)
    train, test = split_data(data, ratio=0.8, rand=False)
    print(f'train={len(train)}  test={len(test)}')
    print('first row:', {k: train[0][k] for k in list(train[0])[:5]}, '...')
    return train, test
```

### Step 3: Wake up the hosted server

The free hosted service sleeps when idle and the first request can fail or be slow. A `/health` request wakes it up. Retry a few times if it does not answer right away.

```python
def wake_server():
    """Ping /health until the hosted service answers."""
    for attempt in range(5):
        try:
            r = requests.get(SERVER + '/health', timeout=60)
            print('server:', r.json())
            return
        except Exception as e:
            print(f'  server not ready ({type(e).__name__}), retrying...')
            time.sleep(10)
    print('server did not respond; fit() may fail')
```

### Step 4: Train on the server

`fit` obfuscates the training rows (feature names, category values and numbers are replaced with tokens or random transforms, so the server never sees the real data), encrypts them, POSTs them, and receives the learned rules, which it de-obfuscates locally. FOLD-SE differs from FOLD-R++ inside the server, so that code is not part of this client:

- **Gini impurity** picks the next literal, instead of FOLD-R++'s information gain.
- **Prefix sums** make scoring every numeric threshold fast.
- The goal is *scalable explainability*: the rule set stays small, in both the number of rules and the number of literals, as the data grows.

`training_seconds` is the server-side time. The whole call is wrapped in a retry, since the hosted service can be flaky.

```python
def train_model(model, train):
    """Send the (obfuscated + encrypted) data to the server and get the rules back."""
    for attempt in range(3):
        try:
            t = time.time()
            model.fit(train, ratio=RATIO, tail=TAIL)
            print(f'done in {time.time() - t:.1f}s wall  (server training: {model.training_seconds:.1f}s)')
            return True
        except Exception as e:
            print('fit failed, retrying:', repr(e))
            time.sleep(20)
    print('training failed')
    return False
```

### Step 5: Print the learned rules

Same logic-program format as FOLD-R++: `head :- literals, not ab.` The rules were de-obfuscated, so they use the real feature names and thresholds again. `training_metrics` also reports the rule and predicate counts, which is the key measure of FOLD-SE's compactness.

```python
def show_rules(model):
    """Print the model as a logic program."""
    print('% learned rules')
    for rule in model.asp():
        print(rule)
    print('training_metrics:', model.training_metrics)
```

### Step 6: Evaluate on the test set

Prediction runs locally from the returned rules. No server call is needed.

```python
def evaluate_model(model, test):
    """Accuracy / precision / recall / F1 on the held-out rows."""
    preds = model.predict(test)
    truth = [x['label'] for x in test]
    acc, prec, rec, f1 = get_scores(preds, truth)
    print(f'accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}')
    return acc
```

### Step 7: Run everything

Wires the steps together. Paste this last, then run `python foldse_walkthrough.py`.

```python
def main():
    str_attrs, num_attrs = prepare_adult()                    # step 1
    model = Foldse(str_attrs, num_attrs, LABEL, POS_VAL)
    train, test = load_and_split(model)                       # step 2
    print('\n--- server ---'); wake_server()                 # step 3
    print('\n--- training ---')                                # step 4
    if not train_model(model, train):
        return
    print(); show_rules(model)                                # step 5
    print('\n--- evaluation ---'); evaluate_model(model, test)  # step 6

if __name__ == '__main__':
    main()
```


---

## Results on Adult

Output of the two finished scripts (80/20 file-order split, `ratio=0.5`):

| | FOLD-R++ | FOLD-SE |
|---|---|---|
| Rules (main + exception) | 14 | 2 |
| Predicates (literals) | 46 | 5 |
| Accuracy | 0.8437 | 0.8439 |
| Precision | 0.8591 | 0.8591 |
| Recall | 0.9483 | 0.9485 |
| F1 | 0.9015 | 0.9016 |
| Training time | about 3.6 s (local) | about 9 s (server, plus network) |

Same accuracy, but FOLD-SE explains it with two rules instead of fourteen. FOLD-SE's whole model:

```
label(X,'<=50K') :- not marital_status(X,'Married-civ-spouse'), capital_gain(X,N1), N1=<6849.0.
label(X,'<=50K') :- marital_status(X,'Married-civ-spouse'), capital_gain(X,N1), N1=<5013.0, education_num(X,N2), N2=<12.0.
```

Timings depend on your machine and on whether the hosted server was asleep.


## References

- Wang & Gupta, *FOLD-R++: A Scalable Toolset for Automated Inductive Learning of Default Theories from Mixed Data*, FLOPS 2022.
- Wang & Gupta, *FOLD-SE: an efficient rule-based machine learning algorithm with scalable explainability*, PADL 2024.
- Code: https://github.com/hwd404/FOLD-R-PP and https://github.com/hwd404/foldse_client
