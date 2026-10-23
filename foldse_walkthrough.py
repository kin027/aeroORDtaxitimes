"""FOLD-SE walkthrough on the UCI Adult dataset. Run from this folder: python foldse_walkthrough.py"""

# ======================================================================
# STEP 0: Imports and settings
# ======================================================================
import os, sys, time
import pandas as pd
import requests

from foldse_client import Foldse, split_data, get_scores

SERVER   = 'https://foldse-server.onrender.com'
RAW_FILE = 'otp_tables/adult.data'  # original UCI file (no header)
CSV_FILE = 'prepared/adult.csv'            # clean CSV created in step 1
LABEL    = 'label'                         # name of the target column
POS_VAL  = '<=50K'                         # the class the rules will describe
RATIO    = 0.5                             # same meaning as in FOLD-R++
TAIL     = 5e-3                            # FOLD-SE stopping / pruning threshold (client default)


# ======================================================================
# STEP 1: Prepare the Adult data
# ======================================================================
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


# ======================================================================
# STEP 2: Load the CSV and split train / test
# ======================================================================
def load_and_split(model):
    """Parse the CSV locally and split 80/20."""
    data = model.load_data(CSV_FILE)
    train, test = split_data(data, ratio=0.8, rand=False)
    print(f'train={len(train)}  test={len(test)}')
    print('first row:', {k: train[0][k] for k in list(train[0])[:5]}, '...')
    return train, test


# ======================================================================
# STEP 3: Wake up the hosted server
# ======================================================================
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


# ======================================================================
# STEP 4: Train on the server
# ======================================================================
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


# ======================================================================
# STEP 5: Print the learned rules
# ======================================================================
def show_rules(model):
    """Print the model as a logic program."""
    print('% learned rules')
    for rule in model.asp():
        print(rule)
    print('training_metrics:', model.training_metrics)


# ======================================================================
# STEP 6: Evaluate on the test set
# ======================================================================
def evaluate_model(model, test):
    """Accuracy / precision / recall / F1 on the held-out rows."""
    preds = model.predict(test)
    truth = [x['label'] for x in test]
    acc, prec, rec, f1 = get_scores(preds, truth)
    print(f'accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}')
    return acc


# ======================================================================
# STEP 7: Run everything
# ======================================================================
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
