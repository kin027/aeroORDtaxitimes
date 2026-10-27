# ======================================================================
# STEP 0: Imports and settings
# ======================================================================
import os, sys, time
import pandas as pd
import requests

from foldse_client import Foldse, split_data, get_scores

SERVER = "https://foldse-server.onrender.com"
RAW_FILE = "data/cleaned_otp.csv"  # Cleaned source dataset
CSV_FILE = "prepared/final_otp.csv"  # Clean CSV created in step 1
LABEL = "long_taxi_in"  # Target column name
POS_VAL = "True"  # Positive class to describe (taxi_in > 14m)
RATIO = 0.5  # Same meaning as in FOLD-R++
TAIL = 5e-3  # Stopping / pruning threshold
SAMPLE_SIZE = 150000  # Prevents Render 502 Bad Gateway / OOM


# ======================================================================
# STEP 1: Prepare the OTP data
# ======================================================================
def prepare_otp():
    df = pd.read_csv(RAW_FILE)
    df.columns = df.columns.str.strip().str.lower()
    label_col = LABEL.strip().lower()

    # Ensure native boolean
    df[label_col] = df[label_col].astype(bool)

    # Take a stratified sample of the data
    if len(df) > SAMPLE_SIZE:
        print(
            f"Taking a stratified sample of {SAMPLE_SIZE} rows from {len(df)} for FOLD-SE"
        )
        frac = SAMPLE_SIZE / len(df)
        df = (
            df.groupby(label_col, group_keys=False)
            .sample(frac=frac, random_state=42)
            .reset_index(drop=True)
        )

    # Shuffle data so False and True are mixed throughout the file
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    categorical_cols = ["day_of_week", "op_carrier", "month"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    features = [c for c in df.columns if c != label_col]
    str_attrs = [c for c in features if c in categorical_cols]
    num_attrs = [c for c in features if c not in categorical_cols]

    os.makedirs(os.path.dirname(CSV_FILE) or ".", exist_ok=True)
    df[str_attrs + num_attrs + [label_col]].to_csv(CSV_FILE, index=False)

    print(f"rows={len(df)}  categorical={len(str_attrs)}  numeric={len(num_attrs)}")
    print("label counts:", df[label_col].value_counts().to_dict())
    return str_attrs, num_attrs


# ======================================================================
# STEP 2: Load the CSV and split train / test
# ======================================================================
def load_and_split(model):
    """Parse the CSV locally and split 80/20."""
    data = model.load_data(CSV_FILE)
    train, test = split_data(data, ratio=0.8, rand=True)
    print(f"train={len(train)}  test={len(test)}")
    print("first row:", {k: train[0][k] for k in list(train[0])[:5]}, "...")
    return train, test


# ======================================================================
# STEP 3: Wake up the hosted server
# ======================================================================
def wake_server():
    """Ping /health until the hosted service answers."""
    for attempt in range(5):
        try:
            r = requests.get(SERVER + "/health", timeout=60)
            print("server:", r.json())
            return
        except Exception as e:
            print(f"  server not ready ({type(e).__name__}), retrying...")
            time.sleep(10)
    print("server did not respond; fit() may fail")


# ======================================================================
# STEP 4: Train on the server
# ======================================================================
def train_model(model, train):
    """Send the (obfuscated + encrypted) data to the server and get the rules back."""
    for attempt in range(3):
        try:
            t = time.time()
            model.fit(train, ratio=RATIO, tail=TAIL)
            print(
                f"done in {time.time() - t:.1f}s wall  (server training: {model.training_seconds:.1f}s)"
            )
            return True
        except Exception as e:
            print("fit failed, retrying:", repr(e))
            time.sleep(20)
    print("training failed")
    return False


# ======================================================================
# STEP 5: Print the learned rules
# ======================================================================
def show_rules(model):
    """Print the model as a logic program."""
    print("% learned rules")
    for rule in model.asp():
        print(rule)
    print("training_metrics:", model.training_metrics)


# ======================================================================
# STEP 6: Evaluate on the test set
# ======================================================================
def evaluate_model(model, test):
    """Accuracy / precision / recall / F1 on the held-out rows."""
    raw_preds = model.predict(test)
    raw_truth = [x["label"] for x in test]

    # Convert predictions and ground truth strictly to list[bool] for get_scores
    preds = [
        bool(p) if not isinstance(p, str) else p.strip().lower() in ("true", "1")
        for p in raw_preds
    ]
    truth = [
        bool(t) if not isinstance(t, str) else t.strip().lower() in ("true", "1")
        for t in raw_truth
    ]

    acc, prec, rec, f1 = get_scores(preds, truth)
    print(f"accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")
    return acc


# ======================================================================
# STEP 7: Run everything
# ======================================================================
def main():
    str_attrs, num_attrs = prepare_otp()  # step 1
    model = Foldse(str_attrs, num_attrs, LABEL, POS_VAL)
    train, test = load_and_split(model)  # step 2
    print("\n--- server ---")
    wake_server()  # step 3
    print("\n--- training ---")  # step 4
    if not train_model(model, train):
        return
    print()
    show_rules(model)  # step 5
    print("\n--- evaluation ---")
    evaluate_model(model, test)  # step 6


main()
