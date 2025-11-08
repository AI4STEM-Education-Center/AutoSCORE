import os
import time
from tqdm import tqdm 
import numpy as np
import pandas as pd
import json, re
import requests

from utils.metrics import evaluate_all, save_metrics


INPUT_TSV = "./datasets/ASAP-SAS/train.tsv"
OUTPUT_CSV = "./outputs/ASAP-SAS/output_set2_single_call_1.csv" 

PROMPT_DIR   = "./prompts/ASAP-SAS/DataSet2"
PROMPT_FILE  = f"{PROMPT_DIR}/prompt_single_agent.txt"
QUESTION_FILE = f"{PROMPT_DIR}/question.txt"
RUBRIC_FILE   = f"{PROMPT_DIR}/rubric.txt"

def read_text(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

PROMPT_TEMPLATE = read_text(PROMPT_FILE)
QUESTION_TEXT   = read_text(QUESTION_FILE)
RUBRIC_TEXT     = read_text(RUBRIC_FILE)

MODEL = "llama-70b"
TEST_ROWS   = None
SLEEP_S    = 0.2

def parse_json(text, default):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        return json.loads(m.group(0)) if m else default

def load_prompt_template():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()

PROMPT_TEMPLATE = load_prompt_template()

def score_once(essay_text: str) -> int:
    prompt = PROMPT_TEMPLATE.format(
        question=QUESTION_TEXT,
        rubric=RUBRIC_TEXT,
        essay_text=essay_text
    )
    print("\n--- Prompt ---")
    print(prompt)
    print("--------------")
    url = "http://127.0.0.1:8000/conversation/llama70b/"
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "configs": {
            "maxlen": 1024,
            "temperature": 0.8,
        }
    }
    resp = requests.post(url, json=payload, timeout=300)
    obj = resp.json()
    print("\n--- obj ---")
    print(obj)
    resp_text = obj.get("response", "").strip()
    print("\n--- resp_text ---")
    print(resp_text)
    data = parse_json(resp_text, {"score": 0})



    try:
        s = int(data.get("score", 0))
    except Exception:
        s = 0
    return max(0, min(3, s))

def append_row_to_csv(row_dict: dict, csv_path: str, header_order: list):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    pd.DataFrame([row_dict], columns=header_order).to_csv(
        csv_path, mode="a", header=write_header, index=False, encoding="utf-8-sig"
    )



if __name__ == "__main__":
    df = pd.read_csv(INPUT_TSV, sep="\t")
    if "Id" not in df.columns:
        raise ValueError("input file lack 'Id' column, please ensure the data contains a unique identifier Id.")
    df = df[df["EssaySet"] == 2].copy()
    if TEST_ROWS is not None:
        df = df.head(TEST_ROWS).copy()

    existing_ids = set()
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        try:
            out_existing = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
            if "Id" in out_existing.columns:
                existing_ids = set(out_existing["Id"].astype(str).tolist())
        except Exception as e:
            print(f"Warning: failed to read existing OUTPUT_CSV ({e}), will recreate.")

    OUT_COLS = ["Id", "EssayText", "Score1", "PredScore"]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring essays"):
        rid = str(row["Id"])
        if rid in existing_ids:
            continue

        pred = score_once(str(row["EssayText"]))
        time.sleep(SLEEP_S)

        row_out = {
            "Id": rid,
            "EssayText": row["EssayText"],
            "Score1": int(row["Score1"]),
            "PredScore": int(pred),
        }
        append_row_to_csv(row_out, OUTPUT_CSV, OUT_COLS)
        existing_ids.add(rid)

    print(f"Incremental predictions saved to {OUTPUT_CSV}")

    df_out = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
    if len(df_out) == 0:
        print("No rows available for evaluation yet.")
    else:
        y_true = df_out["Score1"].astype(int).tolist()
        y_pred = df_out["PredScore"].astype(int).tolist()

        m = evaluate_all(y_true, y_pred, max_rating=3)
        print(f"Rows evaluated: {len(df_out)}")
        print(f"QWK:         {m['QWK']:.4f}")
        print(f"Pearson:     {m['Pearson']:.4f}")
        print(f"Spearman:    {m['Spearman']:.4f}")
        print(f"Accuracy:    {m['Accuracy']:.4f}")
        print(f"AdjAccuracy: {m['AdjAccuracy']:.4f}")
        print(f"MAE:         {m['MAE']:.4f}")
        print(f"CohenKappa:  {m['CohenKappa']:.4f}")

        metrics_path = OUTPUT_CSV.replace(".csv", "_metrics.json")
        save_metrics(metrics_path, m)
        print(f"Saved metrics to {metrics_path}")
