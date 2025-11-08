import os
import time
from tqdm import tqdm 
import numpy as np
import pandas as pd
import json, re
import requests

from utils.metrics import evaluate_all, save_metrics


INPUT_TSV = "./datasets/ASAP-SAS/train.tsv"
OUTPUT_CSV = "./outputs/ASAP-SAS/output_set2_multi_calls.csv"

PROMPT_DIR   = "./prompts/ASAP-SAS/DataSet2"
PROMPT_FILE_EXTRACTOR = f"{PROMPT_DIR}/prompt_extractor_agent.txt"
PROMPT_FILE_SCORER    = f"{PROMPT_DIR}/prompt_scoring_agent.txt"
QUESTION_FILE         = f"{PROMPT_DIR}/question.txt"
RUBRIC_FILE           = f"{PROMPT_DIR}/rubric.txt"

def read_text(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

PROMPT_EXTRACTOR = read_text(PROMPT_FILE_EXTRACTOR)
PROMPT_SCORING   = read_text(PROMPT_FILE_SCORER)
QUESTION_TEXT    = read_text(QUESTION_FILE)
RUBRIC_TEXT      = read_text(RUBRIC_FILE)

MODEL = "llama-70b"
TEST_ROWS   = None
SLEEP_S    = 0.2
WRITE_FEEDBACK = False

def parse_json(text, default):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        return json.loads(m.group(0)) if m else default

def load_prompt_template():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()

def norm_list(xs):
    return sorted({(x or "").strip().lower() for x in (xs or []) if x})


# ---------------- Agent 1: Extract ----------------
def agent1_extract(essay_text: str) -> str:
    user_prompt = PROMPT_EXTRACTOR.format(
        question=QUESTION_TEXT,
        rubric=RUBRIC_TEXT,
        essay_text=essay_text
    )
    print("\n--- Prompt ---")
    print(user_prompt)
    print("--------------")

    url = "http://127.0.0.1:8000/conversation/llama70b/"
    payload = {
        "messages": [
            {"role": "user", "content": user_prompt}
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
    resp_text = resp_text.replace("\n", " ")
    data = resp_text
    print("\n--- data ---")
    print(data)



    return data

# ---------------- Agent 2: Score ----------------
def agent2_score(essay_text: str, extraction: str) -> dict:
    user_prompt = PROMPT_SCORING.format(
        question=QUESTION_TEXT,
        rubric=RUBRIC_TEXT,
        essay_text=essay_text,
        # extraction_json=json.dumps(extraction, ensure_ascii=False)
        extraction_json=extraction
    )
    print("\n--- Prompt ---")
    print(user_prompt)
    print("--------------")

    url = "http://127.0.0.1:8000/conversation/llama70b/"
    payload = {
        "messages": [
            {"role": "user", "content": user_prompt}
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
    resp_text = resp_text.replace("\n", " ")
    match = re.search(r'\{\s*"score"\s*:\s*(\d+)\s*\}', resp_text)
    s = 0
    if match:
        try:
            s = int(match.group(1))
        except ValueError:
            s = 0

    print("\n--- score ---")
    print(s)
    print("--------------")
    return {"score": max(0, min(3, s)), "raw": resp_text}



# ---------- Agent 3: Feedback (optional) ----------
def agent3_feedback(essay_text: str, extraction: dict, scoring: dict) -> str:
    sys = "You are a supportive science teacher. Write concise, actionable feedback (<=80 words)."
    usr = f"""# Rubric (ref)
{RUBRIC_TEXT}

# Student Response
{essay_text}

# Agent1 Extraction
{json.dumps(extraction, ensure_ascii=False)}

# Agent2 Score
{json.dumps(scoring, ensure_ascii=False)}

Return only the feedback text (<=80 words)."""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.5,
        messages=[{"role": "system", "content": sys},{"role": "user", "content": usr}],
    )
    return resp.choices[0].message.content.strip()

def append_row_to_csv(row_dict: dict, csv_path: str, header_order: list):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    pd.DataFrame([row_dict], columns=header_order).to_csv(
        csv_path, mode="a", header=write_header, index=False, encoding="utf-8-sig"
    )

if __name__ == "__main__":
    df = pd.read_csv(INPUT_TSV, sep="\t", encoding="utf-8")
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

    OUT_COLS = ["Id", "EssayText", "Extraction", "PredScore", "GoldScore", "ScorerRaw", "Feedback"]

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Multi-agent"):
        rid = str(r["Id"])
        if rid in existing_ids:
            continue

        essay = str(r["EssayText"])
        a1 = agent1_extract(essay)       # str
        a2 = agent2_score(essay, a1)     # dict {score, raw}
        fb = agent3_feedback(essay, a1, a2) if WRITE_FEEDBACK else ""

        row_out = {
            "Id": rid,
            "EssayText": essay,
            "Extraction": a1,
            "PredScore": int(a2["score"]),
            "GoldScore": int(r["Score1"]),
            "ScorerRaw": a2["raw"],
            "Feedback": fb,
        }
        append_row_to_csv(row_out, OUTPUT_CSV, OUT_COLS)

        existing_ids.add(rid)
        time.sleep(SLEEP_S)

    print(f"Incremental predictions saved to {OUTPUT_CSV}")

    out = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
    if len(out) == 0:
        print("No rows available for evaluation yet.")
    else:
        y_true = out["GoldScore"].astype(int).tolist()
        y_pred = out["PredScore"].astype(int).tolist()
        m = evaluate_all(y_true, y_pred, max_rating=3)
        print(f"Rows evaluated: {len(out)}")
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
