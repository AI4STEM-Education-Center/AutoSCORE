import os
import time
from tqdm import tqdm 
import numpy as np
import pandas as pd
import json, re
from openai import OpenAI

from utils.metrics import evaluate_all, save_metrics

client = OpenAI(api_key="[Your API KEY]")

INPUT_TSV = "./datasets/ASAP-SAS/train.tsv"
OUTPUT_CSV = "./outputs/ASAP-SAS/output_set2_single_call.csv" 

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

MODEL = "gpt-4o"
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

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}],
    )
    data = parse_json(resp.choices[0].message.content.strip(), {"score": 0})
    try:
        s = int(data.get("score", 0))
    except Exception:
        s = 0
    return max(0, min(3, s))


if __name__ == "__main__":
    df = pd.read_csv(INPUT_TSV, sep="\t")
    df = df[df["EssaySet"] == 2].copy()
    if TEST_ROWS is not None:
        df = df.head(TEST_ROWS).copy()

    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring essays"):
        preds.append(score_once(str(row["EssayText"])))
        time.sleep(SLEEP_S)

    df_out = df[["EssayText", "Score1"]].copy()
    df_out["PredScore"] = preds
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved predictions to {OUTPUT_CSV}")

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
