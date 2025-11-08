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

MODEL = "gpt-4o"
TEST_ROWS   = 5
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
def agent1_extract(essay_text: str) -> dict:
    user_prompt = PROMPT_EXTRACTOR.format(
        question=QUESTION_TEXT,
        rubric=RUBRIC_TEXT,
        essay_text=essay_text
    )
    # print("\n--- Prompt ---")
    # print(user_prompt)
    # print("--------------")
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are Insight Extractor. Output JSON only; no prose."},
            {"role": "user",   "content": user_prompt},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    data["conclusions"]           = norm_list(data.get("conclusions"))
    data["design_improvements"]   = norm_list(data.get("design_improvements"))
    data["validity_improvements"] = norm_list(data.get("validity_improvements"))
    data["design_count"]          = int(len(data["design_improvements"]))
    data["validity_count"]        = int(len(data["validity_improvements"]))
    data["valid_conclusion"]      = bool(data.get("valid_conclusion"))
    return data

# ---------------- Agent 2: Score ----------------
def agent2_score(essay_text: str, extraction: dict) -> dict:
    user_prompt = PROMPT_SCORING.format(
        question=QUESTION_TEXT,
        rubric=RUBRIC_TEXT,
        essay_text=essay_text,
        extraction_json=json.dumps(extraction, ensure_ascii=False)
    )
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.8,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are Score Judge. Return JSON only."},
            {"role": "user",   "content": user_prompt},
        ],
    )
    got = json.loads(resp.choices[0].message.content)
    try:
        s = int(got.get("score", 0))
    except Exception:
        s = 0
    return {"score": max(0, min(3, s)), "raw": got}

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


if __name__ == "__main__":
    df = pd.read_csv(INPUT_TSV, sep="\t")
    df = df[df["EssaySet"] == 2].copy()
    if TEST_ROWS is not None:
        df = df.head(TEST_ROWS).copy()

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Multi-agent"):
        essay = str(r["EssayText"])
        a1 = agent1_extract(essay)
        a2 = agent2_score(essay, a1)
        fb = agent3_feedback(essay, a1, a2) if WRITE_FEEDBACK else ""

        rows.append({
            "EssayText": essay,
            "Extraction": json.dumps(a1, ensure_ascii=False),
            "PredScore": a2["score"],
            "ScorerRaw": json.dumps(a2["raw"], ensure_ascii=False),
            "Feedback": fb,
            "GoldScore": int(r["Score1"]),
        })
        time.sleep(SLEEP_S)

    out = pd.DataFrame(rows, columns=["EssayText","Extraction","PredScore","GoldScore","ScorerRaw","Feedback"])
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved to {OUTPUT_CSV}")

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
