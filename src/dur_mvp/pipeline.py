"""CLI-wrapped pipeline extracted from the notebook (refined).

- Consolidated imports at top
- Wraps imperative cells inside run()
- Provides argparse-based CLI with typed parameters
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Any

# Consolidated imports detected in notebook:
import os, glob, pandas as pd
import re
import math, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
import pandas as pd
import ast, time
from itertools import zip_longest
from tqdm import tqdm
import ast, time, gc
from dataclasses import dataclass, field
from functools import lru_cache
from kaggle_secrets import UserSecretsClient
from openai import OpenAI

# If the notebook defined functions/classes, they live in utils
try:
    from .utils import *  # noqa: F401,F403
except Exception as e:
    logging.getLogger(__name__).warning("Could not import utils: %s", e)

log = logging.getLogger("dur_mvp.pipeline")

def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a chosen level."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log.setLevel(numeric)

def run(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    config: Optional[Path] = None,
    save_figures: bool = False,
    seed: Optional[int] = None,
    dry_run: bool = False,
    **kwargs: Any,
) -> None:
    """Execute the DUR MVP pipeline.

    Args:
        input_path: Optional path to input file or folder (CSV/Parquet/etc.).
        output_path: Optional destination folder for outputs (created if missing).
        config: Optional path to a configuration file.
        save_figures: If True, save plots to output_path / "figures".
        seed: Optional random seed for reproducibility.
        dry_run: If True, do not write outputs; just exercise the code paths.
        **kwargs: Reserved for future extensions.
    """
    log.info("Starting pipeline")
    if input_path:
        input_path = Path(input_path)
        log.info("Input path: %s", input_path.resolve())
    if output_path:
        output_path = Path(output_path)
        if not dry_run:
            output_path.mkdir(parents=True, exist_ok=True)
        log.info("Output path: %s", output_path.resolve())
    if config:
        config = Path(config)
        log.info("Config path: %s", config.resolve())

    # ---- from notebook cell 1 ----
    import os, glob, pandas as pd
    import re
    import math, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
    pd.set_option('display.max_colwidth', None)   # do not cut off long text in any column
    pd.set_option('display.width', None)          # allow very wide tables
    pd.set_option('display.max_columns', None)    # show all columns

    import pandas as pd
    import ast, time
    from itertools import zip_longest
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import ast, time, gc
    from dataclasses import dataclass, field

    from functools import lru_cache
    from kaggle_secrets import UserSecretsClient
    from openai import OpenAI
    # ---- from notebook cell 4 ----
    BASE = "/kaggle/input/one-sides/onsides_v3/csv"  # adjust if needed

    # --- Load the 7 ON-SIDES tables you listed ---
    pl  = pd.read_csv(f"{BASE}/product_label.csv", dtype=str)                       # ['label_id','source',...]
    ae  = pd.read_csv(f"{BASE}/product_adverse_effect.csv", dtype=str)              # ['product_label_id','effect_id','label_section','effect_meddra_id',...]
    p2p = pd.read_csv(f"{BASE}/product_to_rxnorm.csv", dtype=str)                   # ['label_id','rxnorm_product_id']
    vrp = pd.read_csv(f"{BASE}/vocab_rxnorm_product.csv", dtype=str)                # ['rxnorm_id','rxnorm_name','rxnorm_term_type']
    i2p = pd.read_csv(f"{BASE}/vocab_rxnorm_ingredient_to_product.csv", dtype=str)  # ['product_id','ingredient_id']
    ing = pd.read_csv(f"{BASE}/vocab_rxnorm_ingredient.csv", dtype=str)             # ['rxnorm_id','rxnorm_name','rxnorm_term_type']
    med = pd.read_csv(f"{BASE}/vocab_meddra_adverse_effect.csv", dtype=str)         # ['meddra_id','meddra_name','meddra_term_type']

    # --- Build LABEL -> INGREDIENT mapping (many-to-many) ---
    # product_label (label_id) -> product_to_rxnorm (rxnorm_product_id) -> ingredient_to_product (ingredient_id)
    lab_to_ing = (
        pl[['label_id']]
        .merge(p2p, on='label_id', how='left')                          # to rxnorm_product_id
        .merge(i2p, left_on='rxnorm_product_id', right_on='product_id', how='left')  # to ingredient_id
        [['label_id','ingredient_id']]
        .dropna()
        .drop_duplicates()
    )

    # --- Total labels per ingredient (denominator) ---
    total_labels = (
        lab_to_ing
        .groupby('ingredient_id', as_index=False)['label_id']
        .nunique()
        .rename(columns={'label_id':'total_labels'})
    )

    # --- Labels-with-event per ingredient & effect (numerator) ---
    # ae uses 'product_label_id' for the label key
    ae_min = ae[['product_label_id','effect_meddra_id']].dropna().drop_duplicates()
    lab_evt_ing = (
        ae_min
        .merge(lab_to_ing, left_on='product_label_id', right_on='label_id', how='inner')
        [['label_id','ingredient_id','effect_meddra_id']]
        .drop_duplicates()
    )

    num_labels_with_event = (
        lab_evt_ing
        .groupby(['ingredient_id','effect_meddra_id'], as_index=False)['label_id']
        .nunique()
        .rename(columns={'label_id':'num_labels_with_event'})
    )

    # --- Attach names for ingredient and MedDRA term ---
    vri = ing.rename(columns={'rxnorm_id':'ingredient_id','rxnorm_name':'ingredient_name'})
    vme = med.rename(columns={'meddra_id':'effect_meddra_id','meddra_name':'effect_term'})

    # Make sure merge keys are strings
    num_labels_with_event['ingredient_id']   = num_labels_with_event['ingredient_id'].astype(str)
    num_labels_with_event['effect_meddra_id'] = num_labels_with_event['effect_meddra_id'].astype(str)
    total_labels['ingredient_id']            = total_labels['ingredient_id'].astype(str)
    vri['ingredient_id']                     = vri['ingredient_id'].astype(str)
    vme['effect_meddra_id']                  = vme['effect_meddra_id'].astype(str)

    # --- Final incidence table (per ingredient x effect) ---
    df = (
        num_labels_with_event
        .merge(total_labels, on='ingredient_id', how='left')
        .merge(vri[['ingredient_id','ingredient_name']], on='ingredient_id', how='left')
        .merge(vme[['effect_meddra_id','effect_term']], on='effect_meddra_id', how='left')
    )

    # incidence rate (label-level prevalence)
    df['percent_labels'] = (df['num_labels_with_event'].astype(float) / df['total_labels'].astype(float) * 100).round(2)

    # --- Helper: filter by ingredient name (case-insensitive contains) ---
    def effects_for(drug_substr, top_n=15):
        m = df[df['ingredient_name'].str.contains(drug_substr, case=False, na=False)].copy()
        if m.empty:
            return m
        # sort highest percent first
        m = m.sort_values(['percent_labels','num_labels_with_event'], ascending=[False, False])
        return m[['ingredient_name','effect_meddra_id','effect_term','num_labels_with_event','total_labels','percent_labels']].head(top_n)
    # ---- from notebook cell 5 ----
    vocab
    # ---- from notebook cell 6 ----
    # 0) Paths (adjust BASE if needed)
    BASE = "/kaggle/input/one-sides/onsides_v3/csv"

    # 1) Load MedDRA vocab (IDs + term + type), then align column names
    vocab = pd.read_csv(f"{BASE}/vocab_meddra_adverse_effect.csv", dtype=str)
    vocab = vocab.rename(columns={
        "meddra_id": "effect_meddra_id",
        "meddra_name": "vocab_effect_term",
        "meddra_term_type": "meddra_term_type"
    })[["effect_meddra_id","vocab_effect_term","meddra_term_type"]]

    # 2) Attach MedDRA term type to your results and keep PT only
    df_pt = (
        df.merge(vocab, on="effect_meddra_id", how="left")
          .query("meddra_term_type == 'PT'")
          .copy()
    )

    # --- ensure needed columns / types ---
    # If you have US/EU/UK/JP flags but no sources_seen, build it:
    if "sources_seen" not in df_pt.columns:
        flag_cols = [c for c in ["US","EU","UK","JP"] if c in df_pt.columns]
        if flag_cols:
            df_pt["sources_seen"] = df_pt[flag_cols].apply(
                lambda r: ", ".join([c for c in flag_cols if pd.to_numeric(r[c], errors="coerce")==1]),
                axis=1
            )
        else:
            df_pt["sources_seen"] = ""

    for c in ["num_labels_with_event", "total_labels", "percent_labels"]:
        if c in df_pt.columns:
            df_pt[c] = pd.to_numeric(df_pt[c], errors="coerce")

    # ---- from notebook cell 10 ----
    # Very light blocklist (skip junky/too generic terms)
    BLOCKLIST = set([
        "pain", "injury", "procedural pain", "condition aggravated",
        "drug interaction", "off label use", "product use issue"
    ])
    # ---- from notebook cell 11 ----
    # 2) Priority bump for inherently high-risk tags (even if combined% is low)
    PRIORITY_BUMP = {"QT_PROLONGATION": 2}  # add others if you like
    # ---- from notebook cell 12 ----
    TAG_MAP = {
        "SEVERE_CUTANEOUS": [
            "stevens-johnson syndrome", "stevens johnson syndrome", "sjs",       # ok, we'll boundary it
            "toxic epidermal necrolysis",                                        # use full phrase
            "dress", "drug reaction with eosinophilia and systemic symptoms",
            "erythema multiforme", "angioedema", "bullous"
        ],
        "CNS_DEPRESSION": [
            "somnolence","sedation","drowsiness","dizziness",
            "confusional state","fatigue","lethargy","hypersomnia"
        ],
        "QT_PROLONGATION": [
            "torsade de pointes","ventricular tachycardia",
            "long qt","qt prolonged","qt prolongation","ventricular arrhythmia"
        ],
        "HEPATOTOXICITY": [
            "hepatitis","liver function test abnormal","transaminases increased",
            "hyperbilirubinaemia","hyperbilirubinemia","hepatic failure",
            "cholestatic hepatitis","alt increased","ast increased"
        ],
        "BLEEDING": [
            "haemorrhage","hemorrhage","epistaxis",
            "gastrointestinal haemorrhage","gastrointestinal hemorrhage",
            "thrombocytopenia","coagulopathy"
        ],
        "NEPHROTOXICITY": [
            "renal failure","acute kidney injury","creatinine increased",
            "interstitial nephritis","proteinuria","haematuria","hematuria"
        ],
        "SEIZURE_RISK": [
            "seizure","seizures","convulsion","convulsions",
            "tonic clonic","clonic convulsion","status epilepticus"
        ],
        "HYPOTENSION": [
            "hypotension","orthostatic hypotension","postural hypotension",
            "blood pressure decreased"
        ],
        "HYPERTENSION": [
            "hypertension","blood pressure increased"
        ],
        "BRADYCARDIA": [
            "bradycardia","heart rate decreased","sinus bradycardia"
        ],
        "TACHYCARDIA": [
            "tachycardia","supraventricular tachycardia"
        ],
        "HYPERKALEMIA": [
            "hyperkalaemia","hyperkalemia","potassium increased"
        ],
        "MYOPATHY_RHABDO": [
            "rhabdomyolysis","myopathy","ck increased","creatine kinase increased"
        ],
        "HYPERGLYCEMIA": [
            "hyperglycaemia","hyperglycemia","blood glucose increased"
        ],
        "HYPOGLYCEMIA": [
            "hypoglycaemia","hypoglycemia"
        ],
        "RESP_DEPRESSION": [
            "respiratory depression","hypoventilation"
        ],
        "C_DIFF": [
            "clostridium difficile","c. difficile","pseudomembranous colitis"
        ],
        "SUICIDALITY": [
            "suicidal ideation","suicidal behaviour","suicidal behavior","suicide attempt"
        ],
        "PANCYTOPENIA": [
            "pancytopenia","agranulocytosis","neutropenia","leukopenia","leucopenia"
        ],
    }
    # ---- from notebook cell 13 ----
    # Clinical severity weights (adjustable)
    TAG_WEIGHTS = {
        "QT_PROLONGATION": 4.0,
        "SEVERE_CUTANEOUS": 4.0,
        "SEIZURE_RISK": 3.5,
        "PANCYTOPENIA": 3.5,
        "BLEEDING": 3.0,
        "HEPATOTOXICITY": 3.0,
        "C_DIFF": 2.5,
        "TACHYCARDIA": 2.5,
        "BRADYCARDIA": 2.5,
        "CNS_DEPRESSION": 2.0,
        "HYPOTENSION": 2.0,
    }
    # ---- from notebook cell 25 ----
    USE_OPENAI_API = True  # False = free dev mode; True = real API calls
    # ---- from notebook cell 26 ----
    user_secrets = UserSecretsClient()
    my_secret_value = user_secrets.get_secret("openai_kaggle")
    os.environ["OPENAI_API_KEY"] = my_secret_value
    client = OpenAI()   # will pick up OPENAI_API_KEY automatically
    # ---- from notebook cell 27 ----
    @lru_cache(maxsize=4096)
    def cached_llm_action(tag, drug_a, drug_b, driver, ex_a, ex_b,
                          tone, style_hint, model, temperature, max_chars,
                          banned, redacts, review_trigs, use_api):
        prompt = build_action_prompt(tag, drug_a, drug_b, driver, ex_a, ex_b, tone, style_hint)
        raw = call_llm(prompt, model=model, temperature=temperature, use_api=use_api)
        clean = sanitize_text(raw, max_chars, list(banned), list(redacts))
        review = needs_review(clean, list(review_trigs))
        return clean, review
    # ---- from notebook cell 40 ----
    grade, table = get_dur2(
        'escitalopram', 'ondansetron',
        use_curated_for_high_risk=False,
        combine_method='max', #max |additive | intersection
        top_n=7,
        llm_mode='always',   # or 'always' or 'never'
        use_api=True        # True to call OpenAI, False to stay free/stubbed
    )
    print("Lexicomp-like overall grade:", grade)


    log.info("Pipeline completed.")

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the DUR MVP pipeline.")
    parser.add_argument("--input", type=Path, default=None, help="Path to input file or directory.")
    parser.add_argument("--output", type=Path, default=None, help="Directory for outputs (created if missing).")
    parser.add_argument("--config", type=Path, default=None, help="Optional config file.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--save-figures", action="store_true", help="Save figures to output/figures.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write outputs.")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    run(
        input_path=args.input,
        output_path=args.output,
        config=args.config,
        save_figures=args.save_figures,
        seed=args.seed,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
