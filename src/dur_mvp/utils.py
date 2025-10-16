"""Utility functions for DUR MVP. Auto-extracted and tidied."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Iterable
from pathlib import Path

def actiongen_policy(
    llm_mode: str = "blend",                     # "never" | "blend" | "always"
    use_curated_for_high_risk: bool = True,
    combine_method: str = "max",                 # "max" | "additive" | "intersection"
    top_n: int = 7,
    temperature: float = 0.0,
    max_chars: int = 280,
    tone: str = "conservative",
    banned_phrases: List[str] = None,
    redact_regexes: List[str] = None,
    add_review_flags_for: List[str] = None,
    tag_overrides: Dict[str, str] = None,
    model_name: str = "gpt-4o-mini",
    use_api: bool = False,
    style_hint: str = None,
) -> Dict[str, Any]:
    """
    Build a DUR action-generation policy (previously ActionGenPolicy dataclass).

    Returns:
        dict: Configuration dictionary equivalent to ActionGenPolicy.
    """
    return {
        "llm_mode": llm_mode,
        "use_curated_for_high_risk": use_curated_for_high_risk,
        "combine_method": combine_method,
        "top_n": top_n,
        "temperature": temperature,
        "max_chars": max_chars,
        "tone": tone,
        "banned_phrases": banned_phrases or [],
        "redact_regexes": redact_regexes or [],
        "add_review_flags_for": add_review_flags_for
            or ["ECG", "contraindicated", "hospitalize", "discontinue"],
        "tag_overrides": tag_overrides or {},
        "model_name": model_name,
        "use_api": use_api,
        "style_hint": style_hint
            or (
                "One sentence, actionable, no citations, do not invent facts; "
                "prefer monitoring/counseling; only advise ECG/labs/avoid if widely standard; ≤ 280 chars."
            ),
    }
# ==== From notebook cell 14 ====
# Normalize helper
def _norm(s: str) -> str:
    return (s or "").strip().lower()

# ==== From notebook cell 15 ====
# --- 4) Simple severity policy ---
def derive_overall_severity(details):
    # Escalate if any severe cutaneous overlap
    for d in details:
        if d["tag"] == "SEVERE_CUTANEOUS" and d["combined%"] >= 1.0:
            return "Major – Avoid/Monitor"
    # Escalate if any QT overlap
    for d in details:
        if d["tag"] == "QT_PROLONGATION" and d["combined%"] >= 1.0:
            return "Moderate – Monitor/ECG if risk"
    # Otherwise
    return "No flagged interaction in this step"

# ==== From notebook cell 16 ====
def normalize_pct(x):
    # Handle mixes like 0.66 vs 94.51
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    # Treat <1 as fraction and >=1 as already percent
    return v*100.0 if 0 <= v <= 1 else v

# ==== From notebook cell 17 ====
def calc_combined(a, b, method="max"):
    """Combine two % signals (already normalized to 0–100)."""
    a = 0.0 if a is None else float(a)
    b = 0.0 if b is None else float(b)

    if method == "max":
        return max(a, b)
    elif method == "additive":
        # union-like: a + b - a*b/100
        return a + b - (a * b / 100.0)
    elif method == "intersection":
        # previous behavior (overlap/min)
        return min(a, b)
    else:
        raise ValueError(f"Unknown combine method: {method}")

# ==== From notebook cell 18 ====
def sanitize_text(text: str, max_chars: int, banned_phrases: list, redact_regexes: list):
    s = (text or "").strip().replace("\n", " ")
    for pat in redact_regexes:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    for bad in banned_phrases:
        if re.search(re.escape(bad), s, flags=re.IGNORECASE):
            s = s.replace(bad, "")
    s = re.sub(r"\s{2,}", " ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip()
    # ensure single sentence end
    if not s.endswith((".", "!", "?")):
        s += "."
    return s

# ==== From notebook cell 19 ====
def needs_review(text: str, review_triggers: list) -> bool:
    return any(trig.lower() in text.lower() for trig in review_triggers)

# ==== From notebook cell 21 ====
def get_details(drug_a: str, drug_b: str, df_pt=None):
    if df_pt is None:
        raise ValueError("df_pt is required. Pass df_pt=... to get_details/get_dur2.")
    severity, details = pd_interaction_report(
        df_pt,
        drug_a,
        drug_b,
        min_pct=0.1,          # only include PTs that appear on ≥0.5% of labels for that ingredient
        require_pt_sources=1  # require at least N distinct label sources (US/EU/UK/JP)
        )
    return severity, details

# ==== From notebook cell 22 ====
def tag_scores_for_drug(df_pt, drug_name, min_pct=0.0, require_pt_sources=1):
    """
    df_pt columns expected: ingredient_name, effect_term_clean, percent_labels, sources_seen
    """
    dx = df_pt[df_pt["ingredient_name"].str.lower() == drug_name.lower()].copy()
    dx = dx.dropna(subset=["vocab_effect_term"])
    dx = dx[~dx["vocab_effect_term"].str.lower().isin(BLOCKLIST)]

    # filter by percent incidence threshold
    if min_pct > 0:
        dx = dx[pd.to_numeric(dx["percent_labels"], errors="coerce") >= float(min_pct)]

    # filter by how many label sources (if available)
    if "sources_seen" in dx.columns and require_pt_sources > 1:
        def _src_count(val):
            if pd.isna(val) or not str(val).strip():
                return 0
            return len([t for t in str(val).replace(" ", "").split(",") if t])
        dx = dx[dx["sources_seen"].map(_src_count) >= int(require_pt_sources)]

    # Map each effect to a TAG (first match wins)
    def effect_to_tag(term):
        t = _norm(term)
        for tag, kws in TAG_MAP.items():
            for kw in kws:
                if kw in t:
                    return tag
        return None

    dx["tag"] = dx["vocab_effect_term"].map(effect_to_tag)
    dx = dx.dropna(subset=["tag"])

    # Aggregate: take max % per tag and keep up to 3 example effects
    out = {}
    for tag, grp in dx.groupby("tag"):
        grp = grp.sort_values("percent_labels", ascending=False)
        max_pct = float(grp["percent_labels"].max())
        examples = grp.loc[:, ["vocab_effect_term", "percent_labels"]].head(3)
        examples = [
            {"effect_term": r["vocab_effect_term"], "percent_labels": float(r["percent_labels"])}
            for _, r in examples.iterrows()
        ]
        out[tag] = {"max_pct": max_pct, "examples": examples}
    return out

# ==== From notebook cell 23 ====
def generate_action_text(tag, drug_a, drug_b, driver, ex_a, ex_b, policy: ActionGenPolicy):
    """
    Returns (action_text, action_source)
    action_source ∈ {"curated","blend","llm"}
    """
    override = policy.tag_overrides.get(tag)
    mode = override if override else policy.llm_mode

    # 1) High-risk lock → curated only
    if policy.use_curated_for_high_risk and tag in HIGH_RISK_TAGS:
        return TAG_ACTIONS.get(tag, "Use conservative monitoring; escalate if concerning symptoms."), "curated"

    if mode == "blend":
        base = TAG_ACTIONS.get(tag, "")
        if base:
            text, review = cached_llm_action(
                tag, drug_a, drug_b, driver, ex_a, ex_b + f" (Base: {base})",
                policy.tone, policy.style_hint, policy.model_name, policy.temperature,
                policy.max_chars, tuple(policy.banned_phrases), tuple(policy.redact_regexes),
                tuple(policy.add_review_flags_for), policy.use_api                 # <<—— NEW
            )
            if review: text = "[REVIEW] " + text
            return text, "blend"
        else:
            text, review = cached_llm_action(
                tag, drug_a, drug_b, driver, ex_a, ex_b,
                policy.tone, policy.style_hint, policy.model_name, policy.temperature,
                policy.max_chars, tuple(policy.banned_phrases), tuple(policy.redact_regexes),
                tuple(policy.add_review_flags_for), policy.use_api                 # <<—— NEW
            )
            if review: text = "[REVIEW] " + text
            return text, "llm"

    if mode == "always":
        text, review = cached_llm_action(
            tag, drug_a, drug_b, driver, ex_a, ex_b,
            policy.tone, policy.style_hint, policy.model_name, policy.temperature,
            policy.max_chars, tuple(policy.banned_phrases), tuple(policy.redact_regexes),
            tuple(policy.add_review_flags_for), policy.use_api                     # <<—— NEW
        )
        if review: text = "[REVIEW] " + text
        return text, "llm"

    # Fallback
    return TAG_ACTIONS.get(tag, "Standard monitoring and counseling."), "curated"

# ==== From notebook cell 28 ====
def build_action_prompt(tag, drug_a, drug_b, driver, ex_a, ex_b, tone, style_hint):
    style_hint = style_hint or "One sentence, conservative, ≤ 280 chars."
    return (
        f"You are a cautious clinical pharmacist. Tone: {tone}.\n"
        f"{style_hint}\n\n"
        f"Drugs: {drug_a} + {drug_b}\n"
        f"Tag: {tag}\n"
        f"Driver drug: {driver or '-'}\n"
        f"Examples {drug_a}: {ex_a or '-'}\n"
        f"Examples {drug_b}: {ex_b or '-'}\n"
        f"Return ONE action sentence:"
    )

# ==== From notebook cell 29 ====
def call_llm(prompt: str, model="gpt-4o-mini", temperature=0.0, use_api=False) -> str:
    if not use_api:
        # Free, local dev output
        return "[dev stub] " + prompt[:200]
    # Live API call
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a cautious clinical pharmacist."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return r.choices[0].message.content.strip()

# ==== From notebook cell 31 ====
def pd_interaction_report(df_pt, drug_a: str, drug_b: str, min_pct: float = 0.5, require_pt_sources: int = 1):
    A = tag_scores_for_drug(df_pt, drug_a, min_pct=min_pct, require_pt_sources=require_pt_sources)
    B = tag_scores_for_drug(df_pt, drug_b, min_pct=min_pct, require_pt_sources=require_pt_sources)
    details = combine_pd(A, B)
    severity = derive_overall_severity(details)
    return severity, details

# ==== From notebook cell 32 ====
def combine_pd(scores_a: dict, scores_b: dict):
    """
    Build a 'details' list: for each tag present in A or B,
    show drug_a_max%, drug_b_max%, combined% = min(a,b), examples.
    """
    tags = set(scores_a.keys()) | set(scores_b.keys())
    details = []
    for tag in sorted(tags):
        a = scores_a.get(tag, {"max_pct": 0.0, "examples": []})
        b = scores_b.get(tag, {"max_pct": 0.0, "examples": []})
        details.append({
            "tag": tag,
            "drug_a_max%": a["max_pct"],
            "drug_b_max%": b["max_pct"],
            "combined%": min(a["max_pct"], b["max_pct"]),
            "examples_a": a["examples"],
            "examples_b": b["examples"],
        })
    details.sort(key=lambda x: x["combined%"], reverse=True)
    return details

# ==== From notebook cell 33 ====
def to_actionable_table(details, drug_a, drug_b, top_n=7, combine_method="max", policy: ActionGenPolicy=None):
    if policy is None:
        policy = ActionGenPolicy()

    rows = []
    for d in details:
        tag = d.get("tag")
        a = normalize_pct(d.get("drug_a_max%"))
        b = normalize_pct(d.get("drug_b_max%"))
        if tag is None:
            continue

        ca = calc_combined(a, b, method=combine_method)
        driver = drug_a if (a or 0) > (b or 0) else (drug_b if (b is not None) else None)
        priority = (ca or 0) + PRIORITY_BUMP.get(tag, 0)

        ex_a = ", ".join(sorted({e["effect_term"] for e in d.get("examples_a", [])})) or "-"
        ex_b = ", ".join(sorted({e["effect_term"] for e in d.get("examples_b", [])})) or "-"

        action_text, action_source = generate_action_text(tag, drug_a, drug_b, driver, ex_a, ex_b, policy)

        rows.append({
            "tag": tag,
            "combined_pct": round(ca, 2) if ca is not None else None,
            f"{drug_a}_max_pct": round(a, 2) if a is not None else None,
            f"{drug_b}_max_pct": round(b, 2) if b is not None else None,
            "driver": driver,
            f"examples_{drug_a}": ex_a,
            f"examples_{drug_b}": ex_b,
            "action": action_text,
            "action_source": action_source,   # <-- NEW COLUMN
            "priority_score": round(priority, 2)
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["priority_score", "combined_pct"], ascending=[False, False]) \
               .head(top_n).reset_index(drop=True)
    return df

# ==== From notebook cell 34 ====
# --- build df, sorted high→low by combined ---
def build_gauge_df(details, drug_a: str, drug_b: str, top_n=6, combine_method="max"):
    rows = []
    for d in details:
        tag = d.get("tag")
        a = normalize_pct(d.get("drug_a_max%"))
        b = normalize_pct(d.get("drug_b_max%"))
        combined = calc_combined(a, b, method=combine_method)

        ex_a = ", ".join([e["effect_term"] for e in d.get("examples_a", [])]) or "-"
        ex_b = ", ".join([e["effect_term"] for e in d.get("examples_b", [])]) or "-"

        rows.append({
            "tag": tag,
            "combined_pct": round(combined, 2),
            f"{drug_a}_pct": a,
            f"{drug_b}_pct": b,
            "examples_a": ex_a,
            "examples_b": ex_b,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # sort high → low, then take top_n
    df = df.sort_values("combined_pct", ascending=False).head(top_n).reset_index(drop=True)
    return df

# ==== From notebook cell 35 ====
def draw_semi_gauge(value_pct, ax=None, label=None):
    theta = np.linspace(-np.pi, 0, 181)
    if ax is None: ax = plt.gca()
    ax.plot(np.cos(theta), np.sin(theta))                 # arc
    ang = -np.pi * (float(value_pct)/100.0)               # needle angle
    ax.plot([0, np.cos(ang)], [0, np.sin(ang)])           # needle
    ax.set_aspect('equal'); ax.axis('off')
    ax.text(0, -0.2, f"{float(value_pct):.1f}%", ha="center", va="center", fontsize=11)
    if label: ax.text(0, 0.25, label, ha="center", va="center", fontsize=10)

# ==== From notebook cell 36 ====
def plot_gauge_grid(df_combined: pd.DataFrame, drug_a: str, drug_b: str,
                    top_n=7, suptitle="Overlap gauges (combined %)"):
    """
    Plot all gauges in a single horizontal row, left→right by combined_pct.
    """
    if df_combined.empty:
        print("No tags to plot."); return

    # sort and trim
    col = "combined_pct" if "combined_pct" in df_combined.columns else "combined%"
    d = df_combined.sort_values(col, ascending=False).head(top_n).copy()

    # force one row of N columns
    cols = len(d)
    rows = 1
    plt.figure(figsize=(cols * 3.0, 3.0))   # width scales with number of gauges

    for i, (_, row) in enumerate(d.iterrows(), start=1):
        ax = plt.subplot(rows, cols, i)
        value = float(row[col])
        subtitle = str(row["tag"]).replace("_", " ")
        draw_semi_gauge(value, ax=ax, label=subtitle.title())

        ex_a = "\n".join([e for e in str(row["examples_a"]).split(", ") if e and e != "nan"])
        ex_b = "\n".join([e for e in str(row["examples_b"]).split(", ") if e and e != "nan"])

        ax.text(0, -1.0,
                f"{drug_a}:\n{ex_a}\n\n{drug_b}:\n{ex_b}",
                ha="center", va="top", fontsize=9, wrap=True)

    plt.suptitle(suptitle, y=1.05)
    plt.tight_layout()

# ==== From notebook cell 38 ====
def letter_score_from_actionable(df_actionable, drug_a_col=None, drug_b_col=None):
    """
    Derive a Lexicomp-style A/B/C/D/X severity letter from an actionable-table DataFrame.
    Gives higher weight when BOTH drugs contribute to the risk.
    """
    if df_actionable.empty:
        return "A"  # no known interaction

    tag_scores = []
    for _, row in df_actionable.iterrows():
        tag = row["tag"]
        w = TAG_WEIGHTS.get(tag, 1.0)
        combined = row.get("combined_pct", 0) or 0

        # Determine each drug's own % if columns are present
        if not drug_a_col or not drug_b_col:
            # try to detect automatically (first two ending in '_max_pct')
            pct_cols = [c for c in row.index if c.endswith("_max_pct")]
            if len(pct_cols) >= 2:
                drug_a_col, drug_b_col = pct_cols[0], pct_cols[1]
        a_pct = row.get(drug_a_col, 0) or 0
        b_pct = row.get(drug_b_col, 0) or 0

        # Synergy boost: only if both drugs have measurable contribution
        if a_pct > 0 and b_pct > 0:
            ratio = min(a_pct, b_pct) / max(a_pct, b_pct)
            synergy = 1.0 + 0.5 * ratio   # up to +50% boost if perfectly balanced
        else:
            synergy = 1.0

        # sqrt scaling tempers extreme values
        score = w * synergy * (combined ** 0.5)
        tag_scores.append(score)

    # Emphasize the worst few risks
    total = sum(sorted(tag_scores, reverse=True)[:3])

    # Map to Lexicomp-like letter grades (tune as needed)
    if total >= 200:
        return "X"   # Contraindicated / Do not use together
    elif total >= 140:
        return "D"   # Consider therapy modification
    elif total >= 80:
        return "C"   # Monitor therapy
    elif total >= 30:
        return "B"   # No special action usually needed
    else:
        return "A"   # No known interaction

# ==== From notebook cell 39 ====
def get_dur2(drug_a, drug_b, use_curated_for_high_risk=True,
            combine_method='max', top_n=7, llm_mode='blend', use_api=False, df_pt=None):
    """
    Build an actionable drug–drug interaction table, draw gauges,
    and provide a single Lexicomp-style severity grade (A/B/C/D/X).
    """
    policy = actiongen_policy(
        llm_mode=llm_mode,
        use_curated_for_high_risk=use_curated_for_high_risk,
        combine_method=combine_method,
        top_n=top_n,
        use_api=use_api,
        model_name="gpt-4o-mini",
        temperature=0.0,
        add_review_flags_for=["ECG","contraindicated","hospitalize","discontinue"],
        )


    # run the DDI pipeline
    severity, details = get_details(drug_a, drug_b, df_pt=df_pt)
    df_actionable = to_actionable_table(details, drug_a, drug_b,
                                        top_n=top_n, combine_method=combine_method, policy=policy)

    # compute overall Lexicomp-like grade
    #overall_grade = letter_score_from_actionable(df_actionable)

    overall_grade = letter_score_from_actionable(df_actionable,
    drug_a_col=f"{drug_a}_max_pct",
    drug_b_col=f"{drug_b}_max_pct"
    )
    print(f"Overall Interaction Grade (Lexicomp style): {overall_grade}")


    # display results
    print(f"Using OPENAI = {use_api}")
    print(f"Overall Interaction Grade (Lexicomp style): {overall_grade}")
    display(df_actionable)

    # draw gauges
    df_combined = build_gauge_df(details, drug_a, drug_b,
                                 top_n=top_n, combine_method=combine_method)
    plot_gauge_grid(df_combined, drug_a, drug_b, top_n=top_n,
                    suptitle=f"Overlap gauges ({combine_method} method)")

    # return the grade and full table for further use
    return overall_grade, df_actionable
