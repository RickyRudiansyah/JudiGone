import pandas as pd
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────
LABELED_CSV  = "output/iaa_1700_subset.csv"   # output dari llm_labeling.py
REPORT_CSV   = "output/iaa_1700_report.csv"
ANNOTATOR_A  = "output/iaa_1700_annotator_A.csv"   # Ricky (sudah diisi)
ANNOTATOR_B  = "output/iaa_1700_annotator_B.csv"   # Sinclair (sudah diisi)


def safe_read_csv(filepath: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"   Loaded '{filepath}' | encoding: {enc}")
            return df
        except Exception:
            continue
    raise ValueError(f"Tidak bisa baca: {filepath}")

def load_annotator(filepath: str, label: str) -> pd.DataFrame | None:
    if not Path(filepath).exists():
        print(f"File tidak ditemukan: {filepath}")
        return None
    df = safe_read_csv(filepath)
    required = {"comment_id", "comment_text", "manual_label", "manual_confidence"}
    missing  = required - set(df.columns)
    if missing:
        print(f"Kolom kurang di {filepath}: {missing}")
        return None
    df = df.rename(columns={
        "manual_label":      f"label_{label}",
        "manual_confidence": f"confidence_{label}",
    })
    print(f"Annotator {label} loaded: {len(df)} rows")
    return df[["comment_id", f"label_{label}", f"confidence_{label}"]]

def _interpret(kappa: float) -> str:
    if   kappa >= 0.81: return "Almost Perfect"
    elif kappa >= 0.61: return "Substantial"
    elif kappa >= 0.41: return "Moderate"
    elif kappa >= 0.21: return "Fair"
    else:               return "Slight/Poor"

def _compute_iaa(df: pd.DataFrame, col1: str, col2: str) -> tuple:
    try:
        from sklearn.metrics import cohen_kappa_score, confusion_matrix
    except ImportError:
        print("pip install scikit-learn")
        return None, None, None, None

    v3 = df[[col1, col2]].dropna()
    v3 = v3[(v3[col1] != "") & (v3[col2] != "")]
    if len(v3) >= 2:
        try:
            k3    = cohen_kappa_score(v3[col1], v3[col2],
                                      labels=["SPAM", "NOT_SPAM", "UNCERTAIN"])
            raw_3 = sum(a == b for a, b in zip(v3[col1], v3[col2])) / len(v3) * 100
            print(f"\n[3-class incl. UNCERTAIN]  n={len(v3)}")
            print(f"   Cohen's kappa : {k3:.4f} — {_interpret(k3)}")
            print(f"   Raw Agreement : {raw_3:.2f}%")
        except Exception:
            pass

    vd = df[[col1, col2]].dropna()
    vd = vd[~vd[col1].isin(["UNCERTAIN", ""]) & ~vd[col2].isin(["UNCERTAIN", ""])]
    if len(vd) < 2:
        print(f"   Data tidak cukup (n={len(vd)})")
        return None, None, None, 0

    y1, y2     = vd[col1].tolist(), vd[col2].tolist()
    kappa      = cohen_kappa_score(y1, y2)
    raw_agr    = sum(a == b for a, b in zip(y1, y2)) / len(y1) * 100
    conflicts  = vd[vd[col1] != vd[col2]]
    n_conflict = len(conflicts)

    print(f"\n[Decisive only: SPAM / NOT_SPAM]  n={len(vd)}")
    print(f"   Cohen's kappa : {kappa:.4f} — {_interpret(kappa)}")
    print(f"   Raw Agreement : {raw_agr:.2f}%")
    print(f"   Konflik       : {n_conflict} kasus ({n_conflict/len(vd)*100:.1f}%)")

    labels = ["SPAM", "NOT_SPAM"]
    cm     = confusion_matrix(y1, y2, labels=labels)
    cm_df  = pd.DataFrame(cm,
                          index=[f"{col1}={l}" for l in labels],
                          columns=[f"{col2}={l}" for l in labels])
    print(f"\n[Confusion Matrix]\n{cm_df.to_string()}")

    if n_conflict > 0:
        print(f"\n[Detail Konflik — {n_conflict} kasus]")
        for _, row in conflicts.iterrows():
            text = str(df.loc[row.name, "comment_text"])[:65] \
                   if "comment_text" in df.columns else "N/A"
            print(f"  {col1}={row[col1]:10s} | {col2}={row[col2]:10s} | {text}...")

    return kappa, raw_agr, n_conflict, len(vd)


def generate_report():
    df_llm = safe_read_csv(LABELED_CSV)

    df_a = load_annotator(ANNOTATOR_A, "ricky")
    df_b = load_annotator(ANNOTATOR_B, "sinclair")

    df_merged = df_llm.copy()
    if df_a is not None:
        df_merged = df_merged.merge(df_a, on="comment_id", how="left")
    if df_b is not None:
        df_merged = df_merged.merge(df_b, on="comment_id", how="left")

    print(f"\nDistribusi LLM Labels (n={len(df_merged)}):")
    for label, count in df_merged["llm_label"].value_counts().items():
        print(f"   {label:12s}: {count:4d} ({count/len(df_merged)*100:.1f}%)")

    pairs = [
        ("llm_label",   "label_ricky",   "LLM vs Ricky (Annotator A)"),
        ("llm_label",   "label_sinclair", "LLM vs Sinclair (Annotator B)"),
        ("label_ricky", "label_sinclair", "Ricky vs Sinclair (Human IAA)"),
    ]

    report_summary = []
    for col1, col2, title in pairs:
        if col1 in df_merged.columns and col2 in df_merged.columns:
            print("\n" + "=" * 55)
            print(title)
            kappa, raw_agr, n_conflict, n_valid = _compute_iaa(df_merged, col1, col2)
            if kappa is not None:
                report_summary.append({
                    "pair": title,
                    "kappa": round(kappa, 4),
                    "raw_agreement_pct": round(raw_agr, 2),
                    "n_conflict": n_conflict,
                    "n_valid": n_valid,
                    "interpretation": _interpret(kappa)
                })

    df_merged.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    summary_path = REPORT_CSV.replace(".csv", "_summary.csv")
    pd.DataFrame(report_summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ -> {REPORT_CSV}")
    print(f"✅ -> {summary_path}")


if __name__ == "__main__":
    generate_report()