import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

IAA_SUBSET   = "output/iaa_1700_subset.csv"
ANNOTATOR_A  = "output/iaa_1700_annotator_A.csv"
ANNOTATOR_B  = "output/iaa_1700_annotator_B.csv"
FULL_LABELED = "output/full_labeled.csv"
OUTPUT_FINAL = "output/dataset_final.csv"
RANDOM_SEED  = 42

def safe_read(f):
    for enc in ["utf-8-sig","utf-8","cp1252","latin-1"]:
        try:
            df = pd.read_csv(f, encoding=enc)
            print(f"  Loaded '{f}' | rows: {len(df)}")
            return df
        except: continue
    raise ValueError(f"Tidak bisa baca: {f}")

# ── 1. Human set ────────────────────────────────────────────────────────────
df_iaa = safe_read(IAA_SUBSET)
df_a   = safe_read(ANNOTATOR_A)[["comment_text","manual_label"]].rename(
            columns={"manual_label":"label_ricky"})
df_b   = safe_read(ANNOTATOR_B)[["comment_text","manual_label"]].rename(
            columns={"manual_label":"label_sinclair"})

df_human = df_iaa.merge(df_a, on="comment_text", how="left")
df_human = df_human.merge(df_b, on="comment_text", how="left")

def resolve(row):
    r = str(row.get("label_ricky","")).strip().upper()
    s = str(row.get("label_sinclair","")).strip().upper()
    if r == s and r in {"SPAM","NOT_SPAM"}: return r
    if r != s: return "SPAM"   # adjudicated conflicts
    return "NOT_SPAM"

df_human["final_label"]  = df_human.apply(resolve, axis=1)
df_human["label_source"] = "human"
df_human = df_human[df_human["final_label"].isin(["SPAM","NOT_SPAM"])]
# Drop kolom anotasi, keep semua kolom metadata asli
df_human = df_human.drop(columns=["label_ricky","label_sinclair"], errors="ignore")

print(f"\nHuman set  : {len(df_human)}")
print(df_human["final_label"].value_counts())

# ── 2. LLM set — keep semua kolom metadata ─────────────────────────────────
df_llm = safe_read(FULL_LABELED)
df_llm = df_llm[df_llm["final_label"].isin(["SPAM","NOT_SPAM"])].copy()

# Exclude overlap dengan human set
human_texts = set(df_human["comment_text"].astype(str))
df_llm = df_llm[~df_llm["comment_text"].isin(human_texts)].copy()

print(f"\nLLM set    : {len(df_llm)}")
print(df_llm["final_label"].value_counts())

# ── 3. Concat ───────────────────────────────────────────────────────────────
df_final = pd.concat([df_human, df_llm], ignore_index=True)
df_final = df_final.drop_duplicates(subset="comment_text").reset_index(drop=True)

print(f"\nDataset final: {len(df_final)}")
print(df_final["final_label"].value_counts())
print(df_final["final_label"].value_counts(normalize=True).mul(100).round(1))
print(f"\nKolom: {list(df_final.columns)}")

# ── 4. Stratified split 80/10/10 ────────────────────────────────────────────
df_spam     = df_final[df_final["final_label"]=="SPAM"]
df_not_spam = df_final[df_final["final_label"]=="NOT_SPAM"]

def split(df):
    train, temp = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    val, test   = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED)
    return train, val, test

s_tr, s_va, s_te = split(df_spam)
n_tr, n_va, n_te = split(df_not_spam)

df_train = pd.concat([s_tr,n_tr]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
df_val   = pd.concat([s_va,n_va]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
df_test  = pd.concat([s_te,n_te]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\nSplit (seed={RANDOM_SEED}):")
print(f"  Train : {len(df_train):5d} | SPAM={( df_train['final_label']=='SPAM').sum()} ({(df_train['final_label']=='SPAM').sum()/len(df_train)*100:.1f}%)")
print(f"  Val   : {len(df_val):5d}   | SPAM={(df_val['final_label']=='SPAM').sum()} ({(df_val['final_label']=='SPAM').sum()/len(df_val)*100:.1f}%)")
print(f"  Test  : {len(df_test):5d}  | SPAM={(df_test['final_label']=='SPAM').sum()} ({(df_test['final_label']=='SPAM').sum()/len(df_test)*100:.1f}%)")

# ── 5. Save ─────────────────────────────────────────────────────────────────
df_final.to_csv(OUTPUT_FINAL,         index=False, encoding="utf-8-sig")
df_train.to_csv("output/train.csv",   index=False, encoding="utf-8-sig")
df_val.to_csv(  "output/val.csv",     index=False, encoding="utf-8-sig")
df_test.to_csv( "output/test.csv",    index=False, encoding="utf-8-sig")

print(f"\n✅ Tersimpan:")
print(f"  output/dataset_final.csv")
print(f"  output/train.csv / val.csv / test.csv")
print(f"\nKasih folder output/ ini ke Elmer.")