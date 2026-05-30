import pandas as pd
import numpy as np
import re
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────
FULL_DATASET  = "output/raw_75k.csv"
OUTPUT_SUBSET = "output/iaa_1700_subset.csv"
ANNOTATOR_A   = "output/iaa_1700_annotator_A.csv"  # Ricky
ANNOTATOR_B   = "output/iaa_1700_annotator_B.csv"  # Sinclair
TARGET_N      = 1700
RANDOM_SEED   = 42

# ── HELPERS ────────────────────────────────────────────────────────────────

def safe_read_csv(filepath: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"   Loaded '{filepath}' | encoding: {enc} | rows: {len(df)}")
            return df
        except Exception:
            continue
    raise ValueError(f"Tidak bisa baca: {filepath}")

def has_obfuscation(text: str) -> bool:
    if pd.isna(text):
        return False
    t = str(text)
    # Mathematical unicode block
    if any('\U0001D400' <= c <= '\U0001D7FF' for c in t):
        return True
    # Zero-width / BIDI chars
    if any(z in t for z in ['\u200b','\u200c','\u200d','\u2060','\ufeff','\u200e','\u200f']):
        return True
    # Fullwidth latin
    if any('\uFF21' <= c <= '\uFF5A' for c in t):
        return True
    # Spaced latin (P U L A U)
    if re.search(r'(?<=[A-Za-z]) (?=[A-Za-z])', t):
        return True
    # Cyrillic mixing
    if any('\u0400' <= c <= '\u04FF' for c in t):
        return True
    # Enclosed alphanumerics
    if any('\u2460' <= c <= '\u24FF' for c in t):
        return True
    return False

# ── MAIN ───────────────────────────────────────────────────────────────────

def create_iaa_subset():
    df = safe_read_csv(FULL_DATASET)
    print(f"\nTotal dataset: {len(df)} rows")
    print(f"Kolom tersedia: {list(df.columns)}")

    # Validasi kolom comment_text
    if "comment_text" not in df.columns:
        print(f"❌ Kolom 'comment_text' tidak ditemukan!")
        return

    # Auto-generate comment_id kalau tidak ada
    if "comment_id" not in df.columns:
        df.insert(0, "comment_id", range(1, len(df) + 1))
        print(f"   ℹ️  comment_id di-generate otomatis (1 s/d {len(df)})")

    # Drop duplikat berdasarkan teks
    before = len(df)
    df = df.drop_duplicates(subset="comment_text").reset_index(drop=True)
    print(f"Setelah dedup: {len(df)} rows (drop {before - len(df)} duplikat)")

    # Deteksi obfuskasi
    print("\nMendeteksi obfuskasi... (mungkin butuh 10–30 detik)")
    df["any_obf"] = df["comment_text"].apply(has_obfuscation)

    obf_pool    = df[df["any_obf"] == True].copy()
    no_obf_pool = df[df["any_obf"] == False].copy()

    print(f"\nPool obf=1 : {len(obf_pool):,} rows ({len(obf_pool)/len(df)*100:.1f}%)")
    print(f"Pool obf=0 : {len(no_obf_pool):,} rows ({len(no_obf_pool)/len(df)*100:.1f}%)")

    # Stratified sampling — equal allocation per stratum
    n_per_stratum = TARGET_N // 2  # 850 + 850

    if len(obf_pool) < n_per_stratum:
        print(f"⚠️  Pool obf hanya {len(obf_pool)}, ambil semua + kompensasi dari no_obf")
        n_obf    = len(obf_pool)
        n_no_obf = TARGET_N - n_obf
    else:
        n_obf    = n_per_stratum
        n_no_obf = n_per_stratum

    sample_obf    = obf_pool.sample(n=n_obf,    random_state=RANDOM_SEED)
    sample_no_obf = no_obf_pool.sample(n=n_no_obf, random_state=RANDOM_SEED)

    df_subset = pd.concat([sample_obf, sample_no_obf]).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(f"\n{'='*55}")
    print(f"Subset IAA {TARGET_N}:")
    print(f"   obf=1 : {df_subset['any_obf'].sum():,} ({df_subset['any_obf'].sum()/len(df_subset)*100:.1f}%)")
    print(f"   obf=0 : {(~df_subset['any_obf']).sum():,} ({(~df_subset['any_obf']).sum()/len(df_subset)*100:.1f}%)")
    print(f"   Total : {len(df_subset):,}")
    print(f"   Seed  : {RANDOM_SEED} (reproducible)")

    # Simpan subset lengkap (untuk LLM labeling)
    df_subset.to_csv(OUTPUT_SUBSET, index=False, encoding="utf-8-sig")
    print(f"\n✅ Subset tersimpan : {OUTPUT_SUBSET}")

    # Simpan file blind untuk annotator (hanya comment_id + comment_text)
    df_blind = df_subset[["comment_id", "comment_text"]].copy()
    df_blind["manual_label"]      = ""
    df_blind["manual_confidence"] = ""
    df_blind["notes"]             = ""

    df_blind.to_csv(ANNOTATOR_A, index=False, encoding="utf-8-sig")
    df_blind.to_csv(ANNOTATOR_B, index=False, encoding="utf-8-sig")

    print(f"✅ File annotator:")
    print(f"   → {ANNOTATOR_A}  (Ricky)")
    print(f"   → {ANNOTATOR_B}  (Sinclair)")
    print(f"\n{'='*55}")
    print(f"Instruksi untuk annotator:")
    print(f"   Buka file CSV, isi kolom:")
    print(f"   • manual_label      : SPAM / NOT_SPAM / UNCERTAIN")
    print(f"   • manual_confidence : HIGH / MEDIUM / LOW")
    print(f"   • notes             : (opsional) catatan per komentar")
    print(f"\n⚠️  PENTING: Jangan lihat file annotator satu sama lain!")
    print(f"   Labeling harus independen dan blind.")

if __name__ == "__main__":
    create_iaa_subset()