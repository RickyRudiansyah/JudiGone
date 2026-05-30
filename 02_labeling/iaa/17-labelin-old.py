import requests, json, re, time, unicodedata
import pandas as pd

# ── CONFIG ─────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:14b"
INPUT_CSV    = "output/iaa_1700_subset.csv"   # ganti ke file 1700
SAVE_EVERY   = 50                              # naikkan ke 50 karena 1700 data
MAX_CHARS    = 1500

KNOWN_BRANDS = [
    "ALEXIS17", "JUNIOR88", "BERKAH99", "TKP303", "MIYA88",
    "MANTAP89", "PSTOTO99", "PULAU777", "MANDALIKA77", "APALAGIINI07",
    "PROBET855", "RA77", "JPTOGEL77", "RO88", "LEXIS17",
    "XL777", "ULAU77", "LIATPSTOTO99", "MAINDIDORA77", "KING328",
    "HANYADIKING328", "KE10", "ANAM1431", "MAKASIPULAU777",
    "CUMADIKING328", "BUKIT888", "LUCKI79",
    "PULAUWIN", "PULAU77", "MAHJONG303", "MAHJONG88",
    "AERO77", "AERO88", "AERO303", "AERO", "KOISLOT",
    "SLOT88", "GATES77", "XUXU4D", "GARUDAHOK", "GARUDA88",
]

ZW_CHARS = ['\u200b','\u200c','\u200d','\u2060','\ufeff','\u200e','\u200f']

USERNAME_MENTION_PATTERN = re.compile(r'@[A-Za-z0-9_.]+', re.IGNORECASE)

EMOJI_OBFUSCATION_PATTERN = re.compile(
    r'[\U0001F170-\U0001F171\U0001F17E-\U0001F17F'
    r'\U0001F191-\U0001F19A'
    r'\U0001F1E0-\U0001F1FF'
    r'\U00002648-\U00002653'
    r'\U0001F004\U0001F0CF]{2,}',
    re.UNICODE
)

GUIDELINE_PROMPT = """Kamu adalah classifier spam komentar YouTube Indonesia untuk penelitian akademik.

PRINSIP UTAMA: Jika ragu antara NOT_SPAM dan SPAM, pilih SPAM.
UNCERTAIN hanya untuk kasus yang benar-benar tidak ada sinyal APAPUN.

=== RULES (URUTAN PRIORITAS) ===

RULE 1 — BRAND MENTION (prioritas tertinggi, override semua konteks):
Brand judol dikenal → SPAM apapun isi kalimatnya, termasuk komentar negatif tentang brand.
Brand: ALEXIS17, JUNIOR88, BERKAH99, TKP303, MIYA88, PSTOTO99,
PULAU777, PULAUWIN, PULAU77, MANDALIKA77, PROBET855, JPTOGEL77,
KING328, BUKIT888, MAHJONG303, MAHJONG88, AERO, AERO77, AERO88,
GATES77, XUXU4D, GARUDAHOK, GARUDA88, KOISLOT
PENTING: Komentar NEGATIF tentang brand tetap SPAM karena menyebut brand judol.

RULE 2 — OBFUSKASI BRAND (dikenal maupun tidak dikenal) → SPAM
Teknik: spasi (P U L A U W I N), unicode math (𝐏𝐔𝐋𝐀𝐔𝐖𝐈𝐍),
fullwidth (ＰＵＬＡＵＷＩＮ), emoji huruf (🅿🆄🅻🅰🆄),
emoji+angka membentuk pola brand, Cyrillic mixed (РULAU)

RULE 3 — GAMBLING TERMINOLOGY KUAT (>=1 sinyal + konteks judi) → SPAM
Sinyal: WD, withdraw, deposit, daftar sekarang, maxwin, scatter,
jackpot/JP, gacor (konteks judi), modal receh, cuan (konteks judi)

RULE 4 — SOFT MARKETING judol → SPAM
Pola: "Ketik di Google [nama]", "sawer tipis baliknya tebel",
"main santai tapi cuan", ajakan implisit ke platform judi

RULE 5 — SLANG/KONTEKS AMBIGU tanpa sinyal lain → NOT_SPAM
Username mention (@nama) bukan brand → NOT_SPAM
Opini/diskusi/humor → NOT_SPAM
Emoji tanpa konteks judol → NOT_SPAM

RULE 6 — UNCERTAIN: HANYA jika tidak ada sinyal sama sekali dan
teks terlalu pendek/tidak jelas untuk diklasifikasi.

=== FEW-SHOT EXAMPLES ===

INPUT: "A L E X I S 1 7 memang rekomendasi terbaik."
OUTPUT: {"label": "SPAM", "confidence": "HIGH", "reason": "brand ALEXIS17 obfuskasi spasi"}

INPUT: "AERO BIKIN BANGKRUT JGN DI DOWNLOAD"
OUTPUT: {"label": "SPAM", "confidence": "HIGH", "reason": "brand AERO disebut meski konteks negatif tetap SPAM"}

INPUT: "🅿🆄🅻🅰🆄🆆🅸🅽 tuh selalu muncul pas lagi pengen healing"
OUTPUT: {"label": "SPAM", "confidence": "HIGH", "reason": "brand PULAUWIN obfuskasi emoji box letter"}

INPUT: "🎰 Main santai tapi berisi di 🎰🎰🎰 ♟ 🎲 🃏"
OUTPUT: {"label": "SPAM", "confidence": "HIGH", "reason": "soft marketing judol dengan emoji gambling"}

INPUT: "Gates gacor parah, lgsng WD 3 juta"
OUTPUT: {"label": "SPAM", "confidence": "HIGH", "reason": "WD + nominal uang + gambling terminology"}

INPUT: "Sawer tipis baliknya tebel Ketik di Google"
OUTPUT: {"label": "SPAM", "confidence": "HIGH", "reason": "soft marketing judol CTA Ketik di Google"}

INPUT: "@chanelr124 sudah rahasia umum sih om"
OUTPUT: {"label": "NOT_SPAM", "confidence": "HIGH", "reason": "mention username YouTube bukan brand judol"}

INPUT: "Judol = HAMA"
OUTPUT: {"label": "NOT_SPAM", "confidence": "HIGH", "reason": "komentar anti-judol tidak promosi tidak sebut brand"}

INPUT: "Season 10 kalo ga salah malah ada issue sabotase chef juna"
OUTPUT: {"label": "NOT_SPAM", "confidence": "HIGH", "reason": "diskusi TV show tanpa sinyal judol"}

=== OUTPUT FORMAT ===
Balas HANYA JSON tanpa teks lain:
{"label": "SPAM|NOT_SPAM|UNCERTAIN", "confidence": "HIGH|MEDIUM|LOW", "reason": "max 20 kata"}"""


# ── HELPERS ────────────────────────────────────────────────────────────────

def safe_read_csv(filepath: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"   Loaded '{filepath}' | encoding: {enc}")
            return df
        except Exception:
            continue
    raise ValueError(f"Tidak bisa baca: {filepath}")

def normalize_unicode(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    for zw in ZW_CHARS:
        normalized = normalized.replace(zw, "")
    return normalized

def truncate_text(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_CHARS:
        return text, False
    half = MAX_CHARS // 2
    return text[:half] + " [...dipotong...] " + text[-half:], True

def rule_based_precheck(text: str) -> dict | None:
    text_clean = USERNAME_MENTION_PATTERN.sub("", text)
    normalized = unicodedata.normalize("NFKC", text_clean.upper())
    no_space   = re.sub(r'(?<=[A-Z])\s+(?=[A-Z0-9])', '', normalized)
    for zw in ZW_CHARS:
        no_space = no_space.replace(zw, "")

    for brand in KNOWN_BRANDS:
        if brand in normalized or brand in no_space:
            return {"label": "SPAM", "confidence": "HIGH",
                    "reason": f"known brand: {brand}", "source": "rule_based"}

    if EMOJI_OBFUSCATION_PATTERN.search(text):
        return {"label": "SPAM", "confidence": "MEDIUM",
                "reason": "emoji obfuskasi pola brand judol", "source": "rule_based_emoji"}
    return None

def label_comment(comment_text: str, retries: int = 3) -> dict:
    precheck = rule_based_precheck(comment_text)
    if precheck:
        return precheck

    processed, truncated = truncate_text(normalize_unicode(str(comment_text)))
    note = "\n[Teks dipotong karena terlalu panjang]" if truncated else ""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": GUIDELINE_PROMPT},
            {"role": "user",   "content": f'INPUT: "{processed}"{note}\nOUTPUT:'}
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_ctx": 4096}
    }

    for attempt in range(retries):
        try:
            resp   = requests.post(OLLAMA_URL, json=payload, timeout=120)
            raw    = resp.json()["message"]["content"]
            raw    = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw    = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(raw)
            result["source"] = "llm"
            if truncated:
                result["reason"] = "[truncated] " + result.get("reason", "")
            if result.get("label") not in {"SPAM", "NOT_SPAM", "UNCERTAIN"}:
                result["label"] = "UNCERTAIN"
            if result.get("confidence") not in {"HIGH", "MEDIUM", "LOW"}:
                result["confidence"] = "LOW"
            return result
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"label": "UNCERTAIN", "confidence": "LOW",
                        "reason": f"error: {str(e)[:80]}", "source": "error"}


# ── RESET ──────────────────────────────────────────────────────────────────

def reset_labels():
    df = safe_read_csv(INPUT_CSV)
    for col in ["llm_label", "llm_confidence", "llm_reason", "llm_source"]:
        df[col] = pd.Series([""] * len(df), dtype=str)
    df.to_csv(INPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Reset selesai: {len(df)} rows -> {INPUT_CSV}")


# ── MAIN ───────────────────────────────────────────────────────────────────

def run_labeling():
    df_out = safe_read_csv(INPUT_CSV)
    print(f"Loaded: {len(df_out)} rows")

    for col in ["llm_label", "llm_confidence", "llm_reason", "llm_source"]:
        if col not in df_out.columns:
            df_out[col] = pd.Series([""] * len(df_out), dtype=str)
        else:
            df_out[col] = df_out[col].fillna("").astype(str)

    todo = df_out[df_out["llm_label"] == ""].index
    print(f"Belum dilabel: {len(todo)} | Model: {OLLAMA_MODEL}")
    print("=" * 60)

    if len(todo) == 0:
        print("Semua sudah dilabel!")
        return

    stats = {"SPAM": 0, "NOT_SPAM": 0, "UNCERTAIN": 0,
             "rule_based": 0, "rule_based_emoji": 0, "llm": 0, "error": 0}
    start = time.time()

    for i, idx in enumerate(todo):
        result = label_comment(str(df_out.at[idx, "comment_text"]))
        df_out.loc[idx, "llm_label"]      = str(result.get("label", "UNCERTAIN"))
        df_out.loc[idx, "llm_confidence"] = str(result.get("confidence", "LOW"))
        df_out.loc[idx, "llm_reason"]     = str(result.get("reason", ""))
        df_out.loc[idx, "llm_source"]     = str(result.get("source", "unknown"))

        lbl = result.get("label", "UNCERTAIN")
        src = result.get("source", "llm")
        stats[lbl] = stats.get(lbl, 0) + 1
        stats[src] = stats.get(src, 0) + 1

        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(todo):
            df_out.to_csv(INPUT_CSV, index=False, encoding="utf-8-sig")
            elapsed   = time.time() - start
            remaining = (elapsed / (i + 1)) * (len(todo) - i - 1)
            print(
                f"[{i+1:4d}/{len(todo)}] "
                f"SPAM={stats['SPAM']} NOT_SPAM={stats['NOT_SPAM']} "
                f"UNCERTAIN={stats['UNCERTAIN']} | "
                f"rule={stats['rule_based']} emoji={stats['rule_based_emoji']} "
                f"llm={stats['llm']} | "
                f"ETA: {remaining/60:.1f} min"
            )

    print(f"\nLabeling selesai! -> {INPUT_CSV}")


if __name__ == "__main__":
    reset_labels()
    run_labeling()