import requests, json, re, time, unicodedata, sys
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ── CONFIG ─────────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_MODEL  = "qwen3:14b"
FULL_DATASET  = "output/raw_75k.csv"
IAA_SUBSET    = "output/iaa_1700_subset.csv"
OUTPUT_CSV    = "output/full_labeled.csv"
SAVE_EVERY    = 50
MAX_CHARS     = 1500
WORKERS       = 24

# ── KNOWN BRANDS ───────────────────────────────────────────────────────────
KNOWN_BRANDS = [
    "ALEXIS17", "JUNIOR88", "BERKAH99", "TKP303", "MIYA88",
    "MANTAP89", "PSTOTO99", "PULAU777", "MANDALIKA77", "APALAGIINI07",
    "PROBET855", "RA77", "JPTOGEL77", "RO88", "LEXIS17",
    "XL777", "ULAU77", "LIATPSTOTO99", "MAINDIDORA77", "KING328",
    "HANYADIKING328", "ANAM1431", "MAKASIPULAU777",
    "CUMADIKING328", "BUKIT888", "LUCKI79",
    "PULAUWIN", "PULAU77", "MAHJONG303", "MAHJONG88",
    "AERO77", "AERO88", "AERO303", "KOISLOT", "KOSLOT",
    "SLOT88", "GATES77", "XUXU4D", "GARUDAHOK", "GARUDA88",
    "VICTORY007", "ROMA4D", "BLACKMAMBA78", "LEVIS4D", "TANGANAGUS88",
]

ZW_CHARS = ['\u200b', '\u200c', '\u200d', '\u2060', '\ufeff', '\u200e', '\u200f']
USERNAME_MENTION_PATTERN = re.compile(r'@[A-Za-z0-9_.]+', re.IGNORECASE)

GAMBLING_EMOJI_PATTERN = re.compile(
    r'[\U0001F170-\U0001F171\U0001F17E-\U0001F17F'
    r'\U0001F191-\U0001F19A'
    r'\U00002648-\U00002653'
    r'\U0001F004\U0001F0CF]{3,}',
    re.UNICODE
)

HOMOGLYPH_MAP = str.maketrans({
    'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H',
    'І': 'I', 'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'P',
    'Т': 'T', 'Х': 'X', 'а': 'a', 'е': 'e', 'о': 'o',
    'р': 'p', 'с': 'c', 'х': 'x',
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H',
    'Ι': 'I', 'Κ': 'K', 'Μ': 'M', 'Ν': 'N', 'Ο': 'O',
    'Ρ': 'P', 'Τ': 'T', 'Υ': 'Y', 'Χ': 'X',
})

LEET_MAP = str.maketrans({'0': 'O', '1': 'I', '3': 'E', '4': 'A', '5': 'S'})


def _map_fancy_char(c):
    cp = ord(c)
    if 0x2460 <= cp <= 0x2469: return str(cp - 0x2460 + 1)
    if 0x2776 <= cp <= 0x277F: return str(cp - 0x2776 + 1)
    if 0x278A <= cp <= 0x2793: return str(cp - 0x278A + 1)
    if 0x2780 <= cp <= 0x2789: return str(cp - 0x2780 + 1)
    if 0x24B6 <= cp <= 0x24CF: return chr(cp - 0x24B6 + ord('A'))
    if 0x24D0 <= cp <= 0x24E9: return chr(cp - 0x24D0 + ord('A'))
    return c


# ── PROMPT ─────────────────────────────────────────────────────────────────
GUIDELINE_PROMPT = """You are an expert annotator specializing in Indonesian YouTube comment \
spam detection, with focus on online gambling promotion.

Your classification must be DETERMINISTIC — follow the rules in order.
Do NOT over-generalize. Do NOT flag ambiguous content as SPAM without \
strong evidence.

═══════════════════════════════════════════════
TASK DEFINITION
═══════════════════════════════════════════════
Classify the given Indonesian YouTube comment as:
- SPAM      : Promotes, advertises, or solicits online gambling/betting/
              slot services — directly or through obfuscation
- NOT_SPAM  : Everything else (opinions, reactions, jokes, emojis,
              unrelated promotions, general content)
- UNCERTAIN : ONLY in very specific cases (see Rule 7)

═══════════════════════════════════════════════
CLASSIFICATION RULES (apply in order)
═══════════════════════════════════════════════

── RULE 1: KNOWN BRAND MATCH → SPAM ──────────────
If the comment contains any of these brand names — including unicode/math
font variants (e.g., 𝕍𝕀ℂ𝕋𝕆ℝ𝕐, 𝐑𝐨𝐦𝐚𝟒𝐝, 𝓟𝓤𝓛𝓐𝓤 in gambling context) —
classify immediately as SPAM:

VICTORY007, ROMA4D, LEVIS4D, BLACKMAMBA78, TANGANAGUS88,
ALEXIS17, JUNIOR88, BERKAH99, TKP303, MIYA88, MANTAP89, PSTOTO99,
PULAU777, MANDALIKA77, PROBET855, JPTOGEL77, KING328, BUKIT888,
MAHJONG303, MAHJONG88, AERO77, AERO88, AERO303, SLOT88, GATES77,
XUXU4D, GARUDAHOK, GARUDA88, PULAUWIN, PULAU77, KOISLOT

→ SPAM (no further analysis needed)

── RULE 2: FLAG EMOJI STRINGS → NOT_SPAM ─────────
Comments consisting ONLY of national flag emojis or mixed flag sequences,
with NO gambling keyword or brand name → NOT_SPAM.

NOT_SPAM: "🇧🇭🇧🇳🇧🇭🇧🇳🇧🇭🇧🇳🇧🇭🇧🇳..."
NOT_SPAM: "🇮🇨🇮🇨🇮🇱🇮🇱🇮🇩🇮🇱🇮🇨🇮🇱..."
NOT_SPAM: "🇨🇿🇩🇯🇨🇿🇩🇯🇨🇿🇩🇯..."
SPAM:     "🎰 Daftar SLOT88 sekarang! 🎰🔥" ← has brand + CTA

── RULE 3: EMOJI-ONLY / REACTION COMMENTS → NOT_SPAM ──
Short comments with only emojis, milestone counts, or brief reactions
with NO gambling signal → NOT_SPAM.

NOT_SPAM: "😭😭😭😭🤝🤝🥹😭😁🙏🙏💪💪"
NOT_SPAM: "Ke 1000 komentar 😊"
NOT_SPAM: "Ke 10..."
NOT_SPAM: ",igi 🇮🇩😎♥️"

── RULE 4: SINGLE KEYWORD WITHOUT CONTEXT → NOT_SPAM ──
Gambling-adjacent keywords ALONE (no brand, no CTA, no monetary promise)
→ NOT_SPAM.

NOT_SPAM: "jackpot"   ← single word, no context
NOT_SPAM: "maxwin"    ← could be slang/joke
SPAM:     "Jackpot terus di ROMA4D, coba sekarang!" ← keyword + brand + CTA

── RULE 5: DOMAIN CONTEXT CHECK ──────────────────
Words that overlap with gambling terms but appear in non-gambling context
→ NOT_SPAM.

NOT_SPAM: "Kesan aerodinamis nya hilang, lebih ke velg bapak-bapak"
          ↑ "AERO" = automotive context
NOT_SPAM: "drama apa lagi ni 😂 dulunya ramadhan max win 😂"
          ↑ "max win" = sarcasm/commentary, not gambling

── RULE 5b: ENDORSEMENT SPAM WITH OBFUSCATED PRAISE ───
Even if a word looks innocent in isolation, classify as SPAM if:
- The comment reads like a testimonial/endorsement
- AND it contains a stylized/spaced brand-like token

SPAM: "Gaya editnya sih 𝓟𝓤𝓛𝓐𝓤... approve 👌"
SPAM: "Bermain di А 𝘌 R 𝐎 𝟪 𝟴 mengubah nasib saya..."

── RULE 6: CONTEXTUAL GAMBLING SIGNALS → SPAM ────
Classify as SPAM if ANY combination of:
(a) Known/suspected brand name
(b) + monetary promise (saldo, cuan, withdraw, deposit, jackpot)
(c) OR invitation/CTA (daftar, coba, klik, gabung, link di bio)
(d) OR gambling-specific terms (slot, gacor, scatter, RTP, maxwin)
WITH additional supporting signal

SPAM: "Baru login sebentar di 𝕍𝕀ℂ𝕋𝕆ℝ𝕐𝟘𝟘𝟟, langsung saldo ngembang"
SPAM: "Bersama 🔥 LEVIS4D 🔥 setiap tantangan jadi lebih mudah!"
SPAM: "dimana-mana ada ℝ𝕠𝕞𝕒𝟜𝕕 wkwk 😭" ← brand mention = sufficient

── RULE 7: UNCERTAIN — HIGHLY RESTRICTED ─────────
Use UNCERTAIN ONLY IF ALL conditions are met:
✓ Contains a brand-like pattern (uppercase + numbers) NOT in known list
✓ AND contains at least ONE gambling signal from Rule 6
✓ AND after applying all rules above, intent is still genuinely unclear
✓ AND the comment is longer than 5 words

DO NOT use UNCERTAIN for:
✗ Emoji-only → Rule 2 or 3
✗ Single keywords → Rule 4
✗ Non-gambling context → Rule 5
✗ Short/vague text → default NOT_SPAM

⚠ Target: UNCERTAIN rate must be < 5% of all comments.

═══════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════
Respond with ONLY valid JSON, no explanation, no markdown:
{{
  "label":      "SPAM" | "NOT_SPAM" | "UNCERTAIN",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reason":     "",
  "source":     "llm"
}}

═══════════════════════════════════════════════
COMMENT:

"{comment}"
═══════════════════════════════════════════════
"""


# ── HELPERS ────────────────────────────────────────────────────────────────
def safe_read_csv(filepath):
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"  Loaded '{filepath}' | encoding: {enc} | rows: {len(df)}")
            return df
        except Exception:
            continue
    raise ValueError(f"Tidak bisa baca: {filepath}")


def normalize_unicode(text):
    normalized = unicodedata.normalize("NFKC", text)
    for zw in ZW_CHARS:
        normalized = normalized.replace(zw, "")
    return normalized


def truncate_text(text):
    if len(text) <= MAX_CHARS:
        return text, False
    half = MAX_CHARS // 2
    return text[:half] + " [...dipotong...] " + text[-half:], True


def rule_based_precheck(text):
    text_clean = USERNAME_MENTION_PATTERN.sub("", text)
    s = unicodedata.normalize("NFKC", text_clean)
    s = s.translate(HOMOGLYPH_MAP)
    s = ''.join(_map_fancy_char(c) for c in s)
    s = s.upper()
    for zw in ZW_CHARS:
        s = s.replace(zw, "")
    normalized = s
    no_space = re.sub(r'[^A-Z0-9]', '', normalized)
    no_space_leet = no_space.translate(LEET_MAP)

    for brand in KNOWN_BRANDS:
        if brand in normalized or brand in no_space or brand in no_space_leet:
            return {
                "label": "SPAM", "confidence": "HIGH",
                "reason": f"known brand: {brand}", "source": "rule_based"
            }

    if GAMBLING_EMOJI_PATTERN.search(text):
        return {
            "label": "SPAM", "confidence": "MEDIUM",
            "reason": "emoji obfuskasi gambling", "source": "rule_based_emoji"
        }

    return None


def label_comment(comment_text, retries=3):
    precheck = rule_based_precheck(comment_text)
    if precheck:
        return precheck

    processed, truncated = truncate_text(normalize_unicode(str(comment_text)))

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": GUIDELINE_PROMPT.format(comment=processed)}
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096,
            "num_predict": 120
        }
    }

    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            raw = resp.json()["message"]["content"]
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                raw = match.group(0)
            result = json.loads(raw)
            result.setdefault("label", "UNCERTAIN")
            result.setdefault("confidence", "LOW")
            result.setdefault("reason", "")
            result["source"] = "llm"
            if truncated:
                result["reason"] = "[truncated] " + result["reason"]
            if result["label"] not in {"SPAM", "NOT_SPAM", "UNCERTAIN"}:
                result["label"] = "UNCERTAIN"
            if result["confidence"] not in {"HIGH", "MEDIUM", "LOW"}:
                result["confidence"] = "LOW"
            return result
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "label": "UNCERTAIN", "confidence": "LOW",
                    "reason": f"error: {str(e)[:80]}", "source": "error"
                }


def atomic_save(records, filepath):
    """Tulis ke .tmp dulu, baru rename — mencegah korupsi file saat crash."""
    tmp = filepath + ".tmp"
    pd.DataFrame(records).to_csv(tmp, index=False, encoding="utf-8-sig")
    Path(tmp).replace(filepath)


# ── MAIN ───────────────────────────────────────────────────────────────────
def run():
    Path("output").mkdir(exist_ok=True)

    # 1. Guard: raw_75k.csv harus sudah ada — script ini TIDAK generate ulang
    if not Path(FULL_DATASET).exists():
        print(f"❌ File tidak ditemukan: '{FULL_DATASET}'")
        print("   Pastikan raw_75k.csv sudah ada di folder output/ sebelum menjalankan script ini.")
        sys.exit(1)

    # 2. Load raw 75k (read-only — tidak pernah ditimpa)
    df_full = safe_read_csv(FULL_DATASET)
    print(f"\nTotal raw: {len(df_full)} rows")

    # 3. Validasi kolom
    if "comment_text" not in df_full.columns:
        print("❌ Kolom 'comment_text' tidak ditemukan!")
        sys.exit(1)

    # 4. Dedup by text
    before = len(df_full)
    df_full = df_full.drop_duplicates(subset="comment_text").reset_index(drop=True)
    print(f"Setelah dedup: {len(df_full)} rows (drop {before - len(df_full)})")

    # 5. Exclude IAA 1700
    df_iaa = safe_read_csv(IAA_SUBSET)
    iaa_texts = set(df_iaa["comment_text"].astype(str))
    df_todo = df_full[~df_full["comment_text"].isin(iaa_texts)].copy()
    df_todo = df_todo.reset_index(drop=True)
    print(f"Setelah exclude IAA 1700: {len(df_todo)} rows to label")

    # 6. Resume support — skip yang sudah dilabel, nimpa OUTPUT_CSV yang sama
    if Path(OUTPUT_CSV).exists():
        df_done = safe_read_csv(OUTPUT_CSV)
        done_texts = set(df_done["comment_text"].astype(str))
        df_todo = df_todo[~df_todo["comment_text"].isin(done_texts)].copy()
        df_todo = df_todo.reset_index(drop=True)
        print(f"Resume: {len(df_todo)} rows belum dilabel (sudah ada {len(df_done)} rows di output)")
        existing_results = df_done.to_dict("records")

        # Hitung stats dari hasil sebelumnya agar log akurat
        stats = {"SPAM": 0, "NOT_SPAM": 0, "UNCERTAIN": 0,
                 "rule_based": 0, "rule_based_emoji": 0, "llm": 0, "error": 0}
        for r in existing_results:
            lbl = r.get("final_label", "UNCERTAIN")
            src = r.get("llm_source", "llm")
            stats[lbl] = stats.get(lbl, 0) + 1
            stats[src] = stats.get(src, 0) + 1
    else:
        existing_results = []
        stats = {"SPAM": 0, "NOT_SPAM": 0, "UNCERTAIN": 0,
                 "rule_based": 0, "rule_based_emoji": 0, "llm": 0, "error": 0}
        print("Fresh start — tidak ada checkpoint sebelumnya")

    if len(df_todo) == 0:
        print("✅ Semua sudah dilabel!")
        return

    print(f"\nMulai labeling {len(df_todo)} komentar...")
    print(f"Model: {OLLAMA_MODEL} | Workers: {WORKERS} | Save every: {SAVE_EVERY}")
    print("=" * 60)

    # 7. Parallel labeling
    lock = threading.Lock()
    completed = [0]
    results = list(existing_results)
    start = time.time()

    def process_one(args):
        idx, row = args
        result = label_comment(str(row["comment_text"]))
        record = {
            "comment_text":   str(row["comment_text"]),
            "llm_label":      result.get("label", "UNCERTAIN"),
            "llm_confidence": result.get("confidence", "LOW"),
            "llm_reason":     result.get("reason", ""),
            "llm_source":     result.get("source", "llm"),
            "final_label":    result.get("label", "UNCERTAIN"),
            "label_source":   result.get("source", "llm"),   # fix: pakai source asli
        }
        for col in ["comment_published_at", "author_display_name",
                    "like_count", "reply_count",
                    "channel_subscriber_count", "channel_video_count",
                    "channel_view_count"]:
            if col in row:
                record[col] = row[col]
        return record, result

    rows_list = list(df_todo.iterrows())

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_one, item): item[0] for item in rows_list}

        for future in as_completed(futures):
            record, result = future.result()

            with lock:
                results.append(record)

                lbl = result.get("label", "UNCERTAIN")
                src = result.get("source", "llm")
                stats[lbl] = stats.get(lbl, 0) + 1
                stats[src] = stats.get(src, 0) + 1
                completed[0] += 1
                n = completed[0]

                if n % SAVE_EVERY == 0 or n == len(df_todo):
                    # Atomic save: tulis .tmp → rename, raw_75k.csv tidak tersentuh
                    atomic_save(results, OUTPUT_CSV)

                    elapsed = time.time() - start
                    remaining = (elapsed / n) * (len(df_todo) - n) if n < len(df_todo) else 0
                    uncertain_pct = stats["UNCERTAIN"] / max(n + len(existing_results), 1) * 100
                    speed = n / elapsed * 60
                    print(
                        f"[{n:5d}/{len(df_todo)}] "
                        f"SPAM={stats['SPAM']:4d} NOT_SPAM={stats['NOT_SPAM']:5d} "
                        f"UNCERTAIN={stats['UNCERTAIN']} ({uncertain_pct:.1f}%) | "
                        f"rule={stats['rule_based']} emoji={stats['rule_based_emoji']} "
                        f"llm={stats['llm']} err={stats['error']} | "
                        f"{speed:.0f} row/min | ETA: {remaining/60:.1f} min"
                    )

    # 8. Summary final
    total_done = len(df_todo)
    print(f"\n{'='*60}")
    print(f"✅ Labeling selesai! -> {OUTPUT_CSV}")
    print(f"  Total dilabel  : {total_done}")
    print(f"  SPAM           : {stats['SPAM']} ({stats['SPAM']/max(total_done,1)*100:.1f}%)")
    print(f"  NOT_SPAM       : {stats['NOT_SPAM']} ({stats['NOT_SPAM']/max(total_done,1)*100:.1f}%)")
    print(f"  UNCERTAIN      : {stats['UNCERTAIN']} ({stats['UNCERTAIN']/max(total_done,1)*100:.1f}%)")
    print(f"  Error          : {stats['error']} ({stats['error']/max(total_done,1)*100:.1f}%)")
    print(f"{'='*60}")
    print(f"\nStep berikutnya: jalankan merge_dataset.py")


if __name__ == "__main__":
    run()