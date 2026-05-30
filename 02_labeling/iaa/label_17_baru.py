import requests, json, re, time, unicodedata, sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ── CONFIG ─────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:14b"
INPUT_CSV    = "iaa_1700_subset.csv"
SAVE_EVERY   = 50
MAX_CHARS    = 1500
WORKERS      = 12  # naikkan ke 12-16 kalau GPU masih santai

KNOWN_BRANDS = [
    "ALEXIS17", "JUNIOR88", "BERKAH99", "TKP303", "MIYA88",
    "MANTAP89", "PSTOTO99", "PULAU777", "MANDALIKA77", "APALAGIINI07",
    "PROBET855", "RA77", "JPTOGEL77", "RO88", "LEXIS17",
    "XL777", "ULAU77", "LIATPSTOTO99", "MAINDIDORA77", "KING328",
    "HANYADIKING328", "ANAM1431", "MAKASIPULAU777",
    "CUMADIKING328", "BUKIT888", "LUCKI79",
    "PULAUWIN", "PULAU77", "MAHJONG303", "MAHJONG88",
    "AERO77", "AERO88", "AERO303", "AERO", "KOISLOT",
    "SLOT88", "GATES77", "XUXU4D", "GARUDAHOK", "GARUDA88",
    "VICTORY007", "ROMA4D", "BLACKMAMBA78", "LEVIS4D", "TANGANAGUS88",
    # ── FIX: variant obfuscation yang sebelumnya terlewat ──────────────────
    "KOSLOT",   # KO!SL0T  → strip simbol → KOSL0T → leet-O → KOSLOT
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

# ── FIX: Homoglyph map (Cyrillic + Greek lookalikes ke Latin) ──────────────
HOMOGLYPH_MAP = str.maketrans({
    # Cyrillic
    'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H',
    'І': 'I', 'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'P',
    'Т': 'T', 'Х': 'X', 'а': 'a', 'е': 'e', 'о': 'o',
    'р': 'p', 'с': 'c', 'х': 'x',
    # Greek
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H',
    'Ι': 'I', 'Κ': 'K', 'Μ': 'M', 'Ν': 'N', 'Ο': 'O',
    'Ρ': 'P', 'Τ': 'T', 'Υ': 'Y', 'Χ': 'X',
})

# ── FIX: Leet-speak map untuk no_space ────────────────────────────────────
LEET_MAP = str.maketrans({'0': 'O', '1': 'I', '3': 'E', '4': 'A', '5': 'S'})

# ── FIX: Dingbat / circled digit/letter mapper ────────────────────────────
def _map_fancy_char(c):
    cp = ord(c)
    if 0x2460 <= cp <= 0x2469: return str(cp - 0x2460 + 1)   # ①–⑩
    if 0x2776 <= cp <= 0x277F: return str(cp - 0x2776 + 1)   # ❶–❿
    if 0x278A <= cp <= 0x2793: return str(cp - 0x278A + 1)   # ➊–➓
    if 0x2780 <= cp <= 0x2789: return str(cp - 0x2780 + 1)   # ➀–➉
    if 0x24B6 <= cp <= 0x24CF: return chr(cp - 0x24B6 + ord('A'))  # Ⓐ–Ⓩ
    if 0x24D0 <= cp <= 0x24E9: return chr(cp - 0x24D0 + ord('A'))  # ⓐ–ⓩ
    return c

GUIDELINE_PROMPT = """You are an expert annotator specializing in Indonesian YouTube comment \
spam detection, with focus on online gambling promotion.

Your classification must be DETERMINISTIC — follow the rules in order.
Do NOT over-generalize. Do NOT flag ambiguous content as SPAM without \
strong evidence.

═══════════════════════════════════════════════
TASK DEFINITION
═══════════════════════════════════════════════
Classify the given Indonesian YouTube comment as:
- SPAM : Promotes, advertises, or solicits online gambling/betting/
slot services — directly or through obfuscation
- NOT_SPAM : Everything else (opinions, reactions, jokes, emojis,
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
SULTAN, MAXWIN88, SLOT88, GACOR77, [... full list ...]

→ SPAM (no further analysis needed)

── RULE 2: FLAG EMOJI STRINGS → NOT_SPAM ─────────
Comments consisting ONLY of national flag emojis or mixed flag sequences,
with NO gambling keyword or brand name → NOT_SPAM.

NOT_SPAM: "🇧🇭🇧🇳🇧🇭🇧🇳🇧🇭🇧🇳🇧🇭🇧🇳..."
NOT_SPAM: "🇮🇨🇮🇨🇮🇱🇮🇱🇮🇩🇮🇱🇮🇨🇮🇱..."
NOT_SPAM: "🇨🇿🇩🇯🇨🇿🇩🇯🇨🇿🇩🇯..."
SPAM: "🎰 Daftar SLOT88 sekarang! 🎰🔥" ← has brand + CTA

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

NOT_SPAM: "jackpot" ← single word, no context
NOT_SPAM: "maxwin" ← could be slang/joke
SPAM: "Jackpot terus di ROMA4D, coba sekarang!" ← keyword + brand + CTA

── RULE 5: DOMAIN CONTEXT CHECK ──────────────────
Words that overlap with gambling terms but appear in non-gambling context
→ NOT_SPAM.

NOT_SPAM: "Kesan aerodinamis nya hilang, lebih ke velg bapak-bapak"
↑ "AERO" = automotive context
NOT_SPAM: "drama apa lagi ni 😂 dulunya ramadhan max win 😂"
↑ "max win" = sarcasm/commentary, not gambling

── RULE 5b: ENDORSEMENT SPAM WITH OBFUSCATED PRAISE ─── ← NEW
Even if a word looks innocent in isolation, classify as SPAM if:
- The comment reads like a testimonial/endorsement ("sukses", "mengubah nasib",
  "member setia", "jempolan", "gokil", "top abis", "makin sukses")
- AND it contains a stylized/spaced brand-like token (Unicode font, spaced letters,
  or symbol-padded name like ⭐ A L E X I S 1 7 ⭐, 𝓟𝓤𝓛𝓐𝓤, 𝑴𝑨𝑵𝑫𝑨𝙇𝙄𝙆𝘼❼❼)
- The endorsement framing + stylized brand token = SPAM regardless of word meaning

SPAM: "Gaya editnya sih 𝓟𝓤𝓛𝓐𝓤... approve 👌"
↑ 𝓟𝓤𝓛𝓐𝓤 = stylized brand token framed as endorsement → SPAM
SPAM: "Kualitasnya sejalan sama review, ⭐ A L E X I S 1 7 ⭐ top abis!"
↑ spaced + star-padded brand token + praise = SPAM
SPAM: "Bermain di А 𝘌 R 𝐎 𝟪 𝟴 mengubah nasib saya..."
↑ mixed-script brand + life-story testimonial = SPAM

── RULE 6: CONTEXTUAL GAMBLING SIGNALS → SPAM ────
Classify as SPAM if the comment has ANY combination of:
(a) Known/suspected brand name
(b) + monetary promise (saldo, cuan, withdraw, deposit, jackpot)
(c) OR invitation/CTA (daftar, coba, klik, gabung, link di bio)
(d) OR gambling-specific terms (slot, gacor, scatter, RTP, maxwin)
WITH additional supporting signal

SPAM: "Baru login sebentar di 𝕍𝕀ℂ𝕋𝕆ℝ𝕐𝟘𝟘𝟟, langsung saldo ngembang"
SPAM: "Bersama 🔥 LEVIS4D 🔥 setiap tantangan jadi lebih mudah!"
SPAM: "Dari yang simple sampe ribet tetep diladenin, kaya layanan di 💎𝕎𝔾..."
SPAM: "dimana-mana ada ℝ𝕠𝕞𝕒𝟜𝕕 wkwk 😭" ← brand mention = sufficient
SPAM: "💡KO!SL0T🧱" ← obfuscated KOISLOT brand
SPAM: "MInimal byar bang kalo komen Judi pul au 7 7 7 kismin"
↑ spaced out PULAU777 brand with gambling keyword (Judi)

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
"label": "SPAM" | "NOT_SPAM" | "UNCERTAIN",
"confidence": "HIGH" | "MEDIUM" | "LOW",
"reason": "",
"source": "llm"
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
            print(f"   Loaded '{filepath}' | encoding: {enc}")
            return df
        except Exception:
            continue
    raise ValueError(f"Tidak bisa baca: {filepath}")

def normalize_unicode(text):
    """Ringan: NFKC + strip ZW — dipakai untuk text yg dikirim ke LLM."""
    normalized = unicodedata.normalize("NFKC", text)
    for zw in ZW_CHARS:
        normalized = normalized.replace(zw, "")
    return normalized

def truncate_text(text):
    if len(text) <= MAX_CHARS:
        return text, False
    half = MAX_CHARS // 2
    return text[:half] + " [...dipotong...] " + text[-half:], True

# ── FIX: rule_based_precheck dengan normalisasi berlapis ──────────────────
def rule_based_precheck(text):
    """
    Multi-layer normalization sebelum brand matching:
      1. Strip @mention
      2. NFKC  (math fonts → ASCII)
      3. Homoglyph map (Cyrillic/Greek → Latin)
      4. Dingbat digit/letter map (❼ → 7, Ⓐ → A)
      5. Uppercase + strip ZW
      6. Strip semua non-alphanumeric → `no_space`
      7. Leet substitution (0→O, 1→I, …) pada `no_space`
    Brand match dilakukan pada tiga representasi: normalized, no_space, no_space_leet.
    """
    # Step 1
    text_clean = USERNAME_MENTION_PATTERN.sub("", text)

    # Step 2: NFKC
    s = unicodedata.normalize("NFKC", text_clean)

    # Step 3: homoglyph
    s = s.translate(HOMOGLYPH_MAP)

    # Step 4: dingbat digits/letters
    s = ''.join(_map_fancy_char(c) for c in s)

    # Step 5: uppercase + strip ZW
    s = s.upper()
    for zw in ZW_CHARS:
        s = s.replace(zw, "")

    normalized = s  # spasi masih ada

    # Step 6: strip semua non-alphanumeric
    no_space = re.sub(r'[^A-Z0-9]', '', normalized)

    # Step 7: leet pada no_space
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

# ── RESET ──────────────────────────────────────────────────────────────────
def reset_labels():
    df = safe_read_csv(INPUT_CSV)
    for col in ["llm_label", "llm_confidence", "llm_reason", "llm_source"]:
        df[col] = pd.Series([""] * len(df), dtype=str)
    df.to_csv(INPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Reset selesai: {len(df)} rows -> {INPUT_CSV}")

# ── MAIN (PARALLEL) ────────────────────────────────────────────────────────
def run_labeling():
    df_out = safe_read_csv(INPUT_CSV)
    print(f"Loaded: {len(df_out)} rows")

    for col in ["llm_label", "llm_confidence", "llm_reason", "llm_source"]:
        if col not in df_out.columns:
            df_out[col] = pd.Series([""] * len(df_out), dtype=str)
        else:
            df_out[col] = df_out[col].fillna("").astype(str)

    todo = df_out[df_out["llm_label"] == ""].index.tolist()
    print(f"Belum dilabel: {len(todo)} | Model: {OLLAMA_MODEL} | Workers: {WORKERS}")
    print("=" * 60)

    if not todo:
        print("Semua sudah dilabel!")
        return

    stats   = {"SPAM": 0, "NOT_SPAM": 0, "UNCERTAIN": 0,
               "rule_based": 0, "rule_based_emoji": 0, "llm": 0, "error": 0}
    lock    = threading.Lock()
    completed = [0]
    start   = time.time()

    def process_one(idx):
        comment = str(df_out.at[idx, "comment_text"])
        result  = label_comment(comment)
        return idx, result

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_one, idx): idx for idx in todo}

        for future in as_completed(futures):
            idx, result = future.result()

            with lock:
                df_out.loc[idx, "llm_label"]      = str(result.get("label",      "UNCERTAIN"))
                df_out.loc[idx, "llm_confidence"]  = str(result.get("confidence", "LOW"))
                df_out.loc[idx, "llm_reason"]      = str(result.get("reason",     ""))
                df_out.loc[idx, "llm_source"]      = str(result.get("source",     "unknown"))

                lbl = result.get("label",  "UNCERTAIN")
                src = result.get("source", "llm")
                stats[lbl] = stats.get(lbl, 0) + 1
                stats[src] = stats.get(src, 0) + 1
                completed[0] += 1
                n = completed[0]

                if n % SAVE_EVERY == 0 or n == len(todo):
                    df_out.to_csv(INPUT_CSV, index=False, encoding="utf-8-sig")
                    elapsed   = time.time() - start
                    remaining = (elapsed / n) * (len(todo) - n)
                    uncertain_pct = stats["UNCERTAIN"] / max(n, 1) * 100
                    speed = n / elapsed * 60
                    print(
                        f"[{n:4d}/{len(todo)}] "
                        f"SPAM={stats['SPAM']} NOT_SPAM={stats['NOT_SPAM']} "
                        f"UNCERTAIN={stats['UNCERTAIN']} ({uncertain_pct:.1f}%) | "
                        f"rule={stats['rule_based']} emoji={stats['rule_based_emoji']} "
                        f"llm={stats['llm']} err={stats['error']} | "
                        f"{speed:.0f} row/min | ETA: {remaining/60:.1f} min"
                    )

    print(f"\n✅ Labeling selesai! -> {INPUT_CSV}")
    print(f"   UNCERTAIN rate final: {stats['UNCERTAIN'] / len(todo) * 100:.1f}%")
    print(f"   Total error: {stats['error']} ({stats['error']/len(todo)*100:.1f}%)")

if __name__ == "__main__":
    if "--reset" in sys.argv:
        reset_labels()
    run_labeling()