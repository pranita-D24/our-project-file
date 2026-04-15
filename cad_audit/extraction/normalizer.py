import unicodedata

SYMBOL_MAP = {
    "\u00f8": "⌀",   # ø → diameter
    "\u2205": "⌀",   # ∅ → diameter
    "\u00b0": "°",   # degree
    "\u00b1": "±",   # plus-minus
    "\ufffd": "?",   # replacement char
    "\x00":   "",    # null byte
}

def normalize_cad_text(raw: str) -> str:
    """Normalize CAD-specific unicode symbols and strip garbage bytes."""
    if not raw: return ""
    result = []
    for ch in raw:
        if ch in SYMBOL_MAP:
            result.append(SYMBOL_MAP[ch])
        elif unicodedata.category(ch) in ("Cc", "Cs", "Co"):
            # Control chars, surrogates, private use → drop
            continue
        else:
            result.append(ch)
    return "".join(result).strip()
