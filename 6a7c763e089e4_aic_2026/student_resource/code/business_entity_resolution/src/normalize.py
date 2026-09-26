"""Text normalisation for business names and addresses.

Two layers:
1. Base tokenisation (`base_tokens`, `base_components`) - deterministic, dictionary-free. Shared by
   the dictionary miner (mine_dictionaries.py) and the cleaner, so mined keys live in the same space.
2. `Normalizer` - applies the dictionaries mined from training pairs (noisy form -> Source-1 form),
   hand-written legal-suffix / French maps, and an offline transliteration fallback (anyascii),
   then produces the structured fields used by blocking and matching.

Note: Python's regex `\\w` does not treat Indic vowel signs as word characters, so tokens are split
on an explicit punctuation set instead of `\\w+`.
"""
import json
import re
import unicodedata
from pathlib import Path

from anyascii import anyascii

# --------------------------------------------------------------------------------------------------
# Layer 1: base tokenisation
# --------------------------------------------------------------------------------------------------
_SEP = re.compile(r"[\s,;:()\[\]{}<>*|\"`~!?@$%^+=_\\​‌‍﻿]+")
_DOT_ABBR = re.compile(r"(?<=[a-z])\.(?=[a-z])")         # l.l.c -> llc, h.no -> hno
_HASH_NUM = re.compile(r"#(?=\d)")                          # B-#20 -> B-20
_EDGE = re.compile(r"^[\-/#.']+|[\-/.']+$")
_LATIN_EXT = re.compile(r"[À-ɏḀ-ỿ]")
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_DOMAIN = re.compile(r"\b(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|com|net|org|in|co|io|biz|info|us|fr)\b")


def _fold_latin(tok: str) -> str:
    """Strip accents from Latin tokens only (never touch Indic combining marks)."""
    if _LATIN_EXT.search(tok) and not re.search(r"[^\x00-ɏḀ-ỿ]", tok):
        return anyascii(tok).lower()
    return tok


def _prep(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    t = t.replace("&", " and ").replace("’", "").replace("'", "")
    t = _HASH_NUM.sub("", t)
    t = _DOT_ABBR.sub("", t)
    return t.replace(".", " ")


def base_tokens(text: str) -> list:
    """Lower-cased, accent-folded tokens; punctuation-only tokens dropped. '/', '-' kept inside tokens."""
    out = []
    for tok in _SEP.split(_prep(text)):
        if tok == "#":
            out.append(tok)
            continue
        tok = _EDGE.sub("", tok)
        if tok and re.search(r"[^\-/#]", tok):
            out.append(_fold_latin(tok))
    return out


def base_name_tokens(text: str) -> list:
    """Name tokenisation: web domains reduced to their label (acme.com -> acme), hyphens split
    (noise joins words as 'sons-private'), then base tokens. Returns (tokens, is_domain)."""
    low = unicodedata.normalize("NFKC", text).casefold()
    is_domain = bool(_DOMAIN.search(low))
    if is_domain:
        low = _DOMAIN.sub(r"\1", low)
    return base_tokens(low.replace("-", " ")), is_domain


def base_components(text: str) -> list:
    """Comma-separated address components, each as a list of base tokens (empty components dropped)."""
    comps = []
    for part in text.split(","):
        toks = base_tokens(part)
        if toks:
            comps.append(toks)
    return comps


# --------------------------------------------------------------------------------------------------
# Hand-written, language-level knowledge (no external data): legal suffixes, honorifics, aliases.
# --------------------------------------------------------------------------------------------------
LEGAL_CANON = {
    "private": "pvt", "pvt": "pvt", "pvtltd": "pvt ltd", "limited": "ltd", "ltd": "ltd",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "plc": "plc", "pc": "pc",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cos": "co",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "sca": "sca", "selarl": "selarl",
    # frequent OCR-style corruptions of suffixes seen in mining
    "lnc": "inc", "c0": "co", "c0rp": "corp", "c0rporation": "corp", "1td": "ltd", "l1c": "llc",
}
# Generic descriptor words that the noise process appends to names (mined insertion statistics).
# Stripped only from the *end* of the core name, and only if something remains.
GENERIC_TAIL = {"center", "centre", "services", "service", "partners", "enterprises", "enterprise",
                "district", "trust", "council", "society", "foundation", "association", "board",
                "commission", "authority", "federation", "trading", "ventures", "group", "associates",
                "industries", "clinic", "company"}
# multi-token legal phrases, longest first
LEGAL_PHRASES = [
    (("limited", "liability", "company"), "llc"),
    (("limited", "liability", "partnership"), "llp"),
    (("private", "limited"), "pvt ltd"),
    (("pvt", "ltd"), "pvt ltd"),
    (("pvt", "limited"), "pvt ltd"),
    (("private", "ltd"), "pvt ltd"),
    (("and", "co"), "and co"),
]
HONORIFICS = {"the", "m/s", "messrs", "ms", "mr", "mrs", "smt", "dr", "sri", "shri", "shree", "sree"}
ALIAS_MARKERS = {"dba", "d/b/a", "formerly", "aka", "a/k/a", "fka", "f/k/a", "t/a", "dba-"}
ALIAS_PHRASES = [("trading", "as"), ("doing", "business", "as"), ("also", "known", "as")]
LANDMARK_WORDS = {"near", "nr", "opp", "opposite", "behind", "beside", "adjacent", "adj", "next"}

# Address abbreviation maps for countries without training labels (language knowledge, not lookup).
HAND_ADDR_MAPS = {
    "france": {
        "r": "rue", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard", "av": "avenue",
        "ave": "avenue", "pl": "place", "chem": "chemin", "che": "chemin", "imp": "impasse",
        "all": "allee", "rte": "route", "fg": "faubourg", "fbg": "faubourg", "qu": "quai",
        "sq": "square", "crs": "cours", "pass": "passage", "st": "saint", "ste": "sainte",
        "res": "residence", "zi": "zone industrielle", "za": "zone artisanale", "bis": "bis",
    },
}

_OCR_DIGITS = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"}
_OCR_RE = re.compile(r"(?<=[a-z])[013457](?=[a-z])")
_HAS_DIGIT = re.compile(r"\d")
_POSTCODE = re.compile(r"^\d{5,6}$")


def _replace_phrases(toks, phrases):
    """Greedy left-to-right multi-token phrase replacement."""
    out, i = [], 0
    while i < len(toks):
        for src, dst in phrases:
            n = len(src)
            if tuple(toks[i:i + n]) == src:
                out.extend(dst.split())
                i += n
                break
        else:
            out.append(toks[i])
            i += 1
    return out


def _split_aliases(toks):
    """Split a token list on DBA / formerly / aka markers -> list of alias token lists."""
    toks = _replace_phrases(toks, [(p, "dba") for p in ALIAS_PHRASES])
    parts, cur = [], []
    for t in toks:
        if t in ALIAS_MARKERS:
            if cur:
                parts.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        parts.append(cur)
    return parts or [[]]


# --------------------------------------------------------------------------------------------------
# Layer 2: dictionary-driven normaliser
# --------------------------------------------------------------------------------------------------
class Normalizer:
    """Applies mined maps per country. Unknown countries (e.g. France) fall back to the '_ALL' maps,
    with any hand-written country map taking precedence."""

    def __init__(self, dict_dir):
        dict_dir = Path(dict_dir)
        self.name_tok = json.loads((dict_dir / "name_token_map.json").read_text("utf-8"))
        self.addr_tok = json.loads((dict_dir / "addr_token_map.json").read_text("utf-8"))
        self.addr_comp = json.loads((dict_dir / "addr_component_map.json").read_text("utf-8"))

    # ---- helpers -----------------------------------------------------------------------------
    def _maps(self, table, country):
        c = country.strip().lower()
        m = table.get(c)
        if m is None:
            m = dict(table.get("_all", {}))
            m.update(HAND_ADDR_MAPS.get(c, {}) if table is self.addr_tok else {})
        return m

    @staticmethod
    def _translit(tok):
        """Offline romanisation fallback for tokens not covered by mined maps."""
        if _NON_ASCII.search(tok):
            tok = re.sub(r"[^a-z0-9/\-]", "", anyascii(tok).lower())
        return tok

    def _map_tokens(self, toks, tmap):
        out = []
        for t in toks:
            m = tmap.get(t)
            if m is not None:
                out.extend(m.split())
            else:
                t = self._translit(t)
                if t:
                    out.append(t)
        return out

    # ---- names -------------------------------------------------------------------------------
    def clean_name(self, raw: str, country: str) -> dict:
        btoks, is_domain = base_name_tokens(raw)
        toks = self._map_tokens(btoks, self._maps(self.name_tok, country))
        toks = [_OCR_RE.sub(lambda m: _OCR_DIGITS[m.group()], t) for t in toks]
        toks = _replace_phrases(toks, LEGAL_PHRASES)
        toks = [LEGAL_CANON.get(t, t) for t in toks]
        toks = " ".join(toks).split()                       # re-split multi-token canon values

        aliases = _split_aliases(toks)
        cores, legal = [], []
        for a in aliases:
            core = [t for t in a if t not in LEGAL_CANON.values() and t not in {"pvt", "ltd"}]
            legal += [t for t in a if t in LEGAL_CANON.values()]
            while core and core[0] in HONORIFICS:
                core = core[1:]
            while len(core) > 1 and core[-1] in GENERIC_TAIL:
                core = core[:-1]
            cores.append(core or a)
        flat = [t for a in aliases for t in a]
        return {
            "name_clean": " ".join(flat),
            "name_core": " ".join(cores[0]) if len(cores) == 1 else " ".join(max(cores, key=len)),
            "name_alias": " | ".join(" ".join(c) for c in cores) if len(cores) > 1 else "",
            "name_legal": " ".join(sorted(set(legal))),
            "f_name_domain": is_domain,
            "f_name_alias": len(cores) > 1,
        }

    # ---- addresses ---------------------------------------------------------------------------
    def clean_address(self, raw: str, country: str) -> dict:
        cmap = self._maps(self.addr_comp, country)
        tmap = self._maps(self.addr_tok, country)
        comps, landmarks = [], []
        for ctoks in base_components(raw):
            key = " ".join(ctoks)
            mapped = cmap.get(key)
            ctoks = mapped.split() if mapped is not None else self._map_tokens(ctoks, tmap)
            if not ctoks:
                continue
            (landmarks if ctoks[0] in LANDMARK_WORDS else comps).append(ctoks)

        nums, postcode = [], ""
        for i, c in enumerate(comps):
            for j, t in enumerate(c):
                if _HAS_DIGIT.search(t):
                    if t.isdigit() and len(t) < 5:
                        t = t.lstrip("0") or "0"            # 004 == 4
                    if _POSTCODE.match(t) and j == len(c) - 1 and (i, j) != (0, 0):
                        postcode = t
                    elif t not in nums:
                        nums.append(t)
        first_num = re.search(r"\d+", nums[0]).group() if nums else ""
        return {
            "addr_clean": " ".join(t for c in comps for t in c),
            "addr_comps": "|".join(" ".join(c) for c in comps),
            "addr_landmark": "|".join(" ".join(c) for c in landmarks),
            "addr_nums": " ".join(nums),
            "addr_first_num": first_num,
            "addr_postcode": postcode,
        }

    # ---- record ------------------------------------------------------------------------------
    def clean_record(self, name: str, address: str, country: str) -> dict:
        rec = self.clean_name(name, country)
        rec.update(self.clean_address(address, country))
        rec["f_name_native"] = bool(re.search(r"[^\x00-ɏḀ-ỿ -⁯]", name))
        rec["f_addr_native"] = bool(re.search(r"[^\x00-ɏḀ-ỿ -⁯]", address))
        rec["f_name_caps"] = name.isupper()
        rec["f_addr_caps"] = address.isupper()
        rec["f_addr_empty"] = not address.strip()
        rec["f_name_junk"] = bool(re.match(r"^\s*[^\w\s(\[]{2,}", name))
        return rec
