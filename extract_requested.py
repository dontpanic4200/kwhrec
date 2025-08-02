#!/usr/bin/env python3
# extract_requested.py – enhanced movie-request extraction

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Set, Tuple

from tqdm import tqdm

# ---------------------------------------------------------------------- #
# 0.  Optional spaCy support
# ---------------------------------------------------------------------- #
_NLP = None  # will be lazy-loaded when --use-spacy is requested

def set_spacy_model(model_name: str = "en_core_web_lg") -> None:
    """Load spaCy model if available; otherwise leave _NLP as None."""
    global _NLP
    try:
        import spacy  # type: ignore
        _NLP = spacy.load(model_name)
        print(f"ℹ️  spaCy model '{model_name}' loaded.")
    except ImportError:
        print("⚠️  spaCy not installed; --use-spacy will be ignored.", file=sys.stderr)
    except OSError:
        print(f"⚠️  spaCy model '{model_name}' not found. Run: python -m spacy download {model_name}", file=sys.stderr)
        _NLP = None

# ---------------------------------------------------------------------- #
# 1.  Utility loaders
# ---------------------------------------------------------------------- #

def load_txt_file(path: Optional[str]) -> Set[str]:
    """Load simple newline-delimited TXT into a lower-cased set."""
    if not path:
        return set()
    p = Path(path).expanduser()
    if not p.exists():
        print(f"⚠️  TXT file not found: {p}", file=sys.stderr)
        return set()
    return {line.strip().lower() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_local_titles(json_path: Optional[str]) -> Set[str]:
    """Load a (possibly gzipped) JSON file and return a set of titles (lower-cased)."""
    if not json_path:
        return set()
    path = Path(json_path).expanduser()
    if not path.exists():
        print(f"⚠️  Local JSON file not found: {path}", file=sys.stderr)
        return set()
    open_fn = gzip.open if path.suffix in {".gz", ".gzip"} else open
    try:
        with open_fn(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"⚠️  Failed to parse JSON: {e}", file=sys.stderr)
        return set()
    titles_list: List[str] = []
    if isinstance(data, list):
        if data and isinstance(data[0], str):
            titles_list = data
        else:
            for d in data:
                if isinstance(d, dict):
                    titles_list.append(d.get("original_title") or d.get("title") or "")
    elif isinstance(data, dict):
        titles_list = data.get("titles", [])  # type: ignore
    clean: Set[str] = {t.strip().lower() for t in titles_list if isinstance(t, str) and t.strip()}
    print(f"ℹ️  Loaded {len(clean):,} titles from {path}.")
    return clean

# ---------------------------------------------------------------------- #
# 2.  Text helpers
# ---------------------------------------------------------------------- #

def extract_ngrams(text: str, max_n: int = 3) -> List[str]:
    words = re.findall(r"\b\w[\w']*\b", text.lower())
    return [' '.join(words[i:i + n]) for n in range(1, max_n + 1) for i in range(len(words) - n + 1)]

def fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def is_likely_movie(title: str, min_words: int = 2) -> bool:
    """Determine if a candidate phrase is likely a movie title.

    - Multi-word titles must have at least `min_words` words.
    - Single-word titles are allowed if they have at least 3 letters.
    """
    words = title.split()
    if len(words) == 1:
        return len(words[0]) >= 3
    return len(words) >= min_words

# ---------------------------------------------------------------------- #
# 3.  Main per-file processor
# ---------------------------------------------------------------------- #

def process_file(
        file_path: Path,
        *,
        max_ngram: int,
        min_words: int,
        stopwords: Set[str],
        whitelist: Set[str],
        blacklist: Set[str],
        local_titles: Set[str],
        local_fuzzy: bool,
        local_fuzzy_threshold: float,
        require_unique: bool,
        dry_run: bool,
        use_spacy: bool) -> Tuple[List[str], int]:
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    candidates: Counter[str] = Counter()
    seen: Set[str] = set()
    local_count = 0
    local_hit_counter: Counter[str] = Counter()
    for line in tqdm(lines, desc=f"Scan {file_path.name}"):
        if use_spacy and _NLP:
            doc = _NLP(line)
            for ent in doc.ents:
                if ent.label_ == "WORK_OF_ART":
                    candidates[ent.text.lower()] += 1
        for phrase in extract_ngrams(line, max_ngram):
            if phrase in stopwords or phrase in blacklist:
                continue
            if require_unique and phrase in seen:
                continue
            # 1.  Check local JSON (exact or fuzzy)
            if local_titles and ((phrase in local_titles) or (local_fuzzy and any(fuzzy_score(phrase, lt) >= local_fuzzy_threshold for lt in local_titles))):
                if is_likely_movie(phrase, min_words):
                    # Capital-letter heuristic: require uppercase only for 1-2 words
                    if len(phrase.split()) <= 2:
                        m = re.search(r"\b" + re.escape(phrase) + r"\b", line, re.IGNORECASE)
                        if m:
                            substr = line[m.start():m.end()]
                            if not any(ch.isupper() for ch in substr):
                                continue
                    candidates[phrase] += 1
                    local_count += 1
                    local_hit_counter[phrase] += 1
                    seen.add(phrase)
                    continue

            # 2.  Whitelist / heuristic with capital-letter check
            if phrase in whitelist or is_likely_movie(phrase, min_words):
                m = re.search(r"\b" + re.escape(phrase) + r"\b", line, re.IGNORECASE)
                if m:
                    substr = line[m.start():m.end()]
                    if not any(ch.isupper() for ch in substr):
                        continue
                candidates[phrase] += 1
                seen.add(phrase)
    # Print list of local matches when --dry-run is active
    if dry_run and local_hit_counter:
        print(f"\n[LOCAL MATCH LIST] {file_path.name} – {len(local_hit_counter)} unique titles (total {local_count})")
        for title, freq in local_hit_counter.most_common():
            print(f"  {freq:>3} × {title}")

    if not dry_run:
        out_path = file_path.with_suffix(".titles.txt")
        out_path.write_text("\n".join(t for t, _ in candidates.most_common()), encoding="utf-8")
    return list(candidates), local_count

# ---------------------------------------------------------------------- #
# 4.  Process batch of files
# ---------------------------------------------------------------------- #

def process_files(files: List[Path], **kwargs) -> None:
    total_local = 0
    for f in tqdm(files, desc="Files"):
        _, local_count = process_file(f, **kwargs)
        total_local += local_count
    print("\n=== Grand totals ===")
    print(f"local: {total_local}")

# ---------------------------------------------------------------------- #
# 5.  CLI
# ---------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract movie requests from comments.")
    p.add_argument("files", nargs="*", help="Input .txt files (if not using --folder)")
    p.add_argument("--folder", help="Process every .txt in this folder")
    p.add_argument("--max-ngram", type=int, default=3)
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--require-unique", action="store_true")

    p.add_argument("--stop-file")
    p.add_argument("--whitelist-file")
    p.add_argument("--blacklist-file")

    p.add_argument("--local-json", help="Local JSON(.gz) movie list")
    p.add_argument("--local-fuzzy", action="store_true", help="Fuzzy match local JSON list")
    p.add_argument("--local-fuzzy-threshold", type=float, default=0.97,
                   help="Fuzzy threshold for local JSON (default 0.97)")


    p.add_argument("--use-spacy", action="store_true", help="Enable spaCy WORK_OF_ART NER")
    p.add_argument("--spacy-model", default="en_core_web_lg",
                   help="spaCy model to load (default en_core_web_lg)")

    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    stopwords = load_txt_file(args.stop_file)
    whitelist = load_txt_file(args.whitelist_file)
    blacklist = load_txt_file(args.blacklist_file)
    local_titles = load_local_titles(args.local_json)
    if args.use_spacy:
        set_spacy_model(args.spacy_model)
    if args.folder:
        files = sorted(Path(args.folder).expanduser().glob("*.txt"))
    else:
        files = [Path(f) for f in args.files]
    if not files:
        print("⚠️  No input files found.", file=sys.stderr)
        return
    process_files(
        files,
        max_ngram=args.max_ngram,
        min_words=args.min_words,
        stopwords=stopwords,
        whitelist=whitelist,
        blacklist=blacklist,
        local_titles=local_titles,
        local_fuzzy=args.local_fuzzy,
        local_fuzzy_threshold=args.local_fuzzy_threshold,
        require_unique=args.require_unique,
        dry_run=args.dry_run,
        use_spacy=args.use_spacy and _NLP is not None
    )
if __name__ == "__main__":
    main()
