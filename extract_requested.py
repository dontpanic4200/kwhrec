#!/usr/bin/env python3
"""Simple command-line helper to pull movie requests from text files.

The original script was written for power users, so here we sprinkle in
extra comments to explain what each step does.  That way a curious reader
can follow along even without much programming background.
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Set, Tuple

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------- #
# 0.  Optional spaCy support
# ---------------------------------------------------------------------- #
_NLP = None  # spaCy language model, loaded only when requested

def set_spacy_model(model_name: str = "en_core_web_lg") -> None:
    """Load a spaCy model so we can use its "WORK_OF_ART" entity detector.

    If the library or model is missing we simply warn the user and carry on
    without the extra language smarts.
    """
    global _NLP
    try:
        import spacy  # type: ignore
        _NLP = spacy.load(model_name)
        print(f"ℹ️  spaCy model '{model_name}' loaded.")
    except ImportError:
        print("⚠️  spaCy not installed; --use-spacy will be ignored.", file=sys.stderr)
    except OSError:
        print(
            f"⚠️  spaCy model '{model_name}' not found. Run: python -m spacy download {model_name}",
            file=sys.stderr,
        )
        _NLP = None

# ---------------------------------------------------------------------- #
# 1.  Utility loaders
# ---------------------------------------------------------------------- #

def load_txt_file(path: Optional[str]) -> Set[str]:
    """Read a plain text file (one item per line) into a set of phrases."""
    if not path:
        return set()
    p = Path(path).expanduser()  # allow use of "~" in paths
    if not p.exists():  # missing file -> nothing to load
        print(f"⚠️  TXT file not found: {p}", file=sys.stderr)
        return set()
    # strip whitespace, drop empty lines and normalise to lowercase
    return {
        line.strip().lower()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_local_titles(json_path: Optional[str]) -> Set[str]:
    """Load a JSON or JSON.GZ file of known movies into a set of titles."""
    if not json_path:
        return set()
    path = Path(json_path).expanduser()
    print(f"DEBUG load_local_titles: path={path}, exists={path.exists()}")
    if not path.exists():
        print(f"⚠️  Local JSON file not found: {path}", file=sys.stderr)
        return set()
    # Choose gzip.open for .gz files, otherwise the regular open
    open_fn = gzip.open if path.suffix in {".gz", ".gzip"} else open
    try:
        with open_fn(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"⚠️  Failed to parse JSON: {e}", file=sys.stderr)
        return set()
    titles_list: List[str] = []
    # Support several JSON layouts: either a list of strings or a list of dicts
    if isinstance(data, list):
        if data and isinstance(data[0], str):
            titles_list = data
        else:
            for d in data:
                if isinstance(d, dict):
                    titles_list.append(d.get("original_title") or d.get("title") or "")
    elif isinstance(data, dict):  # sometimes titles are under a "titles" key
        titles_list = data.get("titles", [])  # type: ignore
    # Normalise to lower case and drop empty entries
    clean: Set[str] = {
        t.strip().lower() for t in titles_list if isinstance(t, str) and t.strip()
    }
    print(f"ℹ️  Loaded {len(clean):,} titles from {path}.")
    return clean

# ---------------------------------------------------------------------- #
# 2.  Text helpers
# ---------------------------------------------------------------------- #

def extract_ngrams(text: str, max_n: int = 3) -> List[str]:
    """Break a sentence into every 1- to `max_n`-word combination.

    Example: "I love star wars" with max_n=2 ->
    ['i', 'love', 'star', 'wars', 'i love', 'love star', 'star wars']
    """
    words = re.findall(r"\b\w[\w']*\b", text.lower())
    return [
        ' '.join(words[i : i + n])
        for n in range(1, max_n + 1)
        for i in range(len(words) - n + 1)
    ]

def fuzzy_score(a: str, b: str) -> float:
    """Return a number between 0 and 1 showing how similar two strings are."""
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
    online_verify: bool,
    api_key: Optional[str],
    use_spacy: bool,
    debug_local: bool,
) -> Tuple[List[str], int]:
    """Scan one text file and return potential movie titles found within."""

    # Read the file and split into individual lines
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Track how many times each title is seen
    candidates: Counter[str] = Counter()
    seen: Set[str] = set()
    local_count = 0
    local_hit_counter: Counter[str] = Counter()

    for line in tqdm(lines, desc=f"Scan {file_path.name}"):
        # Optional named-entity recognition via spaCy
        if use_spacy and _NLP:
            doc = _NLP(line)
            for ent in doc.ents:
                if ent.label_ == "WORK_OF_ART":
                    candidates[ent.text.lower()] += 1

        # Build n-gram phrases from the line and run them through filters
        for phrase in extract_ngrams(line, max_ngram):
            if phrase in stopwords or phrase in blacklist:
                continue
            if require_unique and phrase in seen:
                continue

            # 1️⃣  Check local JSON list (exact or fuzzy match)
            if local_titles and (
                (phrase in local_titles)
                or (
                    local_fuzzy
                    and any(
                        fuzzy_score(phrase, lt) >= local_fuzzy_threshold
                        for lt in local_titles
                    )
                )
            ):
                if is_likely_movie(phrase, min_words):
                    # For very short titles we demand at least one capital letter
                    if len(phrase.split()) <= 2:
                        m = re.search(r"\b" + re.escape(phrase) + r"\b", line)
                        if m:
                            substr = line[m.start() : m.end()]
                            if not any(ch.isupper() for ch in substr):
                                continue
                    candidates[phrase] += 1
                    local_count += 1
                    local_hit_counter[phrase] += 1
                    seen.add(phrase)
                    continue

            # 2️⃣  Otherwise rely on whitelist/heuristic with capital-letter check
            if phrase in whitelist or is_likely_movie(phrase, min_words):
                m = re.search(r"\b" + re.escape(phrase) + r"\b", line)
                if m:
                    substr = line[m.start() : m.end()]
                    if not any(ch.isupper() for ch in substr):
                        continue
                candidates[phrase] += 1
                seen.add(phrase)

    # Show debug information when dry-running
    if dry_run and local_hit_counter:
        print(
            f"\n[LOCAL MATCH LIST] {file_path.name} – {len(local_hit_counter)} unique titles (total {local_count})"
        )
        for title, freq in local_hit_counter.most_common():
            print(f"  {freq:>3} × {title}")

    if not dry_run:
        out_path = file_path.with_suffix(".titles.txt")
        out_path.write_text(
            "\n".join(t for t, _ in candidates.most_common()), encoding="utf-8"
        )
    return list(candidates), local_count

# ---------------------------------------------------------------------- #
# 4.  Process batch of files
# ---------------------------------------------------------------------- #

def process_files(files: List[Path], **kwargs) -> None:
    """Run :func:`process_file` on each file and print a simple summary."""

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
    """Handle the command-line interface and return the parsed arguments."""

    p = argparse.ArgumentParser(description="Extract movie requests from comments.")
    p.add_argument("files", nargs="*", help="Input .txt files (if not using --folder)")
    p.add_argument("--folder", help="Process every .txt in this folder")
    p.add_argument("--max-ngram", type=int, default=3)
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--require-unique", action="store_true")

    # Files containing helper lists for filtering
    p.add_argument("--stop-file")
    p.add_argument("--whitelist-file")
    p.add_argument("--blacklist-file")

    # Local JSON database of known titles
    p.add_argument("--local-json", help="Local JSON(.gz) movie list")
    p.add_argument(
        "--local-fuzzy",
        action="store_true",
        help="Fuzzy match local JSON list",
    )
    p.add_argument(
        "--local-fuzzy-threshold",
        type=float,
        default=0.97,
        help="Fuzzy threshold for local JSON (default 0.97)",
    )

    p.add_argument("--online-verify", action="store_true")
    p.add_argument("--api-key", help="TMDb API key (needed if --online-verify)")

    p.add_argument(
        "--use-spacy",
        action="store_true",
        help="Enable spaCy WORK_OF_ART NER",
    )
    p.add_argument(
        "--spacy-model",
        default="en_core_web_lg",
        help="spaCy model to load (default en_core_web_lg)",
    )

    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--debug-local",
        action="store_true",
        help="Print similarity scores for local matching",
    )
    return p.parse_args()

def main() -> None:
    """Entry point for the command-line tool."""

    args = parse_args()
    stopwords = load_txt_file(args.stop_file)
    whitelist = load_txt_file(args.whitelist_file)
    blacklist = load_txt_file(args.blacklist_file)
    local_titles = load_local_titles(args.local_json)

    if args.use_spacy:
        set_spacy_model(args.spacy_model)

    # Build the list of files to scan
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
        online_verify=args.online_verify,
        api_key=args.api_key,
        use_spacy=args.use_spacy and _NLP is not None,
        debug_local=args.debug_local,
    )
if __name__ == "__main__":
    main()
