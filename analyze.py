#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from statistics import mean, pstdev

# Try to use provided starter_preprocess.py
try:
    from starter_preprocess import TextPreprocessor  # If exists
except ImportError:
    # Minimal fallback if starter_preprocess.py is not available
    class TextPreprocessor:
        def __init__(self, lowercase=True):
            self.lowercase = lowercase

        def clean_text(self, text):
            if self.lowercase:
                text = text.lower()
            # Keep letters, numbers, punctuation .,!?:;' and spaces only
            text = re.sub(r"[^a-z0-9\s\.,!?:;'-]", " ", text)
            return re.sub(r"\s+", " ", text).strip()

        def tokens_words(self, text):
            text = self.clean_text(text)
            return re.findall(r"[a-z0-9]+|[.,!?:;]", text)

        def tokens_chars(self, text):
            text = self.clean_text(text)
            return list(re.sub(r"\s+", " ", text))  # keeps spaces as characters

# Utility to count nested frequencies
def nested_add(d, a, b):
    if a not in d:
        d[a] = {}
    d[a][b] = d[a].get(b, 0) + 1

def trigram_add(d, a, b, c):
    if a not in d:
        d[a] = {}
    if b not in d[a]:
        d[a][b] = {}
    d[a][b][c] = d[a][b].get(c, 0) + 1

def sentence_lengths(words):
    lengths, count = [], 0
    for w in words:
        if w in {".", "!", "?"}:
            if count > 0:
                lengths.append(count)
                count = 0
        elif re.match(r"[a-z0-9]+", w):
            count += 1
    if count > 0:
        lengths.append(count)
    return lengths

def analyze_text(text):
    tp = TextPreprocessor()
    
    # Characters
    chars = tp.tokenize_chars(text)  
    char_uni = Counter(chars)
    char_bi = defaultdict(dict)
    char_tri = defaultdict(lambda: defaultdict(dict))

    for i in range(len(chars) - 1):
        nested_add(char_bi, chars[i], chars[i+1])
    for i in range(len(chars) - 2):
        trigram_add(char_tri, chars[i], chars[i+1], chars[i+2])

    # Words
    words = tp.tokenize_words(text)
    word_uni = Counter([w for w in words if re.match(r"[a-z0-9]+", w)])
    word_bi = defaultdict(dict)
    word_tri = defaultdict(lambda: defaultdict(dict))

    for i in range(len(words) - 1):
        nested_add(word_bi, words[i], words[i+1])
    for i in range(len(words) - 2):
        trigram_add(word_tri, words[i], words[i+1], words[i+2])

    # Sentence statistics
    lens = sentence_lengths(words)
    sentence_data = {
        "count": len(lens),
        "mean": mean(lens) if lens else 0,
        "std": pstdev(lens) if len(lens) > 1 else 0,
        "hist": dict(Counter(lens))
    }

    return {
        "char": {
            "unigram": dict(char_uni),
            "bigram": {k: v for k, v in char_bi.items()},
            "trigram": {k: {k2: v2 for k2, v2 in v.items()} for k, v in char_tri.items()}
        },
        "word": {
            "unigram": dict(word_uni),
            "bigram": {k: v for k, v in word_bi.items()},
            "trigram": {k: {k2: v2 for k2, v2 in v.items()} for k, v in word_tri.items()}
        },
        "sentences": {
            "lengths": sentence_data
        }
    }

def save_author(author, model, out_dir="data"):
    base = os.path.join(out_dir, author)
    os.makedirs(base, exist_ok=True)

    def dump(name, data):
        with open(os.path.join(base, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    dump("char_unigram.json", model["char"]["unigram"])
    dump("char_bigram.json", model["char"]["bigram"])
    dump("char_trigram.json", model["char"]["trigram"])
    dump("word_unigram.json", model["word"]["unigram"])
    dump("word_bigram.json", model["word"]["bigram"])
    dump("word_trigram.json", model["word"]["trigram"])
    dump("sentence_lengths.json", model["sentences"]["lengths"])

def main():
    parser = argparse.ArgumentParser(description="Analyze text into n-gram tables")
    parser.add_argument("--author", required=True, choices=["austen", "twain", "doyle"])
    parser.add_argument("--file", required=True, help="Input .txt file")
    parser.add_argument("--outdir", default="data", help="Output directory")
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    result = analyze_text(text)
    save_author(args.author, result, args.outdir)

    print(f"Saved analysis for {args.author} into {args.outdir}/{args.author}")

if __name__ == "__main__":
    main()
