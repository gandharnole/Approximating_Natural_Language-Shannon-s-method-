#!/usr/bin/env python3
import json
import os
import random
import re
from typing import List, Dict

# -------------------------------
# Utility functions for sampling
# -------------------------------

def build_cdf(counts: Dict[str, int]):
    """Convert {token: count} to cumulative distribution [(p, token), ...]"""
    total = sum(counts.values())
    cdf, cumulative = [], 0
    for token, count in counts.items():
        cumulative += count / total
        cdf.append((cumulative, token))
    cdf[-1] = (1.0, cdf[-1][1])  # Ensure exact 1.0 at end
    return cdf

def sample_from_cdf(cdf, rng):
    """Sample a token from a cumulative distribution list."""
    r = rng.random()
    for p, token in cdf:
        if r <= p:
            return token
    return cdf[-1][1]

# -------------------------------
# Load JSON frequency tables
# -------------------------------

class NGramTables:
    def __init__(self, base_dir="data", author="austen"):
        path = os.path.join(base_dir, author)
        self.char_unigram = self._load(path, "char_unigram.json")
        self.char_bigram = self._load(path, "char_bigram.json")
        self.char_trigram = self._load(path, "char_trigram.json")
        self.word_unigram = self._load(path, "word_unigram.json")
        self.word_bigram = self._load(path, "word_bigram.json")
        self.word_trigram = self._load(path, "word_trigram.json")
        self.sentence_lengths = self._load(path, "sentence_lengths.json")

    def _load(self, base_path, filename):
        with open(os.path.join(base_path, filename), "r", encoding="utf-8") as f:
            return json.load(f)

# -------------------------------
# Text Generator
# -------------------------------

class TextGenerator:
    def __init__(self, author, data_dir="data", seed=None):
        self.tables = NGramTables(data_dir, author)
        self.rng = random.Random(seed)

        # Precompute CDFs
        self.char_vocab = list(self.tables.char_unigram.keys())
        self.word_vocab = list(self.tables.word_unigram.keys())
        self.char_uni_cdf = build_cdf(self.tables.char_unigram)
        self.word_uni_cdf = build_cdf(self.tables.word_unigram)

        # Sentence length distribution
        self.sent_lengths = []
        for length, count in self.tables.sentence_lengths["hist"].items():
            self.sent_lengths += [int(length)] * int(count)
        if not self.sent_lengths:
            self.sent_lengths = [8, 10, 12, 15]

    # ------------------ Character-level ------------------

    def generate_chars(self, level, length):
        if level == 0:  # Zero-order (random)
            return "".join(self.rng.choice(self.char_vocab) for _ in range(length))

        if level == 1:  # Unigram
            return "".join(sample_from_cdf(self.char_uni_cdf, self.rng) for _ in range(length))

        if level == 2:  # Bigram
            result = [sample_from_cdf(self.char_uni_cdf, self.rng)]
            for _ in range(length - 1):
                prev = result[-1]
                next_chars = self.tables.char_bigram.get(prev, None)
                if not next_chars:
                    result.append(sample_from_cdf(self.char_uni_cdf, self.rng))
                else:
                    result.append(sample_from_cdf(build_cdf(next_chars), self.rng))
            return "".join(result)

        if level == 3:  # Trigram
            result = [
                sample_from_cdf(self.char_uni_cdf, self.rng),
                sample_from_cdf(self.char_uni_cdf, self.rng)
            ]
            for _ in range(length - 2):
                c1, c2 = result[-2], result[-1]
                next_chars = self.tables.char_trigram.get(c1, {}).get(c2, None)
                if not next_chars:  # backoff to bigram
                    next_chars = self.tables.char_bigram.get(c2, None)
                    if not next_chars:
                        result.append(sample_from_cdf(self.char_uni_cdf, self.rng))
                    else:
                        result.append(sample_from_cdf(build_cdf(next_chars), self.rng))
                else:
                    result.append(sample_from_cdf(build_cdf(next_chars), self.rng))
            return "".join(result)

    # ------------------ Word-level ------------------

    def _sample_sentence_length(self):
        return self.rng.choice(self.sent_lengths)

    def _postprocess(self, words: List[str]):
        text = " ".join(words)
        text = re.sub(r"\s+([.,!?:;])", r"\1", text).strip()
        if text and text[-1] not in ".!?":
            text += "."
        return text.capitalize()

    def generate_words(self, level, sentence_count):
        sentences = []
        for _ in range(sentence_count):
            L = max(5, self._sample_sentence_length())

            if level == 1:
                words = [sample_from_cdf(self.word_uni_cdf, self.rng) for _ in range(L)]

            elif level == 2:
                words = [sample_from_cdf(self.word_uni_cdf, self.rng)]
                for _ in range(L - 1):
                    prev = words[-1]
                    next_words = self.tables.word_bigram.get(prev, None)
                    if not next_words:
                        words.append(sample_from_cdf(self.word_uni_cdf, self.rng))
                    else:
                        words.append(sample_from_cdf(build_cdf(next_words), self.rng))

            elif level == 3:
                words = [
                    sample_from_cdf(self.word_uni_cdf, self.rng),
                    sample_from_cdf(self.word_uni_cdf, self.rng)
                ]
                for _ in range(L - 2):
                    w1, w2 = words[-2], words[-1]
                    next_words = self.tables.word_trigram.get(w1, {}).get(w2, None)
                    if not next_words:
                        next_words = self.tables.word_bigram.get(w2, None)
                        if not next_words:
                            words.append(sample_from_cdf(self.word_uni_cdf, self.rng))
                        else:
                            words.append(sample_from_cdf(build_cdf(next_words), self.rng))
                    else:
                        words.append(sample_from_cdf(build_cdf(next_words), self.rng))

            sentences.append(self._postprocess(words))

        return " ".join(sentences)

    # ------------------ Anchors ------------------

    def generate_with_anchors(self, level_type, level_number, sentences, length, anchors, max_attempts=50):
        anchors = [a.strip().lower() for a in anchors if a.strip()]

        for _ in range(max_attempts):
            if level_type == "char":
                text = self.generate_chars(level_number, length)
            else:
                text = self.generate_words(level_number, sentences)

            flat = re.sub(r"[^a-z0-9\s]", " ", text.lower())
            if all(re.search(rf"\b{re.escape(a)}\b", flat) for a in anchors):
                return text

        # fallback: add anchor sentence
        return text + " " + self._postprocess(anchors)

