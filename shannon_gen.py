#!/usr/bin/env python3
import argparse
from analyze import analyze_text, save_author
from generator import TextGenerator
import os
import json

# Parse levels like "char-2" or "word-3"
def parse_level(level):
    try:
        kind, num = level.split('-')
        num = int(num)
        if kind == "char" and num in [0, 1, 2, 3]:
            return kind, num
        if kind == "word" and num in [1, 2, 3]:
            return kind, num
    except:
        pass
    raise ValueError("Level must be char-0..3 or word-1..3")

# --------------- Commands ----------------

def do_analyze(args):
    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()
    model = analyze_text(text)
    save_author(args.author, model, args.outdir)
    print(f"✅ Saved analysis for {args.author} at {args.outdir}/{args.author}")

def do_generate(args):
    kind, num = parse_level(args.level)
    tg = TextGenerator(args.author, data_dir=args.indir, seed=args.seed)

    if args.anchors:
        anchors = [w.strip() for w in args.anchors.split(",")]
        text = tg.generate_with_anchors(kind, num, args.sentences, args.length, anchors)
    else:
        if kind == "char":
            text = tg.generate_chars(num, args.length or 200)
        else:
            text = tg.generate_words(num, args.sentences or 3)

    print(text)

    # save to file if needed
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)

def do_compare(args):
    tg = TextGenerator(args.author, data_dir=args.indir, seed=args.seed)
    print("=== Character Levels ===")
    for lvl in range(4):
        print(f"\n[char-{lvl}]")
        print(tg.generate_chars(lvl, 200))

    print("\n=== Word Levels ===")
    for lvl in range(1, 4):
        print(f"\n[word-{lvl}]")
        print(tg.generate_words(lvl, args.sentences or 2))

def do_blend(args):
    authors = [a.strip() for a in args.authors.split(",")]
    kind, num = parse_level(args.level)

    tg1 = TextGenerator(authors[0], data_dir=args.indir, seed=args.seed)
    tg2 = TextGenerator(authors[1], data_dir=args.indir, seed=(args.seed or 0) + 1)

    output = []
    for i in range(args.sentences or 3):
        tg = tg1 if i % 2 == 0 else tg2
        if kind == "char":
            output.append(tg.generate_chars(num, args.length or 200))
        else:
            output.append(tg.generate_words(num, 1))

    print(" ".join(output))

# --------------- Main CLI ----------------

def main():
    parser = argparse.ArgumentParser(description="Shannon N-gram Language Model CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_an = subparsers.add_parser("analyze", help="Analyze a text and build n-gram tables")
    p_an.add_argument("--author", required=True, choices=["austen", "twain", "doyle"])
    p_an.add_argument("--file", required=True)
    p_an.add_argument("--outdir", default="data")
    p_an.set_defaults(func=do_analyze)

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate text")
    p_gen.add_argument("--author", required=True, choices=["austen", "twain", "doyle"])
    p_gen.add_argument("--level", required=True)  # e.g. char-2
    p_gen.add_argument("--length", type=int, default=None)
    p_gen.add_argument("--sentences", type=int, default=None)
    p_gen.add_argument("--anchors", type=str, default=None)
    p_gen.add_argument("--indir", default="data")
    p_gen.add_argument("--out", default=None)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.set_defaults(func=do_generate)

    # compare
    p_cmp = subparsers.add_parser("compare", help="Compare levels for same author")
    p_cmp.add_argument("--author", required=True, choices=["austen", "twain", "doyle"])
    p_cmp.add_argument("--sentences", type=int, default=2)
    p_cmp.add_argument("--indir", default="data")
    p_cmp.add_argument("--seed", type=int, default=None)
    p_cmp.set_defaults(func=do_compare)

    # blend (bonus)
    p_b = subparsers.add_parser("blend", help="Blend two authors’ styles")
    p_b.add_argument("--authors", required=True)  # "austen,twain"
    p_b.add_argument("--level", required=True)
    p_b.add_argument("--sentences", type=int, default=3)
    p_b.add_argument("--length", type=int, default=200)
    p_b.add_argument("--indir", default="data")
    p_b.add_argument("--seed", type=int, default=None)
    p_b.set_defaults(func=do_blend)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
