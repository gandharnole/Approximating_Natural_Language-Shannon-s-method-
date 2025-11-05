Shannon N-Gram Text Generator

A reproduction of Claude Shannon’s language experiments using Python

📌 Project Overview

This project implements character and word-level N-gram language models to approximate natural language, inspired by Claude Shannon’s 1948 work. It analyzes texts from Jane Austen, Mark Twain, and Arthur Conan Doyle, generates frequency tables, and produces text emulating their writing styles.

📁 Folder Structure
Assignment 3 BoAI/
│── analyze.py          # Build n-gram frequency tables
│── generator.py        # Generates text using trained models
│── shannon_gen.py      # Command-line interface
│── starter_preprocess.py
│── texts/              # Input novels (Austen, Twain, Doyle)
│── data/               # JSON frequency tables (auto-generated)
│── outputs/            # Saved generated samples
│── save_outputs.py     # (Optional) auto-generate outputs
│── requirements.txt
│── README.md

⚙️ Installation
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m nltk.downloader punkt

🧪 Step 1: Analyze Texts

Generate frequency tables for each author:

python shannon_gen.py analyze --author austen --file texts/austen_pride_prejudice.txt
python shannon_gen.py analyze --author twain --file texts/twain_tom_sawyer.txt
python shannon_gen.py analyze --author doyle --file texts/doyle_sherlock_holmes.txt

📝 Step 2: Generate Text
Character-Level
python shannon_gen.py generate --author austen --level char-2 --length 200

Word-Level
python shannon_gen.py generate --author twain --level word-3 --sentences 5

With Anchor Words
python shannon_gen.py generate --author doyle --level word-2 --sentences 3 --anchors "Watson,elementary,deduce"

📊 Step 3: Compare All Levels
python shannon_gen.py compare --author austen --sentences 2

💾 (Optional) Save All Outputs
python save_outputs.py


This creates files like:

outputs/austen_char_0.txt
outputs/austen_word_3.txt
outputs/doyle_anchor.txt

📚 Technologies Used

Python

NLTK

argparse

JSON-based n-gram storage