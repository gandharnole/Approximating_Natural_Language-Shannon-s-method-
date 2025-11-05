import os
from generator import TextGenerator

# Create output directory if not exists
os.makedirs("outputs", exist_ok=True)

# Authors we have data for
authors = ["austen", "twain", "doyle"]

def save_text(filename, text):
    with open(os.path.join("outputs", filename), "w", encoding="utf-8") as f:
        f.write(text)

# Generate & Save outputs
for author in authors:
    print(f"Generating outputs for {author}...")
    tg = TextGenerator(author, data_dir="data", seed=42)

    # Character-level (char-0 to char-3)
    for level in range(4):
        text = tg.generate_chars(level, length=500)
        save_text(f"{author}_char_{level}.txt", text)

    # Word-level (word-1 to word-3)
    for level in range(1, 4):
        text = tg.generate_words(level, sentences=5)
        save_text(f"{author}_word_{level}.txt", text)

# Special anchor-word example (as required in assignment)
anchor_text = tg.generate_with_anchors(
    level_type="word",
    level_number=2,
    sentences=3,
    length=None,
    anchors=["Watson", "elementary", "deduce"]
)
save_text("doyle_anchor.txt", anchor_text)

print("\n✅ All outputs generated and saved in 'outputs/' folder!")
