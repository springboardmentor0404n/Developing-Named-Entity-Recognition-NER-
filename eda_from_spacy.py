import spacy
from spacy.tokens import DocBin
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the binary .spacy file
doc_bin = DocBin().from_disk("train.spacy")
nlp = spacy.blank("en")
docs = list(doc_bin.get_docs(nlp.vocab))


# Entity distribution
entity_counter = Counter()
text_lengths = []

for doc in docs:
    text_lengths.append(len(doc.text))
    for ent in doc.ents:
        entity_counter[ent.label_] += 1

# Print entity counts
print("Entity Distribution:")
for label, count in entity_counter.items():
    print(f"{label}: {count}")

# Plot entity distribution
plt.figure(figsize=(8, 5))
plt.bar(entity_counter.keys(), entity_counter.values(), color="skyblue")
plt.title("Entity Distribution in Financial NER Dataset")
plt.xlabel("Entity Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Plot text length distribution
plt.figure(figsize=(8, 5))
plt.hist(text_lengths, bins=20, color="salmon", edgecolor="black")
plt.title("Text Length Distribution")
plt.xlabel("Text Length (characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Word cloud of all text
all_text = " ".join([doc.text for doc in docs])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Financial Corpus")
plt.show()



from collections import defaultdict

label_texts = defaultdict(list)

for doc in docs:
    for ent in doc.ents:
        label_texts[ent.label_].append(ent.text)

# Count top 5 entities per label
for label, texts in label_texts.items():
    top_texts = Counter(texts).most_common(5)
    print(f"\nTop entities for {label}:")
    for text, count in top_texts:
        print(f"{text}: {count}")
