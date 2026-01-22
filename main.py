from datasets import load_dataset

stop_words: set = ["a", "to", "the", "is", "in", "and", "was", "with", "his", "her", "that", "for", "from", "such", 
                   "which", "their", "this", "were", "has", "are", "its", "have", "not", "they", "also", "during",
                   "many", "would", "been", "some", "but", "other", "among", "being", "had", "more", "most", "can",
                   "into", "who", "than", "while", "first", "means", "first", "both", "all", "became", "because"]

dataset = load_dataset("NeelNanda/wiki-10k")
df = dataset["train"].to_pandas()

def process_clean_text(clean_text: list, stop_words: set) -> list:
    processed = []

    for word in clean_text:
        if word not in stop_words and len(word) > 2:
            processed.append(word)

    return processed

def generate_weights(clean_text: list) -> dict:
    weights: dict = {}

    for word in clean_text:
        weights[word] = weights.get(word, 0) + 1.0

    weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))

    for key in weights.keys():
        weights[key] = weights[key] / len(weights)

    return weights


df["clean_text"] = (
    df["text"]
    .str.lower() # Set all letters to lowercase
    .str.replace('[\\(\\);\\:\\-\\–,.!?\n]', '', regex=True) # Remove punctuation
    .str.strip() # Remove trailing white space
    .str.split(" ") # Turn into list of words
    .apply(process_clean_text, stop_words=stop_words) # process the text in a custom function
)

df["weights"] = (
    df["clean_text"]
    .apply(generate_weights)
)

print("Done processing text")

print(df.columns)
print(df["weights"].iloc[0])

