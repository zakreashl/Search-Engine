from datasets import load_dataset, Dataset
import pandas
import re

stop_words: set = ["a", "to", "the", "is", "in", "and"]

dataset = load_dataset("NeelNanda/wiki-10k")
df = dataset["train"].to_pandas()

def process_clean_text(clean_text: list, stop_words: set) -> list:
    processed = []

    for word in clean_text:
        if word not in stop_words or len(word) > 2:
            processed.append(word)

    return processed


df["clean_text"] = (
    df["text"]
    .str.lower() # Set all letters to lowercase
    .str.replace('[,.!?\n]', '', regex=True) # Remove punctuation
    .str.strip() # Remove trailing white space
    .str.split(" ") # Turn into list of words
    .apply(process_clean_text, stop_words=stop_words) # process the text in a custom function
)

dataset["train"] = Dataset.from_pandas(df)

print(dataset["train"].features)
print(dataset["train"][0]["clean_text"][:300])
