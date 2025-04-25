# ## Step 1: Prepare Training Data
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

print("Training Corpus :")
for doc in corpus:
    print(doc)

## Step 2: Initialize Vocabulary and Pre-tokenize

unique_chars = set()

for doc in corpus:
    for char in doc:
        unique_chars.add(char)
        

# change to list , for accessing with the index
vocab = list(unique_chars)
vocab.sort() # For consistent order of characters, making the vocabulary list predictable

# add speical end of word token
end_of_word = '</w>'
vocab.append(end_of_word)

print("Initial vocabullary :")
print(vocab)
print(f"Vocabulary Size : {len(vocab)}")

# Pre-tokenize the corpus: Split into words and then characters
# We'll split by space for simplicity and add the end-of-word token

word_splits = {}
for doc in corpus:
    words = doc.split(' ')
    for word in words:
        if word:
            char_list = list(word)+[end_of_word]
            word_tuple = tuple(char_list)
            if word_tuple not in word_splits:
                word_splits[word_tuple]=0
            word_splits[word_tuple] += 1 # Count frequency of each initial word split

print("\nPre-tokenized Word Frequencies")
print(word_splits)

import collections

def get_pair_stats(splits):
    """Counts the frequency of adjacent pairs in the word splits."""
    pair_counts = collections.defaultdict(int)
    for word_tuple,freq in splits.items():
        symbol = list(word_tuple)
        for i in range(len(symbol)-1):
            pair = (symbol[i],symbol[i+1])
            pair_counts[pair] += freq
    
    return pair_counts





