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
vocal = list(unique_chars)

# add speical end of word token
