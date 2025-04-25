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


# {(t,h):2, (i,s):4}
def get_pair_stats(splits):
    """Counts the frequency of adjacent pairs in the word splits."""
    pair_counts = collections.defaultdict(int)
    for word_tuple,freq in splits.items():
        symbol = list(word_tuple)
        for i in range(len(symbol)-1):
            pair = (symbol[i],symbol[i+1])
            pair_counts[pair] += freq
    
    return pair_counts

# ### Helper Function: `merge_pair`
# (t , h, i ,s ) ==> (t, h , is)
def merge_pair(pair_to_merge, splits):
    """Merge the specified pair in the words present in split"""
    (first, second) = pair_to_merge
    merged_token = first+second
    new_splits ={}
    for word_token , freq in splits.items():
        symbols = list(word_token)
        new_symbols = []
        i = 0
        while i < len(symbols):
            if(i<len(symbols)-1 and symbols[i]== first and symbols[i+1] == second ):
                new_symbols.append(merged_token)
                i += 2
            else :
                new_symbols.append(symbols[i])
                i+=1
        new_splits[tuple(new_symbols)]= freq
    return new_splits


# --- BPE Training Loop Initialization ---
num_merges = 15
merges = {}
current_splits = word_splits.copy() # Start with initial word splits

print("\n --- Starting BPE(Bite Pair Encoder) Merges ---")
print(f"Initial Splits : {current_splits}")
print("-" * 30)

for i in range(num_merges) :
    print(f"\nMerge Iteration {i+1}/{num_merges}")
    
    # 1 Calculate pair frequencies
    
    pair_stats = get_pair_stats(current_splits)
    if not pair_stats:
        print("No more pairs to merge.")
        break
    
    #Optional
    sorted_pairs = sorted(pair_stats.items(),key=lambda item:item[1], reverse=True)
    print(f"Top 5 pair Frequencies : {sorted_pairs[:5]}")
    
    # Find the best Pair
    
    best_pair = max(pair_stats,key=pair_stats.get)
    best_freq = pair_stats[best_pair]
    
    print(f"Found Best Pair : {best_pair} with Frequency: {best_freq}")
    
    # Merge the Best Pair
    current_splits  = merge_pair(best_pair,current_splits)
    new_token = best_pair[0] + best_pair[1]
    
    print(f"Merging {best_pair} into {new_token}")
    print(f"Splits after merge : {current_splits}")
    
    # Update Vocabulary
    
    vocab.append(new_token)
    print(f"Updated Vocabulary : {vocab}")
    
    # 5. Store Merge Rule
    
    merges[best_pair] = new_token
    print(f"Updated Merges: {merges}")
    
    print("-"*30)

print("\n--- BPE Merges Complete ---")
print(f"Final Vocabulary Size : {len(vocab)}")
print("\nLearned Merges (Pair -> New Token)")


# Pretty print merges

for pair, token in merges.items():
    print(f"{pair} -> {token}")

print("\nFinal Word Splits after all merges:")
print(current_splits)

print("\nFinal Vocabulary (sorted):")
final_vocab_sorted = sorted(list(vocab))
print(final_vocab_sorted)