from collections import Counter

corpus = ["low", "lowest", "newer", "wider", "lower", "tallest"]

vocab = Counter([" ".join(word) + " </w>" for word in corpus])

def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

for i in range(7):
    pairs = get_stats(vocab)
    best = pairs.most_common(1)[0][0]
    vocab = merge_vocab(best, vocab)
    print(f"Step {i+1} merge: {best}")
