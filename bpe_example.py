from collections import Counter, defaultdict
from itertools import pairwise

corpus = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

vocabulary: dict[int, bytes] = {byte: byte.to_bytes() for byte in range(256)}

pretokenize = corpus.split()
counter = Counter(pretokenize)
frequency_table: dict[tuple[bytes, ...], int] = {
    tuple(bytes([n]) for n in key.encode()): value for key, value in counter.items()
}

for _ in range(6):
    pair_counts: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    for word, frequency in frequency_table.items():
        for left, right in pairwise(word):
            pair_counts[(left, right)] += frequency
    items = list(pair_counts.items())
    new: list[tuple[int, tuple[bytes, bytes]]] = []
    for payload, frequency in items:
        new.append((frequency, payload))
    _, most_frequent_pair = max(new)
    vocabulary[len(vocabulary)] = b"".join(most_frequent_pair)

    new: dict[tuple[bytes, ...], int] = {}
    for word_bytes, frequency in list(frequency_table.items()):
        skip = False
        new_word: list[bytes] = []
        cur = 0
        length = len(word_bytes)
        while cur in range(length):
            next = cur + 1
            if next not in range(length):
                new_word.append(word_bytes[cur])
                break
            if most_frequent_pair == (word_bytes[cur], word_bytes[next]):
                new_word.append(b"".join(most_frequent_pair))
                cur += 1
            else:
                new_word.append(word_bytes[cur])
            cur += 1
        new[tuple(new_word)] = frequency

    frequency_table = new

vocabulary[len(vocabulary)] = b"<|endoftext|>"
