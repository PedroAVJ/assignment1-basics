import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocabulary: dict[int, bytes] = {byte: byte.to_bytes() for byte in range(256)}

    from collections import defaultdict

    frequency_table: defaultdict[tuple[bytes, ...], int] = defaultdict(int)

    import regex as re

    PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    ## Usage
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token

            special_tokens_pattern = "|".join([re.escape(special_token) for special_token in special_tokens])
            stories = re.split(special_tokens_pattern, chunk)

            for story in stories:
                for pretoken_match in re.finditer(PATTERN, story):
                    pretoken = pretoken_match.group()
                    frequency_table[tuple(bytes([n]) for n in pretoken.encode())] += 1

    merges_count = vocab_size - len(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(merges_count):
        pair_counts: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
        for word, frequency in frequency_table.items():
            from itertools import pairwise

            for left, right in pairwise(word):
                pair_counts[(left, right)] += frequency
        items = list(pair_counts.items())
        new: list[tuple[int, tuple[bytes, bytes]]] = []
        for payload, frequency in items:
            new.append((frequency, payload))
        _, most_frequent_pair = max(new)
        merges.append(most_frequent_pair)
        vocabulary[len(vocabulary)] = b"".join(most_frequent_pair)

        new: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
        for word_bytes, frequency in list(frequency_table.items()):
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

    for special_token in special_tokens:
        vocabulary[len(vocabulary)] = special_token.encode()
    return (vocabulary, merges)
