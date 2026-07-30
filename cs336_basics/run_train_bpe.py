import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures._base import Future
from itertools import pairwise
from typing import BinaryIO

import regex as re

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


def pretokenize(
    input_path: str | os.PathLike, special_tokens: list[str], start: int, end: int
) -> Counter[tuple[bytes, ...]]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token

        special_tokens_pattern = "|".join([re.escape(special_token) for special_token in special_tokens])
        stories = re.split(special_tokens_pattern, chunk)

        frequency_table: Counter[tuple[bytes, ...]] = Counter()

        for story in stories:
            for pretoken_match in re.finditer(PATTERN, story):
                pretoken = pretoken_match.group()
                frequency_table[tuple(bytes([n]) for n in pretoken.encode())] += 1
        return frequency_table


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

    frequency_table: Counter[tuple[bytes, ...]] = Counter()

    ## Usage
    with open(input_path, "rb") as f:
        num_processes = 16
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        futures: list[Future[Counter[tuple[bytes, ...]]]] = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for start, end in pairwise(boundaries):
                future = executor.submit(pretokenize, input_path, special_tokens, start, end)
                futures.append(future)

            for future in futures:
                frequency_table += future.result()

    merges_count = vocab_size - 256 - len(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    pair_to_words: defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    pair_counts: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    for word, frequency in frequency_table.items():
        for left, right in pairwise(word):
            pair_counts[(left, right)] += frequency
            pair_to_words[(left, right)].add(word)

    for _ in range(merges_count):
        most_frequent_pair, _ = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        vocabulary[len(vocabulary)] = b"".join(most_frequent_pair)
        merges.append(most_frequent_pair)

        filtered_words = pair_to_words.pop(most_frequent_pair)

        for word_bytes, frequency in (
            (filtered_word, frequency_table[filtered_word]) for filtered_word in filtered_words
        ):
            frequency_table.pop(word_bytes)
            for left, right in pairwise(word_bytes):
                pair_counts[(left, right)] -= frequency
                if word_bytes in pair_to_words[(left, right)]:
                    pair_to_words[(left, right)].remove(word_bytes)
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
            frequency_table[tuple(new_word)] = frequency
            for left, right in pairwise(tuple(new_word)):
                pair_counts[(left, right)] += frequency
                pair_to_words[(left, right)].add(tuple(new_word))

    for special_token in special_tokens:
        vocabulary[len(vocabulary)] = special_token.encode()
    return (vocabulary, merges)
