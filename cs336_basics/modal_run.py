import pickle
import time
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("regex")
    .add_local_python_source("cs336_basics")
)

volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
app = modal.App("bpe-train", image=image)


@app.function(cpu=8, memory=32 * 1024, volumes={"/data": volume}, timeout=3600)
def train(vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], float]:
    from cs336_basics.run_train_bpe import run_train_bpe

    start = time.monotonic()
    vocab, merges = run_train_bpe(
        input_path="/data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    elapsed = time.monotonic() - start
    return vocab, merges, elapsed


@app.local_entrypoint()
def main(vocab_size: int = 10_000) -> None:
    vocab, merges, elapsed = train.remote(vocab_size)

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "tinystories_valid_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(out_dir / "tinystories_valid_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    longest = max(vocab.values(), key=len)
    print(f"trained in {elapsed:.1f}s; vocab={len(vocab)}, merges={len(merges)}")
    print(f"longest token: {longest!r}")
    print(f"saved to {out_dir}/")
