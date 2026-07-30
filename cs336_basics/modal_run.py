import pickle
import time
from pathlib import Path

import modal

image = modal.Image.debian_slim(python_version="3.13").pip_install("regex").add_local_python_source("cs336_basics")

volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
app = modal.App("bpe-train", image=image)

TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"


@app.function(volumes={"/data": volume}, timeout=1800)
def download(filename: str = "TinyStoriesV2-GPT4-train.txt") -> str:
    import urllib.request

    dest = Path("/data") / filename
    if dest.exists():
        return f"already present: {dest} ({dest.stat().st_size / 1e9:.2f} GB)"
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(TRAIN_URL, tmp)
    tmp.rename(dest)
    volume.commit()
    return f"downloaded: {dest} ({dest.stat().st_size / 1e9:.2f} GB)"


@app.function(cpu=16, memory=(30 * 1024, 30 * 1024), volumes={"/data": volume}, timeout=3600)
def train(vocab_size: int, input_file: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], float]:
    from cs336_basics.run_train_bpe import run_train_bpe

    start = time.monotonic()
    vocab, merges = run_train_bpe(
        input_path=f"/data/{input_file}",
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    elapsed = time.monotonic() - start
    return vocab, merges, elapsed


@app.local_entrypoint()
def main(vocab_size: int = 10_000, input_file: str = "TinyStoriesV2-GPT4-valid.txt") -> None:
    vocab, merges, elapsed = train.remote(vocab_size, input_file)

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    stem = Path(input_file).stem
    with open(out_dir / f"{stem}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(out_dir / f"{stem}_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    longest = max(vocab.values(), key=len)
    print(f"trained in {elapsed:.1f}s; vocab={len(vocab)}, merges={len(merges)}")
    print(f"longest token: {longest!r}")
    print(f"saved to {out_dir}/{stem}_*.pkl")
