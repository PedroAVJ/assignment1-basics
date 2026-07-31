import pickle
import time
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("regex", "scalene", "py-spy")
    .add_local_python_source("cs336_basics")
)

volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
app = modal.App("bpe-train", image=image)

TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
OWT_TRAIN_URL = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"


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


@app.function(volumes={"/data": volume}, timeout=7200)
def download_owt(filename: str = "owt_train.txt") -> str:
    """Stream the gzipped OWT sample straight to disk, decompressing as it lands."""
    import gzip
    import shutil
    import urllib.request

    dest = Path("/data") / filename
    if dest.exists():
        return f"already present: {dest} ({dest.stat().st_size / 1e9:.2f} GB)"
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(OWT_TRAIN_URL) as resp, gzip.GzipFile(fileobj=resp) as gz, open(tmp, "wb") as out:
        shutil.copyfileobj(gz, out, length=16 * 1024 * 1024)
    tmp.rename(dest)
    volume.commit()
    return f"downloaded: {dest} ({dest.stat().st_size / 1e9:.2f} GB)"


@app.function(cpu=(64, 64), memory=(100 * 1024, 100 * 1024), volumes={"/data": volume}, timeout=3600)
def profile(vocab_size: int = 10_000, input_file: str = "TinyStoriesV2-GPT4-valid.txt") -> None:
    import os
    import subprocess

    subprocess.run(
        [
            "py-spy",
            "record",
            "--subprocesses",
            "--format",
            "speedscope",
            "-r",
            "10",
            "-o",
            "/data/pyspy-profile.speedscope.json",
            "--",
            "python",
            "cs336_basics/profile_run_train_bpe.py",
        ],
        cwd="/root",
        env={
            **os.environ,
            "PYTHONPATH": "/root",
            "PROFILE_INPUT": f"/data/{input_file}",
            "PROFILE_VOCAB": str(vocab_size),
        },
        check=True,
    )
    volume.commit()


@app.function(cpu=(64, 64), memory=(100 * 1024, 100 * 1024), volumes={"/data": volume}, timeout=12 * 3600)
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


@app.function(cpu=(64, 64), memory=(100 * 1024, 100 * 1024), volumes={"/data": volume}, timeout=12 * 3600)
def train_to_volume(vocab_size: int, input_file: str) -> str:
    """Train and persist results to the volume, so nothing depends on the local client staying connected.

    Pair with `modal run --detach` for long runs.
    """
    from cs336_basics.run_train_bpe import run_train_bpe

    start = time.monotonic()
    vocab, merges = run_train_bpe(
        input_path=f"/data/{input_file}",
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    elapsed = time.monotonic() - start

    stem = Path(input_file).stem
    with open(f"/data/{stem}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"/data/{stem}_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    volume.commit()

    longest = max(vocab.values(), key=len)
    summary = f"trained in {elapsed:.1f}s; vocab={len(vocab)}, merges={len(merges)}; longest token: {longest!r}"
    print(summary)
    return summary


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
