import os

from cs336_basics.run_train_bpe import run_train_bpe

if input_path := os.environ.get("PROFILE_INPUT"):
    vocab_size = int(os.environ.get("PROFILE_VOCAB", "10000"))
else:
    from tests.common import FIXTURES_PATH

    input_path, vocab_size = FIXTURES_PATH / "corpus.en", 500

run_train_bpe(input_path, vocab_size, ["<|endoftext|>"])
