from cs336_basics.run_train_bpe import run_train_bpe
from tests.common import FIXTURES_PATH

run_train_bpe(FIXTURES_PATH / "corpus.en", 500, ["<|endoftext|>"])
