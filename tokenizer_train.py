# tokenizer_train.py
# pip install tokenizers

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing
import argparse
import os

def train_bpe(files, vocab_size=30000, min_frequency=2, special_tokens=None, save_path="bpe-tokenizer.json"):
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    tokenizer.train(files, trainer)
    # Post processing to add BOS/EOS tokens when encoding if desired
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )
    tokenizer.save(save_path)
    print(f"Saved tokenizer to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="Text files to train tokenizer on")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--save_path", type=str, default="bpe-tokenizer.json")
    args = parser.parse_args()
    train_bpe(args.files, vocab_size=args.vocab_size, save_path=args.save_path)
