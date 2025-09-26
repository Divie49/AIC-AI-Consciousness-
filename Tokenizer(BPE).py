from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Initialize a BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# Pre-tokenizer (splits into words/chars before BPE merges)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Trainer for BPE
trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"])

# Train on your dataset
files = ["dataset.txt"]  # put your training text here
tokenizer.train(files, trainer)

# Optional decoder
tokenizer.decoder = decoders.BPEDecoder()

# Save tokenizer
tokenizer.save("bpe_tokenizer.json")

# Load later
loaded_tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

# Example encode/decode
output = tokenizer.encode("Hello world!")
print("Token IDs:", output.ids)
print("Tokens:", output.tokens)
print("Decoded:", tokenizer.decode(output.ids))
