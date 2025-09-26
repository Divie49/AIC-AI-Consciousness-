# train.py
# pip install torch tokenizers

import os
import math
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tokenizers import Tokenizer
from model import GPTlikeModel, count_parameters

# Minimal dataset that reads a tokenized file where each line is space-separated token ids.
class IdDataset(Dataset):
    def __init__(self, tokenized_files, seq_len):
        # tokenized_files: list of .npy or plain text with IDs separated by spaces per line
        self.ids = []
        for f in tokenized_files:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    toks = list(map(int, line.split()))
                    self.ids.extend(toks)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.ids) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.ids[start:start + self.seq_len]
        if len(chunk) < self.seq_len:
            # pad with zeros (assumes 0 is <pad> or valid)
            chunk = chunk + [0] * (self.seq_len - len(chunk))
        return torch.tensor(chunk, dtype=torch.long)

def collate(batch):
    return torch.stack(batch, dim=0)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train_loop(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    epochs=1,
    grad_accum_steps=1,
    save_dir="ckpt",
    mixed_precision=True,
):
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    model.to(device)
    model.train()
    global_step = 0
    for epoch in range(epochs):
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            attention_mask = (input_ids != 0).long()  # simple mask assuming 0 is pad
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                logits = model(input_ids, attention_mask=attention_mask)
                shift_logits = logits.view(-1, logits.size(-1))
                shift_labels = labels.reshape(-1)
                loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=0)
                loss = loss / grad_accum_steps
            if mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                if mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % 100 == 0:
                    print(f"Epoch {epoch+1} step {global_step} loss {loss.item()*grad_accum_steps:.4f} lr {scheduler.get_last_lr()[0]:.6e}")
        print(f"Epoch {epoch+1} done in {time.time()-t0:.1f}s")

        # save checkpoint
        os.makedirs(save_dir, exist_ok=True)
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
        }
        torch.save(ckpt, os.path.join(save_dir, f"ckpt_epoch{epoch+1}.pt"))
        print("Saved checkpoint")

if __name__ == "__main__":
    # Example config (close to ~500M parameters like previous)
    cfg = {
        "vocab_size": 30000,
        "d_model": 2048,
        "n_layers": 9,
        "n_heads": 16,
        "d_ff": 8192,
        "max_seq_len": 1024,
        "dropout": 0.0,
        "tie_word_embeddings": True,
        "use_checkpoint": True
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTlikeModel(**cfg)
    print("Params (M)", count_parameters(model)/1e6)

    # Prepare dataset: for demonstration create a small dataset with token ids stored in text files
    train_files = ["train_ids.txt"]  # each line: space separated token ids (e.g. created from tokenizer)
    seq_len = 128 + 1  # +1 because we'll use input_ids=seq[:-1], labels=seq[1:]
    dataset = IdDataset(train_files, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)

    # optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    total_steps = len(loader) * 3  # epochs=3 in this example
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    train_loop(model, loader, optimizer, scheduler, device, epochs=3, grad_accum_steps=4, save_dir="ckpts", mixed_precision=True)
