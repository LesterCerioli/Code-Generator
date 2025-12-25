
import argparse
import math
import os
import random
import time
import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("llm_solid_go")

# ---------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 2025
    vocab_size: int = 256             # byte-level vocab
    block_size: int = 512
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 8
    dropout: float = 0.15
    batch_size: int = 48
    max_steps: int = 4000
    lr: float = 3e-4
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    sample_every: int = 250
    eval_every: int = 250
    early_patience: int = 8
    min_delta: float = 0.0001
    ckpt_dir: str = "ckpts_go"
    ckpt_name: str = "go_solid"
    resume: bool = True
    amp: bool = True
    temperature: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    sample_len: int = 700
    export_ts: bool = True

cfg = Config()

# ---------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------
# SOLID curriculum (Go)
# ---------------------------------------------------------------------
EXAMPLES: List[str] = [
    # S — Single Responsibility
    """Principle: Single Responsibility
Keep packages/types cohesive; one reason to change.

Example (user package):
package user

type User struct { ID int; Email string }

type Repo interface {
  Get(id int) (User, error)
  Save(u User) error
}

type EmailSvc interface {
  SendWelcome(u User) error
}

// Orchestrators should not persist, render and email at once.
// Split responsibilities across packages (repo, mail, ui, orchestrator).""",

    # O — Open/Closed
    """Principle: Open/Closed
Extend behavior via new types implementing interfaces; avoid modifying stable callers.

Example (billing package):
package billing

type TaxRule interface {
  Apply(amount float64) float64
}

type FlatTax struct{}
func (FlatTax) Apply(a float64) float64 { return a * 0.10 }

type ProgressiveTax struct{}
func (ProgressiveTax) Apply(a float64) float64 {
  if a <= 1000 { return a * 0.08 }
  return a * 0.12
}

func Total(rules []TaxRule, amount float64) float64 {
  for _, r := range rules {
    amount = r.Apply(amount)
  }
  return amount
}

// Callers stay closed to change; adding new rules requires new types, not edits.""",

    # L — Liskov Substitution
    """Principle: Liskov Substitution
Implementations must preserve expected behavior and not narrow contracts.

Example (queue package):
package queue

type Queue interface {
  Push(item string)
  Pop() (string, bool)
}

type FIFOQueue struct { q []string }

func (f *FIFOQueue) Push(item string) { f.q = append(f.q, item) }

func (f *FIFOQueue) Pop() (string, bool) {
  if len(f.q) == 0 { return "", false }
  it := f.q[0]
  f.q = f.q[1:]
  return it, true
}

// Subtypes should not panic on empty or change sync expectations.""",

    # I — Interface Segregation
    """Principle: Interface Segregation
Prefer narrow interfaces tailored to call sites; avoid god interfaces.

Example (app package):
package app

type Logger interface { Info(msg string) }
type Metrics interface { Incr(key string) }

func Process(repo any, logger Logger) {
  logger.Info("start")
}""",

    # D — Dependency Inversion
    """Principle: Dependency Inversion
Depend on abstractions; wire concretes at composition root (main).

Example (signup package):
package signup

type Mailer interface { Send(to, body string) error }

type SMTPMailer struct{}
func (SMTPMailer) Send(to, body string) error { return nil }

type Service struct { Mailer Mailer }

func (s Service) Run(email string) error {
  return s.Mailer.Send(email, "welcome")
}

// In main:
// m := SMTPMailer{}
// svc := Service{Mailer: m}
// _ = svc.Run("user@example.com")""",

    # Anti-patterns + fixes
    """Anti-pattern: Big package doing everything
package app
type App struct {
  // run, save user, send email, render UI, queue tasks...
}
Fix:
Split into focused packages (user, mail, ui, queue, billing).
Inject collaborators via constructor functions and interfaces.""",
]

# ---------------------------------------------------------------------
# Tokenization (byte-level with normalization)
# ---------------------------------------------------------------------
def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def encode(s: str) -> torch.Tensor:
    s = normalize_text(s)
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long)

def decode(t: torch.Tensor) -> str:
    return bytes(t.tolist()).decode("utf-8", errors="ignore")

def build_dataset(examples: List[str], val_ratio: float = 0.12) -> Tuple[torch.Tensor, torch.Tensor]:
    all_bytes = list(("\n\n".join(map(normalize_text, examples))).encode("utf-8"))
    data = torch.tensor(all_bytes, dtype=torch.long)
    n_val = max(1, int(len(data) * val_ratio))
    return data[:-n_val], data[-n_val:]

TRAIN, VAL = build_dataset(EXAMPLES)

# ---------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------
def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError("Dataset too small for chosen block_size")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def eval_loss(model: nn.Module, data: torch.Tensor, cfg: Config, num_batches: int = 25) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(data, cfg.batch_size, cfg.block_size, cfg.device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, cfg.dropout) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

# ---------------------------------------------------------------------
# Sampling with controls
# ---------------------------------------------------------------------
@torch.no_grad()
def sample(
    model: TinyGPT,
    prompt: str,
    cfg: Config,
    length: int,
    temperature: float,
    top_k: int,
    repeat_penalty: float,
    eos_token: Optional[int] = None
) -> str:
    model.eval()
    idx = encode(prompt).unsqueeze(0).to(cfg.device)
    seen: Dict[int, int] = {}
    for _ in range(length):
        idx_cond = idx[:, -cfg.block_size:]
        logits = model(idx_cond)[:, -1, :] / max(temperature, 1e-6)

        if top_k is not None and top_k > 0:
            vals, inds = torch.topk(logits, k=min(top_k, logits.size(-1)))
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, inds, vals)
            logits = mask

        if repeat_penalty and repeat_penalty > 1.0:
            for tok, count in seen.items():
                if count > 0:
                    logits[0, tok] /= repeat_penalty

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        tok = int(next_id.item())
        seen[tok] = seen.get(tok, 0) + 1

        idx = torch.cat([idx, next_id], dim=1)
        if eos_token is not None and tok == eos_token:
            break
    return decode(idx[0].cpu())

def eval_principles(text: str) -> Dict[str, bool]:
    """Heuristic coverage: check cues for S, O, L, I, D in a sample."""
    cues = {
        "S": ["package", "Repo", "EmailSvc", "cohesive", "single responsibility"],
        "O": ["interface", "Apply(", "rules", "extend", "strategy"],
        "L": ["Queue", "FIFO", "Pop()", "Push(", "contract"],
        "I": ["Logger", "Metrics", "narrow", "god interface"],
        "D": ["Mailer", "Service", "main", "composition root", "inject"],
    }
    return {k: any(cue.lower() in text.lower() for cue in v) for k, v in cues.items()}

# ---------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------
def ckpt_path(step: int, cfg: Config) -> str:
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    return os.path.join(cfg.ckpt_dir, f"{cfg.ckpt_name}_step{step}.pt")

def latest_ckpt(cfg: Config) -> Optional[str]:
    if not os.path.exists(cfg.ckpt_dir):
        return None
    files = [f for f in os.listdir(cfg.ckpt_dir) if f.startswith(cfg.ckpt_name) and f.endswith(".pt")]
    if not files:
        return None
    files.sort(key=lambda x: int(x.split("step")[-1].split(".pt")[0]))
    return os.path.join(cfg.ckpt_dir, files[-1])

def save_checkpoint(model: nn.Module, opt: torch.optim.Optimizer, step: int, cfg: Config) -> None:
    path = ckpt_path(step, cfg)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": asdict(cfg)}, path)
    log.info(f"Checkpoint saved -> {path}")

def load_checkpoint(model: nn.Module, opt: torch.optim.Optimizer, cfg: Config) -> int:
    last = latest_ckpt(cfg)
    if last is None:
        return 0
    ckpt = torch.load(last, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    step = int(ckpt.get("step", 0))
    log.info(f"Resumed from {last} at step {step}")
    return step

# ---------------------------------------------------------------------
# Training loop with validation and early stopping
# ---------------------------------------------------------------------
def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = cfg.device
    model = TinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and cfg.amp))

    start_step = load_checkpoint(model, opt, cfg) if cfg.resume else 0
    best_val = float("inf")
    patience = 0

    log.info(f"Device={device}, AMP={device=='cuda' and cfg.amp}")
    log.info(f"Config: {asdict(cfg)}")
    log.info(f"Dataset sizes: train={len(TRAIN)} bytes, val={len(VAL)} bytes")

    for step in range(start_step + 1, cfg.max_steps + 1):
        x, y = get_batch(TRAIN, cfg.batch_size, cfg.block_size, device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda" and cfg.amp)):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.clip_grad:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % 100 == 0:
            log.info(f"step={step} loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.6f}")

        if step % cfg.sample_every == 0:
            s = sample(
                model=model,
                prompt="Principle: ",
                cfg=cfg,
                length=cfg.sample_len,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                repeat_penalty=cfg.repeat_penalty,
                eos_token=None
            )
            log.info("Sample:\n" + s)
            log.info(f"SOLID cues: {eval_principles(s)}")

        if step % cfg.eval_every == 0:
            val_loss = eval_loss(model, VAL, cfg, num_batches=25)
            ppl = math.exp(min(val_loss, 20))
            log.info(f"val_loss={val_loss:.4f} perplexity={ppl:.2f}")

            if best_val - val_loss > cfg.min_delta:
                best_val = val_loss
                patience = 0
                save_checkpoint(model, opt, step, cfg)
            else:
                patience += 1
                log.info(f"No improvement (patience={patience}/{cfg.early_patience})")
                if patience >= cfg.early_patience:
                    log.info("Early stopping triggered.")
                    break

    # Final TorchScript export
    if cfg.export_ts:
        model.eval()
        example = torch.randint(0, cfg.vocab_size, (1, cfg.block_size)).to(device)
        scripted = torch.jit.trace(model, example)
        ts_path = os.path.join(cfg.ckpt_dir, f"{cfg.ckpt_name}.ts.pt")
        scripted.save(ts_path)
        log.info(f"TorchScript exported -> {ts_path}")

# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="LLM training script for SOLID (Golang curriculum, Python code)")
    for field in asdict(cfg).keys():
        val = getattr(cfg, field)
        if isinstance(val, bool):
            p.add_argument(f"--{field}", type=lambda s: s.lower() == "true", default=val)
        elif isinstance(val, int):
            p.add_argument(f"--{field}", type=int, default=val)
        elif isinstance(val, float):
            p.add_argument(f"--{field}", type=float, default=val)
        else:
            p.add_argument(f"--{field}", type=str, default=val)
    args = p.parse_args()
    return Config(**{**asdict(cfg), **vars(args)})

if __name__ == "__main__":
    cfg = parse_args()
    t0 = time.time()
    try:
        train(cfg)
    finally:
        log.info(f"Done in {time.time() - t0:.1f}s")
