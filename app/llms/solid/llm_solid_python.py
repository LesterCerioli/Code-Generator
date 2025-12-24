
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    vocab_size: int = 256  # byte-level
    block_size: int = 256
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    batch_size: int = 32
    max_steps: int = 1500
    lr: float = 3e-4
    sample_every: int = 300
    ckpt_path: str = "ckpt_python_solid.pt"

cfg = Config()
torch.manual_seed(cfg.seed)

# -------------------------
# Curriculum data (Python)
# -------------------------
EXAMPLES: List[str] = [
    # S — Single Responsibility
    """Principle: Single Responsibility
In Python, prefer small modules/classes with one reason to change.

Example:
from dataclasses import dataclass

@dataclass
class User:
    id: int
    email: str

class UserRepository:
    def get(self, user_id: int) -> User: ...
    def save(self, user: User) -> None: ...

class UserEmailService:
    def send_welcome(self, user: User) -> None: ...
""",

    # O — Open/Closed
    """Principle: Open/Closed
Use ABCs and registration to extend behavior without modifying core.

Example:
from abc import ABC, abstractmethod

class TaxRule(ABC):
    @abstractmethod
    def apply(self, amount: float) -> float: ...

class FlatTax(TaxRule):
    def apply(self, amount: float) -> float:
        return amount * 0.1

def total(taxes: list[TaxRule], amount: float) -> float:
    for t in taxes:
        amount = t.apply(amount)
    return amount
""",

    # L — Liskov Substitution
    """Principle: Liskov Substitution
Respect contracts: subclasses must not weaken base behavior/typing.

Example:
from abc import ABC, abstractmethod

class Queue(ABC):
    @abstractmethod
    def push(self, item: str) -> None: ...
    @abstractmethod
    def pop(self) -> str: ...

class FIFOQueue(Queue):
    def __init__(self): self._q: list[str] = []
    def push(self, item: str) -> None: self._q.append(item)
    def pop(self) -> str: return self._q.pop(0)
""",

    # I — Interface Segregation
    """Principle: Interface Segregation
Expose narrow protocols instead of god interfaces.

Example:
class Logger:
    def info(self, msg: str) -> None: ...
    def warn(self, msg: str) -> None: ...

class Metrics:
    def incr(self, key: str) -> None: ...

# Accept what you use:
def process(repo, *, logger: Logger) -> None:
    logger.info("starting")
""",

    # D — Dependency Inversion
    """Principle: Dependency Inversion
Depend on abstractions (protocols/ABCs), inject implementations.

Example:
from abc import ABC, abstractmethod

class Mailer(ABC):
    @abstractmethod
    def send(self, to: str, body: str) -> None: ...

class SMTPMailer(Mailer):
    def send(self, to: str, body: str) -> None: ...

class Signup:
    def __init__(self, mailer: Mailer): self.mailer = mailer
    def run(self, email: str) -> None:
        self.mailer.send(email, "welcome")
""",
]

# Optional: add anti-patterns and fixes
EXAMPLES += [
    """Anti-pattern: God class
class App:
    def run(self): ...
    def save_user(self): ...
    def send_email(self): ...
Fix: split responsibilities into services and repositories.""",
]

# -------------------------
# Tokenization (byte-level)
# -------------------------
def encode(s: str) -> torch.Tensor:
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long)

def decode(t: torch.Tensor) -> str:
    return bytes(t.tolist()).decode("utf-8", errors="ignore")

data = ("\n\n".join(EXAMPLES)).encode("utf-8")
X = torch.tensor(list(data), dtype=torch.long)

# -------------------------
# Dataset
# -------------------------
def get_batch(batch_size: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(X) - block_size, (batch_size,))
    x = torch.stack([X[i:i+block_size] for i in ix])
    y = torch.stack([X[i+1:i+block_size+1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)

# -------------------------
# Model: Tiny GPT-like
# -------------------------
class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float("-inf"))
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

    def forward(self, x):
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

    def forward(self, idx):
        B, T = idx.size()
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# -------------------------
# Training
# -------------------------
def train():
    model = TinyGPT(cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for step in range(1, cfg.max_steps + 1):
        x, y = get_batch(cfg.batch_size, cfg.block_size)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.4f}")

        if step % cfg.sample_every == 0:
            print(sample(model, prompt="Principle: ", length=400))
    torch.save(model.state_dict(), cfg.ckpt_path)

# -------------------------
# Sampling
# -------------------------
@torch.no_grad()
def sample(model: TinyGPT, prompt: str, length: int = 400) -> str:
    idx = encode(prompt).unsqueeze(0).to(cfg.device)
    model.eval()
    for _ in range(length):
        idx_cond = idx[:, -cfg.block_size:]
        logits = model(idx_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return decode(idx[0].cpu())

if __name__ == "__main__":
    start = time.time()
    train()
    print(f"done in {time.time() - start:.1f}s, saved -> {cfg.ckpt_path}")
