
import math, time
from dataclasses import dataclass
from typing import List
import torch, torch.nn as nn, torch.nn.functional as F

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    vocab_size: int = 256
    block_size: int = 256
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    batch_size: int = 32
    max_steps: int = 1500
    lr: float = 3e-4
    sample_every: int = 300
    ckpt_path: str = "ckpt_nodejs_solid.pt"

cfg = Config()
torch.manual_seed(cfg.seed)

EXAMPLES: List[str] = [
    """Principle: Single Responsibility
Split IO, domain, and presentation.

Example (ESM):
export class UserRepo {
  async get(id) { /* db */ }
  async save(user) { /* db */ }
}
export class EmailSvc {
  async sendWelcome(user) { /* send */ }
}
""",
    """Principle: Open/Closed
Register strategies without altering callers.

Example:
export class TaxRule { apply(amount) { throw new Error("override"); } }
export class FlatTax extends TaxRule { apply(a) { return a*0.1; } }
export function total(taxes, amount) {
  return taxes.reduce((acc,t) => t.apply(acc), amount);
}
""",
    """Principle: Liskov Substitution
Async contracts must be consistent.

Example:
export class Queue {
  async push(item) {}
  async pop() {}
}
export class FIFOQueue extends Queue { /* ... */ }
""",
    """Principle: Interface Segregation
Pass only the collaborators needed.

Example:
function process(repo, { logger }) {
  logger.info("start");
}
""",
    """Principle: Dependency Inversion
Depend on abstract protocols; wire concretes at edges.

Example:
export class Mailer { async send(to, body) {} }
export class SMTPMailer extends Mailer { async send(to, body) { /* ... */ } }
export class Signup {
  constructor(mailer) { this.mailer = mailer; }
  async run(email) { await this.mailer.send(email, "welcome"); }
}
""",
    """Anti-pattern: God module
module.exports = { run, saveUser, sendEmail }
Fix: split services and repositories."""
]

def encode(s: str): return torch.tensor(list(s.encode("utf-8")), dtype=torch.long)
def decode(t: torch.Tensor): return bytes(t.tolist()).decode("utf-8", errors="ignore")
X = torch.tensor(list("\n\n".join(EXAMPLES).encode("utf-8")), dtype=torch.long)

def get_batch(b,T):
    ix=torch.randint(len(X)-T,(b,))
    x=torch.stack([X[i:i+T] for i in ix]).to(cfg.device)
    y=torch.stack([X[i+1:i+T+1] for i in ix]).to(cfg.device)
    return x,y

class SelfAttention(nn.Module):
    def __init__(self,C,H,p): super().__init__(); self.H=H; self.k=nn.Linear(C,C,False); self.q=nn.Linear(C,C,False); self.v=nn.Linear(C,C,False); self.proj=nn.Linear(C,C); self.drop=nn.Dropout(p)
    def forward(self,x):
        B,T,C=x.size()
        k=self.k(x).view(B,T,self.H,C//self.H).transpose(1,2)
        q=self.q(x).view(B,T,self.H,C//self.H).transpose(1,2)
        v=self.v(x).view(B,T,self.H,C//self.H).transpose(1,2)
        att=(q@k.transpose(-2,-1))/math.sqrt(k.size(-1))
        att=att.masked_fill(torch.triu(torch.ones(T,T,device=x.device),1).bool(),float("-inf"))
        att=F.softmax(att,dim=-1)
        y=att@v
        y=y.transpose(1,2).contiguous().view(B,T,C)
        return self.drop(self.proj(y))

class Block(nn.Module):
    def __init__(self,C,H,p): super().__init__(); self.ln1=nn.LayerNorm(C); self.attn=SelfAttention(C,H,p); self.ln2=nn.LayerNorm(C); self.ff=nn.Sequential(nn.Linear(C,4*C),nn.GELU(),nn.Linear(4*C,C),nn.Dropout(p))
    def forward(self,x): x=x+self.attn(self.ln1(x)); x=x+self.ff(self.ln2(x)); return x

class TinyGPT(nn.Module):
    def __init__(self,cfg): super().__init__(); self.tok=nn.Embedding(cfg.vocab_size,cfg.n_embd); self.pos=nn.Embedding(cfg.block_size,cfg.n_embd); self.blocks=nn.ModuleList([Block(cfg.n_embd,cfg.n_head,cfg.dropout) for _ in range(cfg.n_layer)]); self.ln=nn.LayerNorm(cfg.n_embd); self.head=nn.Linear(cfg.n_embd,cfg.vocab_size)
    def forward(self,idx):
        B,T=idx.size(); x=self.tok(idx)+self.pos(torch.arange(T,device=idx.device))
        for b in self.blocks: x=b(x)
        return self.head(self.ln(x))

def train():
    model=TinyGPT(cfg).to(cfg.device); opt=torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for step in range(1,cfg.max_steps+1):
        x,y=get_batch(cfg.batch_size,cfg.block_size)
        logits=model(x); loss=F.cross_entropy(logits.view(-1,cfg.vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step%100==0: print(f"step={step} loss={loss.item():.4f}")
        if step%cfg.sample_every==0: print(sample(model,"Principle: ",400))
    torch.save(model.state_dict(), cfg.ckpt_path)

@torch.no_grad()
def sample(model,prompt,length):
    idx=encode(prompt).unsqueeze(0).to(cfg.device); model.eval()
    for _ in range(length):
        logits=model(idx[:,-cfg.block_size:]); probs=F.softmax(logits[:,-1,:],dim=-1); nxt=torch.multinomial(probs,1); idx=torch.cat([idx,nxt],1)
    return decode(idx[0].cpu())

if __name__=="__main__":
    s=time.time(); train(); print(f"done in {time.time()-s:.1f}s -> {cfg.ckpt_path}")
