from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import tiktoken

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

@dataclass
class GPTConfig:
    context_length: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embedding_dimensions: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embedding_dimensions % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embedding_dimensions, 3 * config.n_embedding_dimensions)
        self.c_proj = nn.Linear(config.n_embedding_dimensions, config.n_embedding_dimensions)
        self.c_proj.BPT_SCALE_INIT = 1 # type: ignore
        self.n_head = config.n_head
        self.n_embedding_dimensions = config.n_embedding_dimensions
        
        self.register_buffer("bias", torch.tril(torch.ones(config.context_length, config.context_length))
                             .view(1,1, config.context_length, config.context_length))
        
    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        
        qkv: Tensor = self.c_attn(x)
        
        q, k, v = qkv.split(self.n_embedding_dimensions, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        
        att: Tensor = (q @ k.transpose(-2,-1)) * (1.0 /math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # type: ignore
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.c_proj(y) 
        
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embedding_dimensions, 4 * config.n_embedding_dimensions)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embedding_dimensions, config.n_embedding_dimensions)
        self.c_proj.BPT_SCALE_INIT = 1 # type: ignore


    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)
        
class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embedding_dimensions)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embedding_dimensions)
        self.mlp = MLP(config)
        
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x)) 
        
    
class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embedding_dimensions),
            wpe = nn.Embedding(config.context_length, config.n_embedding_dimensions),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embedding_dimensions)
        ))
        
        self.lm_head = nn.Linear(config.n_embedding_dimensions, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # type: ignore
    def _init_weights(self, module: Tensor):
        std = 0.02
        if hasattr(module, 'BPT_SCALE_INIT'):
            std*= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.weight)
        elif isinstance(model, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, indices: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        B, T = indices.size()
        
        assert T <= self.config.context_length, f"Cannot forward sequence of length {T}, context length: {self.config.context_length}"
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (T)
        positional_embeddings = self.transformer.wpe(pos) # type:ignore # position embeddings of shape (T, n_embedding_dimensions) 
        token_embeddings = self.transformer.wte(indices) # type: ignore # token embeddings of shape (B, T, n_embedding_dimensions)
        x = token_embeddings + positional_embeddings
        
        for block in self.transformer.h: # type: ignore
            x = block(x)
        x = self.transformer.ln_f(x) # type: ignore
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None 
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt-xl'}
        print(f'loading weights from pretrained gpt: {model_type}')

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embedding_dimensions=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embedding_dimensions=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embedding_dimensions=1280),
            'gpt2-xl': dict(n_layer=28, n_head=25, n_embedding_dimensions=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['context_length'] = 1024
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # type: ignore

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] 
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] 
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert(sd_keys_hf == sd_keys), f'mismatches keys: {len(sd_keys_hf) != len(sd_keys)}' 
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert (sd_hf[k].shape[::-1] == sd[k].shape), f"{k} {sd_hf[k].shape[::-1]} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    
        return model

class DataLoaderLite:
    def __init__(self, B: int, T: int) -> None:
        self.B = B
        self.T = T
        with open('../gpt/input.txt') as f:
            text = f.read()
            
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens).to(device)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch =  {len(self.tokens) // (B*T)} batches')
        self.current_position = 0

    def next_batch(self) -> tuple[Tensor, Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        
        x = (buf[:-1].view(B,T))
        y = (buf[1:].view(B,T))
        self.current_position+=B*T
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
            
        return x, y

        
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.mps.manual_seed(1337)

train_loader = DataLoaderLite(4, 32)
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.eval()
model.to(device)
# logits, loss= model(x, y)
# print(f'{logits.shape=}, {loss=}')
n_iterations = 50
learning_rate = 3e-4
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for i in range(n_iterations):
    x, y = train_loader.next_batch()
    optimiser.zero_grad()
    logits, loss= model(x, y)
    loss.backward()
    optimiser.step()
    print(f"step {i} loss: {loss.item()}")

max_return_sequences = 2
max_length =30
enc = tiktoken.get_encoding('gpt2')
t = enc.encode("Hello, I'm a language model,")
tokens: Tensor = torch.tensor(t, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(max_return_sequences, 1) # (5,8)
x = tokens.to(device)

while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x) # (B,T,vocab_size)
        logits = logits[:, -1, :] # take last token (B, vocab_size)
        probs = F.softmax(logits, dim=-1) #  (B, vocab_size)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, vocab_size)
        idx = torch.multinomial(topk_probs, 1)
        x_col = torch.gather(topk_indices, -1, idx)
        x = torch.cat((x, x_col), dim=1) 

for response in x.tolist():
    print(enc.decode(response))
