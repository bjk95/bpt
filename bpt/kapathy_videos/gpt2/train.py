import numpy as np
from dataclasses import dataclass
import inspect
import math
import os
import time
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import tiktoken
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
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
         
        # att: Tensor = (q @ k.transpose(-2,-1)) * (1.0 /math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # type: ignore
        # att = F.softmax(att, dim=-1)
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        # y = att @ v
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
        config_args['vocab_size'] = 50304
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
    def configure_optimizers(self, weight_decay: float, learning_rate: float, device=str):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # if master_process:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        # if master_process:
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


def load_tokens(filename:str):
    npt = np.load(filename)
    return torch.tensor(npt, dtype=torch.long)

class DataLoaderLite:
    def __init__(self, B: int, T: int, process_rank: int, num_processes: int, split: str) -> None:
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if is_master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

if __name__ == '__main__':     
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        print('using ddp')
        assert torch.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        is_master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        is_master_process = True
        
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.mps.manual_seed(1337)
    total_batch_size = 524288
    B = 8
    T = 1024
    assert total_batch_size % (B*T * ddp_world_size) ==0
    grad_accumulation_steps = total_batch_size // (B*T * ddp_world_size)
    print(f'grad_accumulation_steps: {grad_accumulation_steps}')
    print(f'total_batch_size: {total_batch_size}')
    torch.set_float32_matmul_precision('medium')
    train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.eval()
    model.to(device)
    model = torch.compile(model) # type: ignore
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    # logits, loss= m del(x, y)
    # print(f'{logits.shape=}, {loss=}')
    max_lr = 6e-4
    min_lr = max_lr*.1
    warmup_steps = 10
    max_steps = 250
    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return max_lr * (step+1) / warmup_steps
        elif step > max_steps:
            return min_lr
        else:
            decay_ratio = (step-warmup_steps)/ (max_steps-warmup_steps) 
            assert 0 <= decay_ratio <=1
            coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            return max_lr + coeff * (max_lr - min_lr)
    learning_rate = 3e-4
    optimiser = raw_model.configure_optimizers(weight_decay=.1, learning_rate=6e-4, device=device)

    for step in range(max_steps):

        t0= time.time()
        optimiser.zero_grad()
        loss_accum = 0
        for micro_step in range(grad_accumulation_steps):
            x, y = train_loader.next_batch()
            with torch.autocast(device):
                logits, loss= model(x, y)
            loss = loss / grad_accumulation_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accumulation_steps-1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr
        optimiser.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_second = (train_loader.B * train_loader.T * grad_accumulation_steps) / (t1-t0)
        if is_master_process:
            print(f'step {step}, loss: {loss_accum}, dt: {dt}ms, tokens/s = {tokens_per_second} {norm=}, {lr=}')

    max_return_sequences = 2
    max_length =30
    enc = tiktoken.get_encoding('gpt2')
    t = enc.encode("First Citizen: You are all resolved rather to die than to famish")
    tokens: Tensor = torch.tensor(t, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(max_return_sequences, 1) # (5,8)
    x = tokens.to(device )

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
