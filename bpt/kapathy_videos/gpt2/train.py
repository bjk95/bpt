import math
import os
import time
from data_loader import DataLoaderLite
from model import GPT, GPTConfig
import torch
from torch import Tensor
from torch.nn import functional as F
import tiktoken
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def get_lr(step: int, max_steps: int) -> float:
    max_lr = 6e-4
    min_lr = max_lr*.1
    warmup_steps = 715

    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    elif step > max_steps:
        return min_lr
    else:
        decay_ratio = (step-warmup_steps)/ (max_steps-warmup_steps) 
        assert 0 <= decay_ratio <=1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return max_lr + coeff * (max_lr - min_lr)

def sample(model: GPT, prefix: str):
    max_return_sequences = 2
    max_length =30
    enc = tiktoken.get_encoding('gpt2')
    t = enc.encode(prefix)
    tokens: Tensor = torch.tensor(t, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(max_return_sequences, 1) # (5,8)
    x = tokens.to(device)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x) # (B,T,vocab_size)
            logits = logits[:, -1, :] # take last token (B, vocab_size)
            probs = F.softmax(logits, dim=-1) #  (B, vocab_size)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, vocab_size)
            idx = torch.multinomial(topk_probs, 1)
            x_col = torch.gather(topk_indices, -1, idx)
            x = torch.cat((x, x_col), dim=1) 

    for response in x.tolist():
        print(enc.decode(response))

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
 
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
    set_seed(1337) 
    
    total_batch_size = 524288
    B = 8
    T = 1024
    
    assert total_batch_size % (B*T * ddp_world_size) ==0
    grad_accumulation_steps = total_batch_size // (B*T * ddp_world_size)
    print(f'grad_accumulation_steps: {grad_accumulation_steps}')
    print(f'total_batch_size: {total_batch_size}')

    torch.set_float32_matmul_precision('medium')
    train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, is_master_process=is_master_process, split='train')
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, is_master_process=is_master_process,split="val")

    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.eval()
    model.to(device)
    model = torch.compile(model) # type: ignore
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    max_steps = 19073
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    optimiser = raw_model.configure_optimizers(weight_decay=.1, learning_rate=6e-4, device=device)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass
    for step in range(max_steps):
        t0= time.time()
        optimiser.zero_grad()
        last_step = (step == max_steps - 1)

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
        # once in a while evaluate our validation loss
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if is_master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    torch.save(checkpoint, checkpoint_path)
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, max_steps)
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr
        optimiser.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_second = (train_loader.B * train_loader.T * grad_accumulation_steps) / (t1-t0)
        if is_master_process:
            print(f'step {step}, loss: {loss_accum}, dt: {dt}ms, tokens/s = {tokens_per_second} {norm=}, {lr=}')
    
    sample(model, "I'm a language model and")
    
    if ddp: 
        destroy_process_group()
