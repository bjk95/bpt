import math
import os
import time
from bpt.kapathy_videos.gpt2.data_loader import DataLoaderLite
from bpt.kapathy_videos.gpt2.model import GPT, GPTConfig, RunConfig
import torch
from torch import Tensor
from torch.nn import functional as F
import tiktoken
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from bpt.kapathy_videos.gpt2.monitoring import init_wandb


def get_lr(config: RunConfig, step: int) -> float:
    if step < config.warmup_steps:
        return config.max_learning_rate * (step+1) / config.warmup_steps
    elif step > config.max_steps:
        return config.minimum_learning_rate
    else:
        decay_ratio = (step-config.warmup_steps)/ (config.max_steps-config.warmup_steps) 
        assert 0 <= decay_ratio <=1
        coefficient = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return config.max_learning_rate + coefficient * (config.max_learning_rate - config.minimum_learning_rate)

def sample(model: GPT, prefix: str):
    max_return_sequences = 2
    max_length =30
    enc = tiktoken.get_encoding('gpt2')
    t = enc.encode(prefix)
    tokens: Tensor = torch.tensor(t, dtype=torch.long, device=device)
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
    run_config = RunConfig()
    model_config = GPTConfig()
    wandb = init_wandb(model_config, run_config)
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
    set_seed(run_config.seed) 
    
    B = run_config.batch_size
    T = model_config.context_length
    
    assert run_config.total_batch_size % (B*T * ddp_world_size) ==0
    grad_accumulation_steps = run_config.total_batch_size // (B*T * ddp_world_size)
    print(f'grad_accumulation_steps: {grad_accumulation_steps}')
    print(f'total_batch_size: {run_config.total_batch_size}')

    torch.set_float32_matmul_precision('medium')
    train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, is_master_process=is_master_process, split='train')
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, is_master_process=is_master_process,split="val")

    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.eval()
    model.to(device)
    model = torch.compile(model) # type: ignore
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank]) # type: ignore
    raw_model = model.module if ddp else model # type: ignore
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    optimiser = raw_model.configure_optimizers(weight_decay=.1, learning_rate=6e-4, device=device) # type: ignore
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass
    for step in range(run_config.max_steps):
        t0= time.time()
        model.train()
        optimiser.zero_grad()
        last_step = (step == run_config.max_steps - 1)

        loss_accumulator = torch.tensor(0.0, device=device)
        for micro_step in range(grad_accumulation_steps):
            x, y = train_loader.next_batch()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss= model(x, y)
            loss = loss / grad_accumulation_steps
            loss_accumulator += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accumulation_steps-1) # type: ignore
            loss.backward()
        # once in a while evaluate our validation loss
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accumulator = torch.tensor(0.0, device=device)
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accumulator += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accumulator, op=dist.ReduceOp.AVG)
            if is_master_process:
                wandb.log({
                    "val/loss": val_loss_accumulator.item()
                }, step=step)
                print(f"validation loss: {val_loss_accumulator.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accumulator.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accumulator.item()
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    torch.save(checkpoint, checkpoint_path)
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(run_config, step)
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr
        optimiser.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_second = (train_loader.B * train_loader.T * grad_accumulation_steps) / (t1-t0)
        if ddp:
            dist.all_reduce(loss_accumulator, op=dist.ReduceOp.AVG)
        if is_master_process:
            wandb.log({
                "train/loss": loss_accumulator.item(), # Log the averaged loss
                "train/learning_rate": lr,
                "train/grad_norm": norm.item(),
                "perf/step_time_ms": dt,
                "perf/tokens_per_sec": tokens_per_second,
                # Optional: Add epoch if you track it
                # "train/epoch": current_epoch
            }, step=step)
            print(f'step {step}, loss: {loss_accumulator}, dt: {dt}ms, tokens/s = {tokens_per_second} {norm=}, {lr=}')
    
    sample(model, "I'm a language model and")
    wandb.finish() 
    if ddp: 
        destroy_process_group()

# run using:  torchrun --standalone --nproc_per_node=1 -m bpt.kapathy_videos.gpt2.train
