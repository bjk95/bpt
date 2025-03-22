from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from time import time
from torch.amp import autocast, GradScaler

# T4-optimized hyperparameters
use_mixed_precision = True  # Set to False to disable mixed precision training
batch_size = 256  # Use standard batch size without accumulation
context_size = 256
max_iterations = 192000 // batch_size
learning_rate = 1e-3
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f'{device=}')
evaluation_iterations = 50
n_embedding_dimensions = 384
n_blocks = 6
n_heads = 6
dropout = .2

print(f'Hyperparameters: {batch_size=} {context_size=} {max_iterations=} {learning_rate=} {use_mixed_precision=}')

if device == 'cuda':
    torch.set_default_device('cuda')
    # Set memory allocation to be more efficient on T4
    torch.backends.cudnn.benchmark = True  # Optimize CUDNN for fixed input sizes



torch.manual_seed(1337)

class Head(nn.Module):
    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embedding_dimensions, head_size, bias=True)
        self.query = nn.Linear(n_embedding_dimensions, head_size, bias=True)
        self.value = nn.Linear(n_embedding_dimensions, head_size, bias=True)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x: torch.Tensor):
        _,T,C = x.shape
        k: Tensor = self.key(x)
        q: Tensor = self.query(x)
        affinity_weights: Tensor = q @ k.transpose(-2,-1) * C**-.5
        affinity_weights = affinity_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # type: ignore
        affinity_weights = affinity_weights.softmax(dim=-1) 
        affinity_weights = self.dropout(affinity_weights)
        v = self.value(x)
        out = affinity_weights @ v
        return out
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embedding_dimensions, n_embedding_dimensions)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.projection(out))

class FeedForward(nn.Module):
    def __init__(self, n_embed: int) -> None:
        super().__init__()   
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: Tensor):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed: int, n_heads: int) -> None:
        super().__init__()
        head_size = n_embed // n_heads
        self.self_attention = MultiHeadAttention(n_heads, head_size)
        self.feed_forward = FeedForward(n_embed)
        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)
        
    def forward(self, x: Tensor):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        
        return x
    
    
class NanoGPT(nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embedding_dimensions)
        self.position_embedding_table = nn.Embedding(context_size, n_embedding_dimensions)
        self.blocks = nn.Sequential(*[Block(n_embedding_dimensions, n_heads=4) for _ in range(n_blocks)])
        self.final_layer_norm = nn.LayerNorm(n_embedding_dimensions)
        self.language_modelling_head = nn.Linear(n_embedding_dimensions, vocabulary_size)
        
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # (B, T, embedding_dimensions)
        positional_embedding = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x= token_embeddings + positional_embedding
        x = self.blocks(x)
        x = self.final_layer_norm(x)  # Apply final layer norm before language head
        logits = self.language_modelling_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            batches, time_steps, channels = logits.shape
            logits = logits.view(batches * time_steps, channels)
            targets = targets.view(batches* time_steps)
            loss = F.cross_entropy(logits,targets)
        return logits, loss

    
    def generate(self, indices: torch.Tensor, max_new_tokens: int):
        # Set to evaluation mode for generation
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Take only the last context_size tokens
                idx_cond = indices[:, -context_size:]
                
                if use_mixed_precision:
                    # Use inference mode with autocast if mixed precision is enabled
                    with autocast(device_type=device):
                        logits, _ = self(idx_cond)
                        # extract last time step
                        logits = logits[:, -1, :]
                        probs = F.softmax(logits, dim=1)
                        next_index = torch.multinomial(probs, num_samples=1)
                else:
                    # Standard precision inference
                    logits, _ = self(idx_cond)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=1)
                    next_index = torch.multinomial(probs, num_samples=1)
                
                indices = torch.cat((indices, next_index), dim=1)
                
        # Remember to set back to training mode if continuing training
        self.train()
        return indices
      
def load_data() -> tuple[Tensor, Tensor, int, dict[str, int], dict[int, str], Callable[[list[int]], str], list[str]]:
    with open('input.txt', 'r', encoding='utf8') as f:
        text = f.read()

    characters = sorted(list(set(text)))
    vocabulary_size = len(characters)  

    stoi = {ch: i for i, ch in enumerate(characters)}
    itos = {i: ch for ch, i in stoi.items()}
    
    encode: Callable[[str], list[int]] = lambda s: [stoi[c] for c in s]
    decode: Callable[[list[int]], str] = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n_split = int(len(data) *.9)
    training_data = data[:n_split]
    validation_data = data[n_split:]
    
    return training_data, validation_data, vocabulary_size, stoi, itos, decode, characters

def get_batch(split:str | None, training_data: torch.Tensor, validation_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    data = training_data if split == 'train' or None else validation_data
    idx = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in idx]).to(device)
    y = torch.stack([data[i+1:i+context_size+1] for i in idx]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model: nn.Module, training_data: torch.Tensor, validation_data: torch.Tensor) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evaluation_iterations, device=device)
        for k in range(evaluation_iterations):
            X, Y = get_batch(split, training_data, validation_data)
            if use_mixed_precision:
                with autocast(device_type=device):  # Use mixed precision for evaluation too
                    _, loss = model(X,Y)
            else:
                _, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def train(model, get_batch, decode, characters, stoi, itos, max_iterations, training_data, validation_data):
    print(f'{max_iterations=}')
    start = time()
    block_start_time = time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler() if use_mixed_precision else None  # Only create scaler if using mixed precision

    for steps in range(max_iterations):
        X_batch, Y_batch = get_batch('train', training_data, validation_data)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_mixed_precision:
            # Mixed precision forward pass
            with autocast(device_type=device):
                _, loss = model(X_batch, Y_batch)
            
            # Scale gradients and perform backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            _, loss = model(X_batch, Y_batch)
            loss.backward()
            optimizer.step()
        
        if steps % 10 == 0:
            print(f'{steps=}')
        
        if steps % 100 == 0:
            # if False:
            print('Estimating loss')
            current_loss = estimate_loss(model, training_data, validation_data)
            print(f'{steps=} {current_loss=}')
            print(f'{time() - block_start_time:.1f} seconds')
            block_start_time = time()
            
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': steps,
                'loss': current_loss,
                'characters': characters,
                'stoi': stoi,
                'itos': itos
            }
            
            # Save scaler state only if using mixed precision
            if use_mixed_precision:
                checkpoint['scaler'] = scaler.state_dict()
                
            torch.save(checkpoint, f'saved_models/checkpoint_step_{steps}.pth')

    context = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=400)[0].tolist()))
    print(f'Took {time() - start:.1f} seconds')
    torch.save(model.state_dict(), 'saved_models/gpt_model.pth')
    print("Model saved to gpt_model.pth")


def run():
    training_data, validation_data, vocabulary_size, stoi, itos, decode, characters = load_data()
    
    model = NanoGPT(vocabulary_size)
    
    # Calculate and print model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters")
    
    m = model.to(device)   
    train(m, get_batch, decode, characters, stoi, itos, max_iterations, training_data, validation_data)
    
if __name__ == '__main__':
    run()
    
