import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 32
context_length = 8
max_iterations = 30000
evaluation_rate = 300
learning_rate = 1e-2
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f'{device=}')
evaluation_iterations = 200

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf8') as f:
    text = f.read()

characters = sorted(list(set(text)))
vocabulary_size = len(characters)  

stoi = {ch: i for i, ch in enumerate(characters)}
itos = {i: ch for ch, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n_split = int(len(data) *.9)
training_data = data[:n_split]
validation_data = data[n_split:]

def get_batch(split:str = 'train') -> tuple[torch.Tensor, torch.Tensor]:
    data = training_data if split == 'train' else validation_data
    idx = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in idx]).to(device)
    y = torch.stack([data[i+1:i+context_length+1] for i in idx]).to(device)
    return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.token_embedding_table(idx)
        
        if targets is None:
            loss = None
        else:
            batches, time_steps, channels = logits.shape
            logits = logits.view(batches * time_steps, channels)
            targets = targets.view(batches* time_steps)
            loss = F.cross_entropy(logits,targets)
        return logits, loss

    
    def generate(self, indicies: torch.Tensor, max_new_tokens: int):
        for i in range(max_new_tokens):
            logits, loss = self(indicies)
            # extract last time step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_index = torch.multinomial(probs,num_samples=1)
            indicies = torch.cat((indicies, next_index), dim=1)
            
        return indicies
    
model = BigramLanguageModel(vocabulary_size)
m = model.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evaluation_iterations)
        for k in range(evaluation_iterations):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(max_iterations):
    X_batch, Y_batch = get_batch('train')
    
    logits, loss = m(X_batch, Y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if steps % 100 == 0:
        print(f'{steps=} {loss=}')

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=400)[0].tolist()))
