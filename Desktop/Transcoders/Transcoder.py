import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

#have to fix MLP to make it more like the original paper
lr = 0.0004 # learning rate
# l1 sparsity regularization coefficient

class Transcoders(nn.Module):
  def __init__(self, input_dim: int, sparsity_penalty: int, expansion_factor: float = 16, device: str = 'cuda'): # Reorder arguments
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = int(input_dim * expansion_factor)
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=True)
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)
        self.sparsity_penalty = sparsity_penalty
        self.device=device
        self.MLP= nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.input_dim, bias=True),
        ).to(self.device)

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        

        loss= (((self.MLP(x) - decoded) ** 2).mean(dim=1) / (x**2).mean(dim=1)
      ).mean()+self.sparsity_penalty*torch.norm(encoded, p=1)
        
        l0_norm = torch.count_nonzero(encoded,dim=0)
        l0_norm_final=torch.sum(l0_norm)
        return decoded, encoded, loss,l0_norm_final

  def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.relu(self.encoder(x))

  def decode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(x)
from transformers import AutoTokenizer, AutoModelForCausalLM
#change to whatever model you want
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
def gather_mlp_input(model, target_layer, inputs):
    target_mlp_input = None

    def hook_fn(module,inputs, outputs):
        nonlocal target_mlp_input
        # module_input is a tuple; here we assume the first element is the residual stream
        target_mlp_input = inputs[0]
        return outputs

    # Register the hook on the mlp submodule of the target layer.
    handle = model.model.layers[target_layer].mlp.register_forward_hook(hook_fn)
    
    # Run the model's forward pass.
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove the hook after capturing the input.
    handle.remove()
    
    return target_mlp_input
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

from datasets import load_dataset

# Load the openwebtext dataset
dataset = load_dataset("Skylion007/openwebtext",streaming=True)['train']
dataloader = DataLoader(dataset, batch_size=400)
device='cuda'
# Iterate over the DataLoader
model=model.to(device)
#1536 is input dimension, change to whatever layer you need. 1.8e-5 is sparsity penalty. For large
#models generally use lower sparsity penalty. 
model1=Transcoders(1536,1.8e-5).to(device)
i=400
j=0

print("saved")
optimizer = optim.Adam(model1.parameters(), lr=2e-5, betas=(0.9, 0.999))
for batch in dataloader:
    # Each batch is a dict with keys corresponding to the dataset columns
    inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True,max_length=128)
    inputs = {k: v.to("cuda") for k, v in inputs.items()} 
    #change 3 to whatever layer you want
    activations=gather_mlp_input(model,3,inputs)
      
    activations_flattened = activations.reshape(-1, 1536).to(device)
    print(activations_flattened.shape)
      #chunk_file=f"chunk_{j}.pt"
      #j=j+1
      #torch.save(activations_flattened.cpu(), chunk_file)
    optimizer.zero_grad()
    encoded,decode,loss,l0_norm=model1(activations_flattened)
    print(loss)
    print(l0_norm//24576)#this is to find how many active features there are at one time on average
    
    loss.backward()
    optimizer.step()
      
    del activations, activations_flattened  
    torch.cuda.empty_cache()  
    print(i*128)
    if i*128>=150000000:
        break
    i=i+400


torch.save(model1.state_dict(),"Transcoder_for_qwen_distilled_1.5b_layer3.pt")
print("saved")


      # Stop after the first batch
    