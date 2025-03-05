import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from typing import Callable, Any
#have to fix MLP to make it more like the original paper
lr = 0.0004 # learning rate
# l1 sparsity regularization coefficient

k_values=[32,128]
class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict.update({prefix + "k": self.k, prefix + "postact_fn": self.postact_fn.__class__.__name__})
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], strict: bool = True) -> "TopK":
        k = state_dict["k"]
        postact_fn = ACTIVATIONS_CLASSES[state_dict["postact_fn"]]()
        return cls(k=k, postact_fn=postact_fn)


ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}
class Transcoders(nn.Module):
  def __init__(self, batch_size: int,input_dim: int, expansion_factor: float = 16, device: str = 'cuda'): # Reorder arguments
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = int(input_dim * expansion_factor)
        self.decoder = nn.Parameter(torch.zeros(self.latent_dim, self.input_dim))
        dtype=torch.float32
        #nn.init.kaiming_uniform_(self.decoder)
        self.encoder = nn.Linear(self.input_dim, self.latent_dim, device=device, dtype=torch.float32)
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(torch.zeros(input_dim, dtype=dtype, device=device))
        self.W_skip = nn.Parameter(torch.zeros(input_dim, input_dim, device=device, dtype=dtype))
        self.batch_size=batch_size
        self.k_values=k_values
        self.device=device
       


  



  def encode(self, x: torch.Tensor, k : int) -> torch.Tensor:
        
            
      topk=TopK(k=k)
      return topk(self.encoder(x))
  def decode(self,x: torch.Tensor,encoded: torch.Tensor)-> torch.Tensor:
        return encoded@self.decoder+x@self.W_skip+self.b_dec
  
  
  @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported() #speeds up forward by "2x"
    )
  def forward(self, x: torch.Tensor,MLP_output) -> Tuple[torch.Tensor, torch.Tensor]:
        total_loss=0
        mean_encoded=torch.empty(self.batch_size,self.latent_dim).to(device)
        mean_decoded=torch.empty(self.batch_size,self.input_dim).to(device)
        for i in self.k_values:
          encoded = self.encode(x,i)
          decoded = self.decode(x,encoded)
          mean_encoded+=encoded
          mean_decoded+=decoded
          total_variance = (x - x.mean(0)).pow(2).sum()
          loss= (MLP_output - decoded).pow(2).sum() / total_variance
          total_loss+=loss
        mean_encoded=mean_encoded/len(self.k_values)
        mean_decoded=mean_decoded/len(self.k_values)
        
        approximation_of_l0_norm=torch.norm(encoded, p=1)
        l0_norm = torch.count_nonzero(mean_encoded,dim=0)
        l0_norm_final=torch.sum(l0_norm)
        return mean_encoded,mean_decoded,total_loss,l0_norm_final
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def gather_mlp_output(model, target_layer, inputs):
    target_mlp_output = None

    def hook_fn(module, module_inputs, module_outputs):
        nonlocal target_mlp_output
        # Capture the output of the MLP layer
        target_mlp_output = module_outputs
        return module_outputs  # Ensure normal execution

    # Register the hook on the MLP submodule of the target layer.
    handle = model.model.layers[target_layer].mlp.register_forward_hook(hook_fn)
    
    # Run the model's forward pass.
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove the hook after capturing the output.
    handle.remove()
    
    return target_mlp_output



from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

from datasets import load_dataset

# Load the openwebtext dataset
dataset = load_dataset("Skylion007/openwebtext",streaming=True)['train']


# Iterate over the DataLoader
import torch
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=1)
device='cuda'
# Iterate over the DataLoader
model=model.to(device)
model1=Transcoders(batch_size=4,input_dim=1536).to(device)
i=30
j=0
import torch
from torch.optim import Optimizer

lr=1e-4
class SignSGD(Optimizer):
    """Steepest descent in the L-infty norm. From <https://arxiv.org/abs/1802.04434>"""

    def __init__(self, params, lr: float = 1e-3):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super(SignSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: None = None) -> None:
        assert closure is None, "Closure is not supported."

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(p.grad.sign(), alpha=-lr)
optimizer = SignSGD(model.parameters(), lr=lr)

for batch in dataloader:
    # Each batch is a dict with keys corresponding to the dataset columns
    inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True,max_length=4)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    MLP_output=gather_mlp_output(model,3,inputs)
    activations=gather_mlp_input(model,3,inputs)

    activations_flattened = activations.reshape(-1, 1536).to(device)
    mlp_output_flattened = MLP_output.reshape(-1, 1536).to(device)
    print(mlp_output_flattened.shape)
    print(activations_flattened.shape)
      #chunk_file=f"chunk_{j}.pt"
      #j=j+1
      #torch.save(activations_flattened.cpu(), chunk_file)
    optimizer.zero_grad()
    encoded,decode,loss,l0_norm=model1(activations_flattened,mlp_output_flattened)
    print(loss)
    print(l0_norm//4)
    loss.backward()
    optimizer.step()

    del activations, activations_flattened
    torch.cuda.empty_cache()
    print(i*1600)
    if i*1600>=30000000:
        break
    i=i+30
    break