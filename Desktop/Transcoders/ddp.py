import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from typing import Callable, Any
#have to fix MLP to make it more like the original paper
lr = 0.0004 # learning rate
# l1 sparsity regularization coefficient
from torch.optim import Optimizer
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
k_values=[32,128,256]
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
        result.scatter_(-1,topk.indices, values)
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
  def __init__(self, batch_size: int,input_dim: int,expansion_factor: float = 16 ): # Reorder arguments
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = int(input_dim * expansion_factor)
        
        dtype=torch.bfloat16
        #nn.init.kaiming_uniform_(self.decoder)
        self.encoder = nn.Linear(self.input_dim, self.latent_dim, dtype=torch.float32)
        self.decoder = nn.Parameter(torch.zeros_like(self.encoder.weight.data))
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(torch.zeros(input_dim, dtype=dtype))
        self.W_skip = nn.Parameter(torch.zeros(input_dim, input_dim, dtype=dtype))
        self.batch_size=batch_size
        self.global_step=1900
        self.k_decay_steps=1000
        self.initial_k=4096*10
        self.final_k=32
        self.k_values=[32,128,256]
        


  


  def get_current_k(self) -> int:
        """Get the current k value based on a linear decay schedule."""
        if self.global_step >= self.k_decay_steps:
            return self.final_k

        progress = self.global_step / self.k_decay_steps
        return round(self.initial_k * (1 - progress) + self.final_k * progress)
  def encode(self, x: torch.Tensor, k : int) -> torch.Tensor:


      topk=TopK(k=k)
      return topk(self.encoder(x))
  def decode(self,x: torch.Tensor,encoded: torch.Tensor)-> torch.Tensor:
        return encoded@self.decoder+x@self.W_skip+self.b_dec


  @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported() #speeds up forward by "2x"
    )
  def forward(self, x: torch.Tensor,MLP_output,device) -> Tuple[torch.Tensor, torch.Tensor]:
      total_loss=0
      if self.global_step>=1000:
          
        
        k=self.get_current_k()
        encoded = self.encode(x,k)
        decoded = self.decode(x,encoded)
        encoded_2=self.encode(x,k*4)
        decoded_2=self.decode(x,encoded_2)
          
        total_variance = (x - x.mean(0)).pow(2).sum()
        loss= (MLP_output - decoded).pow(2).sum() 
        loss=loss/total_variance
        
        loss_2=(MLP_output-decoded_2).pow(2).sum()
        loss_2=loss_2/total_variance
        total_loss=total_loss+loss+(loss_2/8)
        self.global_step=self.global_step+1
        print(f"global steps is{self.global_step}")  
        return (encoded+encoded_2).mean(),(decoded+decoded_2).mean(),total_loss
      else:
        k=self.get_current_k()
        encoded = self.encode(x,k)
        decoded = self.decode(x,encoded)
          
        total_variance = (x - x.mean(0)).pow(2).sum()
        loss= (MLP_output - decoded).pow(2).sum() 
        loss=loss/total_variance
        
       
        total_loss=total_loss+loss
        self.global_step=self.global_step+1
        print(f"global steps is{self.global_step}")  
        return encoded,decoded,total_loss
      
      
        
        
        
        
        

from transformers import AutoTokenizer, AutoModelForCausalLM


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




def ddp_setup():
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        print("Error: LOCAL_RANK is not set!")
    else:
        print("LOCAL_RANK:", local_rank)
    torch.cuda.set_device(int(local_rank))
    init_process_group(backend="nccl")
    print("Process group initialized.")
    print("Rank:", torch.distributed.get_rank(), "World Size:", torch.distributed.get_world_size())

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        model1: torch.nn.Module
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model=model
        self.model1=model1.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        

# Assuming ddp_setup() has been called already.
        
        self.model = DDP(model,device_ids=[self.gpu_id])
        self.model1=DDP(model1,device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model1.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, batch,epoch):


        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

        if epoch==0:
            inputs = tokenizer(batch['text'], return_tensors='pt', padding="max_length", truncation=True,max_length=2048)
            inputs = {k: v.to(self.gpu_id) for k, v in inputs.items()}
            MLP_output=gather_mlp_output(self.model.module,3,inputs)
            activations=gather_mlp_input(self.model.module,3,inputs)
            activations_flattened = activations.reshape(-1, 4096).to(self.gpu_id)
            mlp_output_flattened = MLP_output.reshape(-1, 4096).to(self.gpu_id)
            print(activations_flattened.shape)
            print(mlp_output_flattened.shape)
            self.optimizer.zero_grad()
            encoded,decode,loss=self.model1.module(activations_flattened,mlp_output_flattened,self.gpu_id)
            empirical_mean=mlp_output_flattened.mean(dim=-1)
            self.model1.module.b_dec.weight=empirical_mean
            print(loss)
            loss.backward()
            self.optimizer.step()
            

            del activations, activations_flattened
            torch.cuda.empty_cache()
            return
            
        
        
        inputs = tokenizer(batch['text'], return_tensors='pt', padding="max_length", truncation=True,max_length=2048)
        inputs = {k: v.to(self.gpu_id) for k, v in inputs.items()}
        MLP_output=gather_mlp_output(self.model.module,3,inputs)
        activations=gather_mlp_input(self.model.module,3,inputs)
        activations_flattened = activations.reshape(-1, 4096).to(self.gpu_id)
        mlp_output_flattened = MLP_output.reshape(-1, 4096).to(self.gpu_id)
        print(activations_flattened.shape)
        print(mlp_output_flattened.shape)
        self.optimizer.zero_grad()
        
        encoded,decode,loss=self.model1.module(activations_flattened,mlp_output_flattened,self.gpu_id)
        print(loss)
        loss.backward()
        self.optimizer.step()
        
        del activations, activations_flattened
        torch.cuda.empty_cache()

    def _run_epoch(self, epoch,batch_size):
        
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {batch_size} ")
        self.train_data.sampler.set_epoch(epoch)

        
        for batch in self.train_data:
            
           
            self._run_batch(batch,epoch)
            
            break
            
            

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model1.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int,batch_size):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch,batch_size)
            if epoch % self.save_every == 0:
                if self.gpu_id == 0:
                    self._save_snapshot(epoch)
            torch.distributed.barrier() 


def load_train_objs(batch_size):
     # load your dataset
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",load_in_8bit=True)
    model1=Transcoders(batch_size=batch_size,input_dim=4096) # load your model
    
    optimizer = SignSGD(model1.parameters(), lr=1e-5)
    return  model,model1,optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    model,model1, optimizer = load_train_objs(batch_size)
    dataset = load_dataset("Skylion007/openwebtext",trust_remote_code=True)['train']
    train_data = prepare_dataloader(dataset, batch_size)

    
        
        
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path,model1)
    trainer.train(total_epochs,batch_size)
    torch.save(model1.state_dict(),"Transcoder_for_Llama_distilled_8b_layer3.pt")
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)