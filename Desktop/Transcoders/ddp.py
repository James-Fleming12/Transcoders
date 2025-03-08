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
        self.decoder = nn.Parameter(torch.zeros(self.latent_dim, self.input_dim))
        dtype=torch.float32
        #nn.init.kaiming_uniform_(self.decoder)
        self.encoder = nn.Linear(self.input_dim, self.latent_dim, dtype=torch.float32)
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(torch.zeros(input_dim, dtype=dtype))
        self.W_skip = nn.Parameter(torch.zeros(input_dim, input_dim, dtype=dtype))
        self.batch_size=batch_size
        self.k_values=[32,128,256]
        


  



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
        mean_encoded=torch.empty(x.shape[0],self.latent_dim).to(device)
        mean_decoded=torch.empty(x.shape[0],self.input_dim).to(device)
        final_l0_norm=0
        for i in self.k_values:
          encoded = self.encode(x,i)
          decoded = self.decode(x,encoded)
          mean_encoded+=encoded
        
          mean_decoded+=decoded
          total_variance = (x - x.mean(0)).pow(2).sum()
          loss= (MLP_output - decoded).pow(2).sum() 
          total_loss+=loss
        mean_encoded=mean_encoded/len(self.k_values)
        mean_decoded=mean_decoded/len(self.k_values)
        l0_norm = torch.count_nonzero(encoded,dim=0)
        l0_norm_final=torch.sum(l0_norm)
        
        total_loss=total_loss/ total_variance
        approximation_of_l0_norm=torch.norm(mean_encoded, p=1)
        
        
        return mean_encoded,mean_decoded,total_loss,l0_norm_final

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
dataset = load_dataset("Skylion007/openwebtext",streaming=True)['train']



def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

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
        self.model = model.to(self.gpu_id)
        self.model1=model1.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.model1=DDP(self.model1,device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, batch,epoch):




        if epoch==0:
            inputs = tokenizer(batch['text'], return_tensors='pt', padding="max_length", truncation=True,max_length=4)
            inputs = {k: v.to(self.gpu_id) for k, v in inputs.items()}
            MLP_output=gather_mlp_output(self.model.module,3,inputs)
            activations=gather_mlp_input(self.model.module,3,inputs)
            activations_flattened = activations.reshape(-1, 4096).to(self.gpu_id)
            mlp_output_flattened = MLP_output.reshape(-1, 4096).to(self.gpu_id)
            model1.b_dec.weight=empirical_mean
            print(activations_flattened.shape)
            print(mlp_output_flattened.shape)
            self.optimizer.zero_grad()
            encoded,decode,loss,l0_norm=self.model1.module(activations_flattened,mlp_output_flattened,self.gpu_id)
            print(loss)
            print(l0_norm//4)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                model1.decoder.copy_(model1.decoder / model1.decoder.norm(dim=1, keepdim=True))

            del activations, activations_flattened
            torch.cuda.empty_cache()
            return
            
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        
        inputs = tokenizer(batch['text'], return_tensors='pt', padding="max_length", truncation=True,max_length=4)
        inputs = {k: v.to(self.gpu_id) for k, v in inputs.items()}
        MLP_output=gather_mlp_output(self.model.module,3,inputs)
        activations=gather_mlp_input(self.model.module,3,inputs)
        activations_flattened = activations.reshape(-1, 4096).to(self.gpu_id)
        mlp_output_flattened = MLP_output.reshape(-1, 4096).to(self.gpu_id)
        print(activations_flattened.shape)
        print(mlp_output_flattened.shape)
        self.optimizer.zero_grad()
        encoded,decode,loss,l0_norm=self.model1.module(activations_flattened,mlp_output_flattened,self.gpu_id)
        print(loss)
        print(l0_norm//4)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            model1.decoder.copy_(model1.decoder / model1.decoder.norm(dim=1, keepdim=True))

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
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int,batch_size):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch,batch_size)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs(batch_size):
     # load your dataset
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    model1=Transcoders(batch_size=batch_size,input_dim=4096) # load your model
    
    optimizer = SignSGD(model.parameters(), lr=1e-5)
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
    dataset = load_dataset("Skylion007/openwebtext")['train']
    train_data = prepare_dataloader(dataset, batch_size)

    
        
        
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path,model1)
    trainer.train(total_epochs,batch_size)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)