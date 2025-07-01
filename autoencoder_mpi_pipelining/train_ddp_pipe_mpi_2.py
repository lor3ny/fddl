import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import torch.distributed as dist

#from mpi4py import MPI
import numpy as np
import os
import argparse
import time
import sys

from autoencoder import Autoencoder_PIPE, Encoder, Decoder
#from mnist_loader import MNISTLoader


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        save_every: int,
        rank: int,
        gpu_rank: int,
        #size: int,
        #comm: MPI.Comm,
        #sync_comm: MPI.Comm
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.model = model
        self.gpu_rank = gpu_rank
        self.rank = rank
        #self.size = size
        #self.comm = comm
        #self.sync_comm = sync_comm  # Used for synchronizing gradients
        self.device="cuda" #  #  Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu,


    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

        
    # def _synchronize_gradients(self) -> None:

    #     for param in self.model.parameters():
    #         if param.grad is None:
    #             continue
    #         # Gather gradients from all ranks
    #         grad = param.grad.data.cpu().numpy() # why cpu?
    #         avg_grad = np.zeros_like(grad)
    #         try:
    #             self.sync_comm.Allreduce(grad, avg_grad, op=MPI.SUM)
    #             avg_grad /= self.size  # Average the gradients
    #             param.grad.data = torch.tensor(avg_grad, device=self.device)

    #         except Exception as e:
    #             raise RuntimeError(f"Error synchronizing gradients: {e}.", flush=True)
            

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            print(f"[RANK {self.rank} GPU {self.gpu_rank}] Epoch {epoch}", flush=True) if self.rank == 0 else None

            dist.barrier()  # Sincronizzazione inizio epoca

            for batch_idx, (batch_data, _) in enumerate(self.train_data):
                # Flatten input
                inputs = batch_data.view(-1, 28*28).to(self.gpu_rank)

                if self.rank == 0:
                    # --- FORWARD SU RANK 0 ---
                    inputs.requires_grad_()
                    activations = self.model(inputs)

                    # Invia attivazioni a Rank 1
                    dist.send(activations.cpu(), dst=1, tag=batch_idx)

                    # Riceve il gradiente del tensore intermedio da Rank 1
                    grad_from_next = torch.empty((len(batch_data), 256), dtype=torch.float32)
                    dist.recv(grad_from_next, src=1, tag=batch_idx)
                    grad_from_next = grad_from_next.to(self.gpu_rank)

                    # --- BACKWARD SU RANK 0 ---
                    self.optimizer.zero_grad()
                    activations.backward(gradient=grad_from_next)
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            print(f"[Rank 0] Grad {name}: {p.grad.norm().item():.6f}", flush=True)
                        else:
                            print(f"[Rank 0] Grad {name}: None", flush=True)
                    self.optimizer.step()

                elif self.rank == 1:
                    # --- RICEVE ESECUZIONE DA RANK 0 ---
                    received_activations = torch.empty((len(batch_data), 256), dtype=torch.float32)
                    dist.recv(received_activations, src=0, tag=batch_idx)
                    received_activations = received_activations.to(self.gpu_rank)
                    received_activations.requires_grad_()
                    print(f"[Rank 1] Activations mean/std: {received_activations.mean():.4f} / {received_activations.std():.4f}", flush=True)
                    # --- FORWARD + BACKWARD SU RANK 1 ---
                    self.optimizer.zero_grad()
                    outputs = self.model(received_activations)

                    targets = batch_data.view(-1, 28*28).to(self.gpu_rank)
                    loss = self.criterion(outputs, targets)

                    loss.backward()
                    if received_activations.grad is not None:
                        print(f"[Rank 1] Grad input: {received_activations.grad.norm():.6f}", flush=True)
                    else:
                        print("[Rank 1] Grad input is None!", flush=True)
                    self.optimizer.step()

                    # Invia gradiente a Rank 0
                    grad_to_send = received_activations.grad.detach().cpu()
                    dist.send(grad_to_send, dst=0, tag=batch_idx)

                    # Stampa loss ogni 10 batch
                    if batch_idx % 10 == 0:
                        print(f"[RANK 1] Epoch [{epoch+1}/{max_epochs}], Batch [{batch_idx}], Loss: {loss.item():.6f}", flush=True)

            dist.barrier()  # Sincronizzazione fine epoca

        if self.rank == 1:
            print(f"[RANK 1] Training completo. Ultima loss: {loss.item():.6f}", flush=True)



#     # NODES SYNCHRONIZATION
#     # sync_start_time = MPI.Wtime()
#     # print(f"{self.rank} GPU {self.gpu_rank} -> Epoch {epoch} | Batch: {batch_idx} SYNCHING START", flush=True)
#     # self._synchronize_gradients() # in questo caso solo tra le gpu 3
#     # print(f"{self.rank} GPU {self.gpu_rank} -> Epoch {epoch} | Batch: {batch_idx} SYNCHING END", flush=True)
#     # sync_end_time = MPI.Wtime()
#     # local_syncs_lat.append(sync_end_time-sync_start_time)
#     # req.Wait() #non va messa qui


# def manual_model_split(model) -> PipelineStage:
#    if stage_index == 0:
#       # prepare the first stage model
#       for i in range(4, 8):
#             del model.layers[str(i)]
#       model.norm = None
#       model.output = None

#    elif stage_index == 1:
#       # prepare the second stage model
#       for i in range(4):
#             del model.layers[str(i)]
#       model.tok_embeddings = None

#    stage = PipelineStage(
#       model,
#       stage_index,
#       num_stages,
#       device,
#    )
#    return stage



def load_distribute_data(
        rank: int, 
        size: int, 
        batch_size: int,
        #comm: MPI.Comm
    ) -> tuple[DataLoader, DataLoader]:


    if not os.path.exists("./data/MNIST"):
        if rank == 0:
            print(f"[RANK: {rank}] Downloading MNIST dataset...")
            datasets.MNIST(root='./data', train=True, download=False, transform=transform)
            datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            print(f"[RANK: {rank}] DONE.")
    dist.barrier()

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def main(
    epochs: int,
    batch_size: int,
    save_every: int,
    world_size: int
):
    # Initialize MPI
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    backend = 'gloo' #if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, init_method="env://")
    print(f"STARTING", flush=True)
    rank = dist.get_rank()

    # group_ranks = [0]
    # world_group = comm.Get_group()
    # sub_group = world_group.Incl(group_ranks)
    # sub_comm = comm.Create(sub_group)

    gpu_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_rank)
    print(f"STARTING", flush=True)
    if rank == 0:
        print("+ ---------- TORCH LIBRARY CHECK ---------- +")
        print(f"PyTorch version: {torch.__version__}", flush=True)
        print(f"PyTorch CUDA version: {torch.version.cuda}", flush=True)
        print(f"PyTorch CUDNN version: {torch.backends.cudnn.version()}", flush=True)
        print(f"GPUs per node {torch.cuda.device_count()}", flush=True)
        print(f"Training with GPUs :)") if torch.cuda.is_available() else print("Training with CPUs", flush=True)
        print("+ ----------------------------------------- +")
    
    dist.barrier()
    print(f"[Rank {rank}] GPU_RANK={gpu_rank} on CUDA device {torch.cuda.current_device()} hostname={os.uname()[1]}", flush=True)

    # DATA MUST BE DIVIDED MANUALLY
    dist.barrier()

    # if rank == 0:
    train_loader, test_loader = load_distribute_data(rank=rank, size=world_size, batch_size=batch_size)

    # #MODEL TRAINING
    print(f"[RANK {rank}] Trainer is running...", flush=True) if rank == 0 else None

    layers = [Encoder, Decoder]
    model = layers[rank]().to(gpu_rank)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # # Il training deve gestire l'aggregazione del gradiente con la allreduce, rivederlo

    trainer = Trainer(
        model=model,
        train_data=train_loader,
        test_data=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        save_every=save_every,
        rank=rank,
        gpu_rank=gpu_rank,
        #size=size,
        #comm=comm,
        #sync_comm=sub_comm if rank in group_ranks else MPI.COMM_NULL
    )
    trainer.train(epochs)

    print(f"[RANK {rank}] Training done.", flush=True) if rank == 0 else None

    #if(rank == 0):
    torch.save(model.state_dict(), f"layer{gpu_rank}.pth")
    print(f"[RANK: {rank}] Model saved to layer{gpu_rank}.pth")

    dist.destroy_process_group()


if __name__ == "__main__":

    epochs = 20
    batch_size = 64
    save_every = 1
    latent_linear_size = 32
    
    parser = argparse.ArgumentParser(description="Example of parsing many CLI arguments.")
    parser.add_argument("--ntasks", type=int, help="Number of tasks", default=1)
    args = parser.parse_args()
    world_size = args.ntasks

    main(
        epochs=epochs,
        batch_size=batch_size,
        save_every=save_every,
        world_size=world_size
    )
