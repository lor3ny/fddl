import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
import numpy as np
import os
import argparse
import time
import sys

from autoencoder import Autoencoder_PIPE, Layer0, Layer1, Layer2, Layer3
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
        size: int,
        comm: MPI.Comm,
        sync_comm: MPI.Comm
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.model = model
        self.gpu_rank = gpu_rank
        self.rank = rank
        self.size = size
        self.comm = comm
        self.sync_comm = sync_comm  # Used for synchronizing gradients
        self.device="cuda" #  #  Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu,


    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    # Actually we are using this trainer with an Autoencoder, so I don't really know if this evaluator is good
    '''
    def _evaluate (self) -> None:
        
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_data:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # Get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100 * correct / total
        
        return accuracy
    '''
        
    def _synchronize_gradients(self) -> None:

        for param in self.model.parameters():
            if param.grad is None:
                continue
            # Gather gradients from all ranks
            grad = param.grad.data.cpu().numpy() # why cpu?
            avg_grad = np.zeros_like(grad)
            try:
                self.sync_comm.Allreduce(grad, avg_grad, op=MPI.SUM)
                avg_grad /= self.size  # Average the gradients
                param.grad.data = torch.tensor(avg_grad, device=self.device)

            except Exception as e:
                raise RuntimeError(f"Error synchronizing gradients: {e}.", flush=True)

    def train(self, max_epochs: int):


        local_syncs_lat = []
        local_epochs_lat = []

        start_time = MPI.Wtime()
        for epoch in range(max_epochs):
            print(f"[RANK {self.rank} GPU {self.gpu_rank}] Epoch {epoch} | Batches: {len(self.test_data)}", flush=True) if self.rank == 0 else None
            e_start_time = MPI.Wtime()
            total_loss = 0.0

            self.comm.Barrier()  # Synchronize all ranks before starting the epoch

            for batch_idx, (batch_data, batch_target) in enumerate(self.train_data):

                if self.gpu_rank == 0:
                    print(f"-> Epoch {epoch} | Batch: {batch_idx}", flush=True)

                    # FORWARD LAYER 0
                    inputs = batch_data.view(-1, 28*28).to(self.gpu_rank)
                    inputs.requires_grad_()
                    outputs_step0 = self.model(inputs)

                    # SEND LAYER 0
                    self.comm.Send(outputs_step0.detach().cpu().numpy(), dest=self.rank+1, tag=0)

                    # WAITING LAYER 1 GRADIENT
                    grad_from_1 = torch.empty((len(batch_data), 784), dtype=torch.float32)
                    self.comm.Recv([grad_from_1.numpy(), MPI.FLOAT], source=self.rank+1, tag=0)

                    # BACKWARD LAYER 0
                    inputs.backward(grad_from_1.to(self.gpu_rank))

                    # NODES SYNCHRONIZATION
                    # sync_start_time = MPI.Wtime()
                    # print(f"{self.rank} GPU {self.gpu_rank} -> Epoch {epoch} | Batch: {batch_idx} SYNCHING START", flush=True)
                    # self._synchronize_gradients() # in questo caso solo tra le gpu 3
                    # print(f"{self.rank} GPU {self.gpu_rank} -> Epoch {epoch} | Batch: {batch_idx} SYNCHING END", flush=True)
                    # sync_end_time = MPI.Wtime()
                    # local_syncs_lat.append(sync_end_time-sync_start_time)
                    # req.Wait() #non va messa qui

                elif self.gpu_rank == 1:

                    print(f"{self.rank} GPU {self.gpu_rank} -> Epoch {epoch} | Batch: {batch_idx} TAKEN", flush=True)
                    # WAITING LAYER 0
                    outputs_step0 = torch.empty((len(batch_data), 128), dtype=torch.float32)
                    self.comm.Recv([outputs_step0.numpy(), MPI.FLOAT], source=self.rank-1, tag=0)
                    outputs_step0 = outputs_step0.to(self.gpu_rank)
                    outputs_step0.requires_grad_()

                    # FORWARD LAYER 1
                    outputs_step1 = self.model(outputs_step0)
                    
                    # SEND LAYER 1
                    self.comm.Send(outputs_step1.detach().cpu().numpy(), dest=self.rank+1, tag=0)

                    # WAITING LAYER 2 GRADIENT
                    grad_from_2 = torch.empty((len(batch_data), 111), dtype=torch.float32)
                    self.comm.Recv([grad_from_2.numpy(), MPI.FLOAT], source=self.rank+1, tag=0)

                    # BACKWARD LAYER 1
                    outputs_step1.backward(grad_from_2.to(self.gpu_rank))

                    # SEND LAYER 1 GRADIENT
                    self.comm.Send(outputs_step0.grad.data.cpu().numpy(), dest=self.rank-1, tag=0)


                elif self.gpu_rank == 2:
                    # WAITING LAYER 1
                    outputs_step1 = torch.empty((len(batch_data), 111), dtype=torch.float32)
                    self.comm.Recv([outputs_step1.numpy(), MPI.FLOAT], source=self.rank-1, tag=0)

                    # FORWARD LAYER 2
                    outputs_step1 = outputs_step1.to(self.gpu_rank)
                    outputs_step1.requires_grad_()
                    outputs_step2 = self.model(outputs_step1)

                    # SEND LAYER 2
                    self.comm.Send(outputs_step2.detach().cpu().numpy(), dest=self.rank+1, tag=0)

                    # WAITING LAYER 3 GRADIENT
                    grad_from_3 = torch.empty((len(batch_data), 128), dtype=torch.float32)
                    self.comm.Recv([grad_from_3.numpy(), MPI.FLOAT], source=self.rank+1, tag=0)

                    # BACKWARD LAYER 2
                    outputs_step2.backward(grad_from_3.to(self.gpu_rank))

                    # SEND LAYER 2 GRADIENT
                    self.comm.Send(outputs_step1.grad.data.cpu().numpy(), dest=self.rank-1, tag=0)
                    


                elif self.gpu_rank == 3:

                    # WAITING LAYER 3   
                    outputs_step2 = torch.empty((len(batch_data), 128), dtype=torch.float32)
                    self.comm.Recv([outputs_step2.numpy(), MPI.FLOAT], source=self.rank-1, tag=0)

                    # FORWARD LAYER 3
                    outputs_step2 = outputs_step2.to(self.gpu_rank)
                    outputs_step2.requires_grad_()
                    outputs = self.model(outputs_step2)

                    # BACKWARD LAYER 3 FROM LOSS
                    inputs = batch_data.view(-1, 28*28).to(self.gpu_rank)
                    loss = self.criterion(inputs, outputs)
                    total_loss += loss.item()
                    loss.backward()

                    # SEND LAYER 3 GRADIENT
                    self.comm.Send([outputs_step2.grad.data.cpu().numpy(), MPI.FLOAT], dest=self.rank - 1, tag=0)

                    #total_loss += loss.item()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    print(f"{self.rank} GPU {self.gpu_rank} -> Epoch {epoch} | Batch: {batch_idx} GRAD SENT", flush=True)
                    
                else:
                    print(f"[RANK {self.rank}] Error on GPU rank")

            self.comm.Barrier()
            if self.gpu_rank == 3:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(self.train_data):.6f}', flush=True)

            '''
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            '''

            # local_epochs_lat.append(MPI.Wtime() - e_start_time)
            # avg_loss = total_loss / len(self.train_data)
            # all_avg_loss = (self.comm.allreduce(avg_loss, op=MPI.SUM) / self.size)
            # print(f"-> Epoch {epoch} | Avg Loss: {all_avg_loss}") if self.rank == 0 else None

            # test_accuracy = self._evaluate()
            
        # local_training_time = MPI.Wtime() - start_time
        # max_training_time = self.comm.allreduce(local_training_time, op=MPI.MAX)

        # Printare Epochs Time (voglio mantenere l'associazione per epoch) #ogni pro crea un file e salva i tempi, il file riporta il suo rank
        # Printare Syncs Time (non so se voglio mantenere l'associazione per epoch) #ogni pro crea un file e salva i tempi, il file riporta il suo rank
        # Per fare plot

        # print(f"Final execution time: {max_training_time}s") if self.rank == 0 else None



def load_distribute_data(
        rank: int, 
        size: int, 
        batch_size: int,
        comm: MPI.Comm
    ) -> tuple[DataLoader, DataLoader]:


    if not os.path.exists("./data/MNIST"):
        if rank == 0:
            print(f"[RANK: {rank}] Downloading MNIST dataset...")
            datasets.MNIST(root='./data', train=True, download=False, transform=transform)
            datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            print(f"[RANK: {rank}] DONE.")
    comm.barrier()

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    # Split dataset into subsets for each rank
    total_samples = len(train_dataset)
    if rank == 0:
        print(f"[RANK {rank}] This dataset has {total_samples} samples", flush=True)

    sample_per_rank = total_samples // size
    remainder = total_samples % size
    indices = list(range(total_samples))

    # Distribute samples among ranks
    start_index = rank * sample_per_rank + min(rank, remainder)
    extra = 1 if rank < remainder else 0
    local_samples = sample_per_rank + extra
    end_index = start_index + local_samples
    local_indices = indices[start_index:end_index]
    print(f"[RANK {rank}] I have {local_samples} samples. From {start_index} to {end_index}", flush=True)

    # Create DataLoader for local dataset
    # Bisogna passare local_indices
    local_dataset = Subset(train_dataset, local_indices)
    train_loader = DataLoader(
        local_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def main(
    epochs: int,
    batch_size: int,
    save_every: int,
):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    group_ranks = [0]
    world_group = comm.Get_group()
    sub_group = world_group.Incl(group_ranks)
    sub_comm = comm.Create(sub_group)

    gpu_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_rank)

    if rank == 0:
        print("+ ---------- TORCH LIBRARY CHECK ---------- +")
        print(f"PyTorch version: {torch.__version__}", flush=True)
        print(f"PyTorch CUDA version: {torch.version.cuda}", flush=True)
        print(f"PyTorch CUDNN version: {torch.backends.cudnn.version()}", flush=True)
        print(f"GPUs per node {torch.cuda.device_count()}", flush=True)
        print(f"Training with GPUs :)") if torch.cuda.is_available() else print("Training with CPUs", flush=True)
        print("+ ----------------------------------------- +")
    
    comm.Barrier()
    print(f"[Rank {rank}] GPU_RANK={gpu_rank} on CUDA device {torch.cuda.current_device()} hostname={os.uname()[1]}", flush=True)

    # DATA MUST BE DIVIDED MANUALLY
    comm.Barrier()

    # if rank == 0:
    train_loader, test_loader = load_distribute_data(rank=rank, size=size, batch_size=batch_size, comm=comm)

    # I dati vanno distribuiti solo tra le gpu 0, gli altri non hanno bisogno
    # Se ci sono 4 nodi 4 gpu per nodo, avremo 16 processi. Quindi solo 4 processi avranno train_data, gli altri no
    # Ricorda sempre che per ora 0 legge e computa, 1-2 computano, 3 computa fa discesa del gradiente e sincronizza (bho forse meh)
    # Tutti dovrebbe fare discesa, forse è meglio se 3 manda a 0-1-2. 0-1-2-3 fanno discesa e poi fanno aggregazione tutti (tutti i processi di tutti i nodi)

    # MODEL INIT
    model = Autoencoder_PIPE(28*28, 32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model = model.to(gpu_rank)
    # not needed
    #model = DDP(model, device_ids=[gpu_rank])

    # #MODEL TRAINING
    print(f"[RANK {rank}] Trainer is running...", flush=True) if rank == 0 else None

    model = Autoencoder_PIPE(28*28, 32)

    layers = [Layer0, Layer1, Layer2, Layer3]
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
        size=size,
        comm=comm,
        sync_comm=sub_comm if rank in group_ranks else MPI.COMM_NULL
    )
    trainer.train(epochs)

    print(f"[RANK {rank}] Training done.", flush=True) if rank == 0 else None

    #if(rank == 0):
    torch.save(model.state_dict(), f"layer{gpu_rank}.pth")
    print(f"[RANK: {rank}] Model saved to layer{gpu_rank}.pth")


if __name__ == "__main__":

    epochs = 50
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
        save_every=save_every
    )
