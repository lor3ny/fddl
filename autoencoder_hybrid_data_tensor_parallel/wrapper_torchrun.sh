#!/bin/bash
HEAD_NODE_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$HEAD_NODE_IP
MASTER_PORT=29600

torchrun --nnodes $SLURM_NNODES --nproc_per_node 4 --rdzv-id $SLURM_JOB_ID --rdzv_backend c10d --rdzv_endpoint "$HEAD_NODE_IP:$MASTER_PORT" train_hybrid_data_tensor_nccl.py --ntasks $SLURM_NTASKS