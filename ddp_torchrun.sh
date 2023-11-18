#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES="2,3"
nohup torchrun --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=3 ddp_torchrun.py >>myresult.out 2>&1 &