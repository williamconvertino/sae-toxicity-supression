#!/bin/bash

echo -e "Node\t\tTotalGPU\tAllocGPU\tAvailGPU\tTotalCPU\tAllocCPU\tAvailCPU\tState"

for node in dcc-h200-gpu-{01..07}; do
    info=$(scontrol show node "$node")

    total_gpu=$(echo "$info" | grep -oP 'CfgTRES=.*?gres/gpu=\K[0-9]+')
    alloc_gpu=$(echo "$info" | grep -oP 'AllocTRES=.*?gres/gpu=\K[0-9]+')
    total_cpu=$(echo "$info" | grep -oP 'CPUTot=\K[0-9]+')
    alloc_cpu=$(echo "$info" | grep -oP 'CPUAlloc=\K[0-9]+')
    state=$(echo "$info" | grep -oP 'State=\K\S+')

    alloc_gpu=${alloc_gpu:-0}
    alloc_cpu=${alloc_cpu:-0}

    avail_gpu=$((total_gpu - alloc_gpu))
    avail_cpu=$((total_cpu - alloc_cpu))

    echo -e "${node}\t${total_gpu}\t\t${alloc_gpu}\t\t${avail_gpu}\t\t${total_cpu}\t\t${alloc_cpu}\t\t${avail_cpu}\t\t${state}"
done