#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."


mkdir -p logs save/objects save

EPOCHS=150           
USERS=100
LR=0.01               
SEED=1

DATASET=mnist
MODEL=mlp

FRAC=0.1              
E_LIST=(1 5 20)

declare -A BMAP=( ["inf"]=10000 ["50"]=50 ["10"]=10 )

run_block () {
  local IID=$1  

  local PART=$([ "$IID" = "1" ] && echo "iid" || echo "noniid")

  for E in "${E_LIST[@]}"; do
    for BL in inf 50 10; do
      B=${BMAP[$BL]}

      LOG="logs/q2_${MODEL}_${PART}_E${E}_B${BL}.txt"

      echo ">> $LOG"
      
      python src/federated_main.py \
        --model=${MODEL} --dataset=${DATASET} --iid=${IID} \
        --epochs=${EPOCHS} --num_users=${USERS} --frac=${FRAC} \
        --local_bs=${B} --local_ep=${E} --optimizer=sgd \
        --lr=${LR} --seed=${SEED} \
        --gpu=0 \
        > "${LOG}"
    done
  done
}

# IID and Non-IID
run_block 1
run_block 0
