#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EPOCHS=100

USERS=100
LR=0.01
SEED=1

DATASET=mnist

MODEL=mlp

E=1

C_LIST=(0.0 0.1 0.2 0.5 1.0)



# 1) IID
# 1.1) B = ∞ (whole dataset at the client is used set B=10000)
for C in "${C_LIST[@]}"; do

  python src/federated_main.py --model=${MODEL} --dataset=${DATASET} --iid=1 \
    --epochs=${EPOCHS} --num_users=${USERS} --frac=${C} \
    --local_bs=10000 --local_ep=${E} --optimizer=sgd --lr=${LR} --seed=${SEED} \
    --gpu=0 \
    > logs/q1_${MODEL}_iid_Binf_C${C}.txt
done

# 1.2) B = 10
for C in "${C_LIST[@]}"; do


  python src/federated_main.py --model=${MODEL} --dataset=${DATASET} --iid=1 \
    --epochs=${EPOCHS} --num_users=${USERS} --frac=${C} \
    --local_bs=10 --local_ep=${E} --optimizer=sgd --lr=${LR} --seed=${SEED} \
    --gpu=0 \
    > logs/q1_${MODEL}_iid_B10_C${C}.txt
done

# 2) non-IID
# 2.1) B = ∞ (here also whole dataset at the client is used)
for C in "${C_LIST[@]}"; do

  python src/federated_main.py --model=${MODEL} --dataset=${DATASET} --iid=0 \
    --epochs=${EPOCHS} --num_users=${USERS} --frac=${C} \
    --local_bs=10000 --local_ep=${E} --optimizer=sgd --lr=${LR} --seed=${SEED} \
    --gpu=0 \
    > logs/q1_${MODEL}_noniid_Binf_C${C}.txt
done

# 2.2) B = 10
for C in "${C_LIST[@]}"; do


  python src/federated_main.py --model=${MODEL} --dataset=${DATASET} --iid=0 \
    --epochs=${EPOCHS} --num_users=${USERS} --frac=${C} \
    --local_bs=10 --local_ep=${E} --optimizer=sgd --lr=${LR} --seed=${SEED} \
    --gpu=0 \
    > logs/q1_${MODEL}_noniid_B10_C${C}.txt
    
done
