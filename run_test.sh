#!/bin/bash

# Definir os valores que serão testados
DEVICES=("cuda" "cpu")
LEARNING_RATES=(1e-4)
BATCH_SIZES=(256)
NUM_EPOCHS=(10)
HIDDEN_DIMS=(128 512)
LATENT_DIMS=(50 100)

# Loop sobre todas as combinações possíveis
for device in "${DEVICES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for epochs in "${NUM_EPOCHS[@]}"; do
                for hidden_dim in "${HIDDEN_DIMS[@]}"; do
                    for latent_dim in "${LATENT_DIMS[@]}"; do
                        echo "Running: device=$device, lr=$lr, batch_size=$batch_size, epochs=$epochs, hidden_dim=$hidden_dim, latent_dim=$latent_dim"
                        python stress_test.py \
                            --device "$device" \
                            --learning_rate "$lr" \
                            --batch_size "$batch_size" \
                            --num_epochs "$epochs" \
                            --hidden_dim "$hidden_dim" \
                            --latent_dim "$latent_dim"
                    done
                done
            done
        done
    done
done
