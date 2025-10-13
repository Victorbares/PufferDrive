#!/bin/bash

INI_PATH="pufferlib/config/ocean/drive.ini"
CONTROLLERS=("invertible" "classic" "tp")

for CT in "${CONTROLLERS[@]}"; do
    # Update controller_type in the ini file
    sed -i "s/controller_type = .*/controller_type = \"$CT\"/" "$INI_PATH"
    # Show the modified line
    echo "Modified line: controller_type = \"$CT\""
    # Indicate which training is running
    echo "Running training with controller_type: $CT"
    puffer train puffer_drive --wandb
done
