#!/bin/sh

:'
GPU Performance
torch.backends.cudnn.benchmark = True

Check GPU Utilization
nvidia-smi -l 2
nvidia-smi --format=csv --query-gpu=power.draw,utilization.gpu,fan.speed,temperature.gpu -l 2


ssh AWS
ssh -v -i ~/Documents/niel_hu.pem ubuntu@

'

if [ ! -d cache ]; then
    mkdir cache
fi
