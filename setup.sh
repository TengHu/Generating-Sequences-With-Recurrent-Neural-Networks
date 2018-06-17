#!/bin/sh

_='
Comments and some resources

GPU Performance
torch.backends.cudnn.benchmark = True

Check GPU Utilization
nvidia-smi -l 2
nvidia-smi --format=csv --query-gpu=power.draw,utilization.gpu,fan.speed,temperature.gpu -l 2

Bottlenecks could be
Data loading, or model is so small that launching the gpu compute is slower than actually computing on the gpu.




ssh AWS
Example
ssh -v -i ~/Documents/niel_hu.pem ubuntu@
scp -i ~/Documents/niel_hu.pem -r ubuntu@18.237.65.192:~/Niel/lstm/Generating-Sequences-With-Recurrent-Neural-Networks/checkpoint_best_DLSTM3.pth .

https://tspankaj.com/2017/08/13/gpu-training-bottlenecks/
https://pytorch.org/docs/stable/bottleneck.html
https://pytorch.org/docs/stable/data.html
https://pytorch.org/docs/stable/checkpoint.html
'

if [ ! -d cache ]; then
    mkdir cache
fi
