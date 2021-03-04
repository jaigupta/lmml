# Local run on CPU
CPU_DISTRIBUTED=1 PYTHONPATH=$PWD python3 scripts/yolov3/train.py --dataset=voc/2012

# Configs
- base:train/gpu
    - gpu: 8 x v100
    - cpu: 3
    - memory: 32
    - eval_every: 10k
    - eval_iters: 1k
- base:eval/gpu
    - gpu: 2 x t4
    - cpu: 3
    - memory: 32