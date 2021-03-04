## Local train on CPU (testing)

CPU_DISTRIBUTED=1 PYTHONPATH=$PWD python3 scripts/yolov3/train.py --dataset=voc/2012

For multi GPU training, do not set CPU_DISTRIBUTED var.

# Configs
- base/train
    - gpu: v100 * 8
    - batch_size: 8
    - cpu: 3
    - memory: 32
- base/eval
    - gpu: v100 * 2
    - batch_size: 2
    - cpu: 3
    - memory: 32
