## train

python3 scripts/yolov3/train.py --dataset=voc/2012


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
