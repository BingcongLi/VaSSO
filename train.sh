
CUDA_VISIBLE_DEVICES=0 \
python train.py --opt sam-sgd \
                --dataset CIFAR100_cutout \
                --seed 3107 \
                --lr 0.1 \
                --weight_decay 1e-3 \
                --model wideresnet28x10 \
                --rho 0.2


CUDA_VISIBLE_DEVICES=0 \
python train.py --opt vasso-sgd \
                --dataset CIFAR100_cutout \
                --seed 3107 \
                --lr 0.1 \
                --weight_decay 1e-3 \
                --model wideresnet28x10 \
                --rho 0.2 \
                --theta 0.9

