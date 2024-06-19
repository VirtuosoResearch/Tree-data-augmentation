python train_simclr_medical.py \
    --config configs/multitask_simclr.json \
    --model resnet50 \
    --train_bilevel \
    --weight_lr 0.1  --update_weight_step 50 \
    --n_gpu 1 --device 0 --save_name best_fundus_multi_bi_50ep --epochs 50