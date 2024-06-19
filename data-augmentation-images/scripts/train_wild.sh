python train_erm.py --config configs/wild_config.json --device 1\
   --group_id 97 286 307 316 --epochs 5 \
   --train_bilevel --weight_lr 0.1 --collect_gradient_step 4 --update_weight_step 50