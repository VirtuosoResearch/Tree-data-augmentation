import os 
os.environ['MKL_THREADING_LAYER'] = "GNU"
import argparse
import numpy as np
import pandas as pd

def main(args):
    task_list = [
                    "DropNodes", "PermuteEdges", "Subgraph", "MaskNodes",
                ]
    scale_list = [0.1, 0.2, 0.3, 0.4] # 0.5, 0.6
    prob_list = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    cur_augmentations = []
    cur_ratios = []
    cur_probs = []
    cur_tree_idxes = []
    potential_idxes = [0]
    
    sampled_task_dir = os.path.join("./sampled_tasks", "{}.txt".format(args.task_set_name))
    if not os.path.exists(sampled_task_dir):
        f = open(sampled_task_dir, "w")
        f.close()

    max_val = -np.inf; cur_depth = 0  
    max_augmentation = None; max_ratio = None; max_prob = None
    while cur_depth <= args.max_depth and len(potential_idxes) != 0:
        if len(potential_idxes) == 0: break

        # choose one point to grow 
        cur_idx = np.random.choice(potential_idxes, size=1)[0]
        potential_idxes.remove(cur_idx)
        tmp_tree_idxes = cur_tree_idxes[:] + [cur_idx]

        improved = False
        for i, augment_name in enumerate(task_list):
            for scale in scale_list:
                
                # check if the left or right node has appeared in the list
                pair_idx = ((cur_idx // 2) * 2 + 1) if cur_idx % 2 == 0 else ((cur_idx // 2) * 2 + 2)
                if pair_idx in cur_tree_idxes:
                    tmp_prob_list = [1-cur_probs[cur_tree_idxes.index(pair_idx)]]
                else:
                    tmp_prob_list = prob_list

                # enumerate the prob list
                for prob in tmp_prob_list:
                    tmp_augmentations = cur_augmentations[:] + [augment_name]
                    tmp_ratios = cur_ratios[:] + [scale]
                    tmp_probs = cur_probs[:] + [prob]
                    
                    augmentation_names = " ".join(tmp_augmentations)
                    augmentation_ratios = " ".join([str(ratio) for ratio in tmp_ratios])
                    augmentation_probs = " ".join([str(prob) for prob in tmp_probs])
                    tree_idxes = " ".join([str(idx) for idx in tmp_tree_idxes])

                    # load the results
                    result_dir = os.path.join("./results", args.save_name)
                    cur_val = None
                    if os.path.exists(result_dir):
                        result_df = pd.read_csv(os.path.join(result_dir, f"{args.save_name}.csv"), index_col=0, dtype={'Probs': 'str', 'Tree': 'str'})
                        target_metric = args.eval_metric

                        # check if exist
                        augmentation_str = "_".join([f"{a}_{r}" for a, r in zip(tmp_augmentations, tmp_ratios)])
                        prob_str = "_".join([f"{p}" for p in tmp_probs])
                        idx_str = "_".join([f"{i}" for i in tmp_tree_idxes])

                        tmp_df = result_df[result_df["Augmentation"] == augmentation_str]
                        tmp_df = tmp_df[tmp_df["Probs"] == prob_str]
                        tmp_df = tmp_df[tmp_df["Tree"] == idx_str]
                        if len(tmp_df) != 0:
                            cur_val = tmp_df[target_metric].values[0]
                            print(augmentation_names, augmentation_ratios, augmentation_probs, tree_idxes, cur_val)
                    if cur_val is None:
                        os.system("python train.py --model ResGCN --dataset {} --loss_name nt_xnet_loss --semi_split 1 --epochs {} --train_simclr --fold_idxes 0\
                                    --augmentation_names {} \
                                    --augmentation_ratios {} \
                                    --augmentation_probs {} \
                                    --tree_idxes {} \
                                    --mnt_metric val_loss --mnt_mode min\
                                    --device {} --save_name {}".format(
                                args.dataset, args.epochs, 
                                augmentation_names, # augmentation_names,
                                augmentation_ratios, # augmentation_ratios,
                                augmentation_probs, # augmentation_probs,
                                tree_idxes, # tree_idxes,
                                args.device, args.save_name
                        ))
                        print(augmentation_names, augmentation_ratios, augmentation_probs, tree_idxes)
                            
                        # load the results
                        result_dir = os.path.join("./results", args.save_name)
                        result_df = pd.read_csv(os.path.join(result_dir, f"{args.save_name}.csv"), index_col=0)
                        target_metric = args.eval_metric

                        # check if the result is better than the current best
                        cur_val = result_df[target_metric].values[-1]
                    if cur_val > max_val:
                        max_val = cur_val
                        max_augmentation = tmp_augmentations[:]
                        max_ratio = tmp_ratios[:]
                        max_prob = tmp_probs[:]
                        improved = True
        if not improved: 
            tmp_tree_idxes = tmp_tree_idxes[:-1]
            continue
        print("Current Tree:")
        print(tmp_tree_idxes, max_augmentation, max_ratio, max_prob)
        cur_tree_idxes = tmp_tree_idxes[:]
        sort_idxes = np.argsort(cur_tree_idxes)
        cur_augmentations = [max_augmentation[idx] for idx in sort_idxes]
        cur_ratios = [max_ratio[idx] for idx in sort_idxes]
        cur_probs = [max_prob[idx] for idx in sort_idxes]

        potential_idxes += [2*cur_idx+1, 2*cur_idx+2]
        print("Potential Idxes:", potential_idxes)

        tmp_sampled_tasks = "_".join(cur_augmentations) + "_" + "_".join([str(ratio) for ratio in cur_ratios]) + "_" \
            + "_".join([str(prob) for prob in cur_probs]) + "_" + "_".join([str(idx) for idx in cur_tree_idxes])
        with open(sampled_task_dir, "a") as f:
            f.write(tmp_sampled_tasks + "\n")

        cur_depth = int(np.math.log(max(cur_tree_idxes)+1, 2))+1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="NCI1", help='dataset name')
    parser.add_argument('--eval_metric', type=str, default="eval_valid_accuracy", help='eval metric')

    parser.add_argument("--task_set_name", type=str, default="test")
    parser.add_argument("--save_name", type=str, default="test")
    parser.add_argument("--max_depth", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="0")
    
    args = parser.parse_args()
    main(args)