# Overview

This repository provides implementations for a top-down search algorithm for learning a tree-structured composition of data augmentations on a given dataset. 

## Data Preparation

**Protein graph classification.**

We provide a link for downloading the newly constructed [protein graph dataset](https://drive.google.com/file/d/1Bbl-urCGbnUNim6A0ddobI8MMxeiW9Bu/view?usp=sharing) from AlphaFold. Please download and unzip it under the folder of `./data-augmentation-graphs/protein-function-prediction/dataset`.  

**Wildlife image classification.**

For wildlife image classification, please refer to [the WILDS benchmark](https://github.com/p-lambda/wilds) for downloading the iWildCam dataset. 

**Medical image classification.**

For medical image datasets containing eye fundus images for diabetic retinopathy classification. Please refer to `./data-augmentation-images/README.md` for downloading and formatting the images.  Thanks to the authors of [BenchMD](https://github.com/rajpurkarlab/BenchMD#datasets) for providing their data processing code online.

**Graph contrastive learning.**

For datasets from TUDataset, our code automatically downloads the data. Please create a `data` folder under the `./data-augmentation-graphs/graph_contrastive_learning` folder. 


## Graph classification

Please use the `./data-augmentation-graphs` folder.

```
cd data-augmentaton-graphs
```

### Step 1: Searching for tree-based compositions

We provide the implementation of our algorithm that searches the tree-based composition in the `search_tree_based_augment.py`. We provide an example below to run our algorithm. 

**Protein graph classification**

Run the following script under the  `./data-augmentation-graphs/protein_graph_classification` folder. 

Use `train_tree_based_augment.py` to search for the composition of data augmentations. 

```bash
python search_tree_based_augment.py --num_groups 4 --group_ids 0\
    --task_set_name tree_augment_group_0 --save_name tree_augment_group_0\
    --max_depth 4 --epochs 100 --device 0
```

- `--group_ids` specifies the group ids to search the composition for. Provide a number between 0 and 15. 
- `--max_depth` specifies the maximum depth of the tree composition. 
- `--task_set_name` specifies the name of the file that saves the searched composition. 
- `--save_name` specifies the name of the 

**Graph Contrastive Learning**

Similarly, we can run a similar script for graph contrastive learning under the `./data-augmentation-graphs/graph_contrastive_learning` folder. We provide an example to find a tree based composition in one graph dataset:

```bash
python train_tree_based_augment.py --dataset NCI1 --epochs 10\
    --task_set_name tree_NCI1 --save_name tree_NCI1\
    --device 0 --max_depth 4
```

### Step 2: Learning a forest of trees on multiple subpopulations

After finding a tree-based composition in each subpopulation, we provide an algorithm to train a joint model by combining the trees of subpopulations in a weighted training manner. We provide an example below to run our algorithm. 

**Protein function prediction**

Run under the  `./data-augmentation-graphs/protein_graph_classification` folder. Use `train.py` to train a model on the protein function prediction dataset.  

```bash
python train_multi_groups.py --task_idxes -1 --num_groups 4 --group_ids -1 --labeling_rate_threshold 0.005 \
    --device 3 --runs 2 --epochs 50 --save_name results\
    --train_bilevel --update_step 50 --weight_lr 0.1
```

- Specify `--train_bilevel` to run our algorithm
- `--update_step` specifies the number of SGD steps to update combination weights
- `--weight_lr` specifies the learning rate to update combination weights

**Graph Contrastive Learning**

Run under the `./data-augmentation-graphs/graph_contrastive_learning` folder. Use `train.py` to train a contrastive model on TUDatasets. We provide an example to run a contrastive learning model on a graph dataset:

```bash
python train.py --dataset NCI1 --train_simclr --loss_name nt_xnet_loss --semi_split 1 --epochs 100 --augmentation_names DropNodes PermuteEdges --augmentation_ratios 0.1 0.1 \
--fold_idxes 0 --mnt_metric val_loss --mnt_mode min\
--device 0 --save_name test
```

## Image classification

This folder includes experiments for finding the tree-based composition of data augmentations on image data sets. For each experiment, we will work on several datasets datasets, which are  `CIFAR`  and `iWildCam` datasets for supervised learning, `fundus` (`Messidor`, `Aptos`, and `Jinchi`) datasets for self-supervised learning. 

### Step 1: Searching for tree-based composition

We implement our algorithm for searching the tree-based composition in the following scripts:

- For CIFAR datasets, use `search_tree_based_augment_cifar.py`
- For the iWildCam dataset, use `search_tree_based_augment_wilds.py`
- For self-supervised learning, use `search_tree_based_augment_simclr.py` 

The three scripts share similar logic. We give an example to use one script below. 

**Search tree-based composition on the iWildCam dataset**

Use the following script to search a tree-based composition on group 307 of the iWildCam dataset

```bash
python search_tree_based_augment_wilds.py \
    --dataset iwildcam --group_id 307 --eval_metric val_accuracy \
    --task_set_name wildcam_307 --save_name wildcam_307 \
    --epochs 10 --batch_size 16 --device 0 
```

Key arguments:

- `--group_id`: The group id of the dataset. It would be used to pick a subset that in the same domain from the whole dataset.
- `--eval_metric` : The value for selecting the best combination.
- `--task_set_name`: specifying the path for saving the best subset.
- `--save_name`: specifying the path for saving all the searching result that saved as `.cvs`

If you need more customization, please check the `.json` file that is used inside `search_tree_based_augment_wilds.py`

The commands to use other scripts would be similar. We provide the examples to run other search files in the `scripts` folder:

- On CIFAR-10, refer to `./scripts/search_cifar10.sh`
- On iWildCam, refer to `./scripts/search_wilds.sh`
- On medical image datasets, refer to `./scripts/search_medical.sh`

### Step 2: Learning a forest of trees on multiple subpopulations

After searching the tree-based composition on each subpopulation, we can run scripts to train a joint model by combining the trees in a weighted training scheme. 

- Use `train_wilds.py` and use `--train_bilevel` to run the algorithm on the iWildCam data set.
- Use `train_simclr_medical` and use `--train_bilevel` to run the algorithm on the medical image data set.

We give an example below. The script runs the algorithm on three groups of iWildCam dataset. Before running this script, specify the corresponding augmentations under a `txt` file in the `specified_augmentations` folder. 

```bash
python train_wilds.py \
    --config configs/wild_config.json \
    --group_ids 97 286 307 \
    --train_bilevel --weight_lr 0.1  --update_weight_step 50\
    --n_gpu 1 --device 1 --save_name test --runs 1\
```

Key arguments:

 - `--group_ids` : the group(domain) id that we used to train together.
 - `--save_name`: the name of the save folder
 - `--weight_lr`: the learning rate of the weight. It is useful when we train the model with `BilevelTrainer` or `GroupDOR`.
 - `--update_weight_step`: the number SGD steps to update weights.

We also provide examples to run our algorithm on other datasets:

- For medical image datasets, refer to `./scripts/run_medical.sh`

## Requirements

We provide two sets of python environment setups for both graph and image data sets. 

For experiments on image datasets, install requirements as requirements inside `data-augmentation-images` dataset:

```bash
cd data-augmentaton-images
pip install -r requirements.txt
```

For experiments on graph datasets, install requirements as requirements inside `data-augmentation-graphs` dataset:

```bash
cd data-augmentaton-graphs
pip install -r requirements.txt
```
### Acknowledgment

Thanks to the authors of the following repositories for providing their implementation publicly available.

- **[GraphCL](https://github.com/Shen-Lab/GraphCL)**
- **[Strategies for Pre-training Graph Neural Networks](https://github.com/snap-stanford/pretrain-gnns)**
- **[BenchMD](https://github.com/rajpurkarlab/BenchMD)**
- **[Pytorch-RandAugment](https://github.com/ildoonet/pytorch-randaugment)**
