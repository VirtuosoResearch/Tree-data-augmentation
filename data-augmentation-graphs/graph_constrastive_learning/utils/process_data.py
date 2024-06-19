from sklearn.model_selection import StratifiedKFold
import torch

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def k_fold(dataset, folds, epoch_select, semi_split=1, seed = 12345):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    if dataset.data.y.shape[0] != len(dataset):
        labels = dataset.data.y==1
        labels = labels.view(len(dataset), -1)
        labels = labels[:, 0]
    else:
        labels = dataset.data.y 
    for _, idx in skf.split(torch.zeros(len(dataset)), labels):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    skf_semi = StratifiedKFold(semi_split, shuffle=True, random_state=seed) if semi_split > 1 else None
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train = train_mask.nonzero().view(-1)

        if semi_split > 1:
            tmp_dataset = dataset[idx_train]
            for _, idx in skf_semi.split(torch.zeros(idx_train.size()[0]), tmp_dataset.data.y[tmp_dataset.indices()]):
                idx_train = idx_train[idx]
                break

        train_indices.append(idx_train)

    return train_indices, test_indices, val_indices