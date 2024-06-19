import torch
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def nt_xnet_loss(x1, x2):
    T = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss

criterions = {
    "multilabel": F.binary_cross_entropy_with_logits,
    "multiclass": F.nll_loss,
    "info_nce": nt_xnet_loss
}