import torch
from torch.functional import Tensor
from knn_pytorch import knn_pytorch


def knn(ref: Tensor, query: Tensor, k=1):
    """
    Compute k nearest neighbors for each query point.
    """
    assert ref.is_contiguous() and query.is_contiguous()
    device = ref.device
    ref = ref.float().to(device)
    query = query.float().to(device)
    inds = torch.empty(query.shape[0], k, query.shape[2]).long().to(device)
    knn_pytorch.knn(ref, query, inds)
    return inds - 1


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    ref = torch.tensor([[[1, 1], [1, 2], [1, 3.1]]]).transpose(-1, -2).contiguous().float()  # (B, C, N)
    query = torch.tensor([[[1, 2], [1, 3]]]).transpose(-1, -2).contiguous().float()  # (B, C, N)
    print('*' * 15 + "ref" + '*' * 15)
    print(ref)
    print(ref.shape)
    print('*' * 15 + "query" + '*' * 15)
    print(query)
    print(query.shape)
    ref.to("cuda")
    query.to("cuda")
    knn_idx = knn(ref, query, k=2)  # (B, k, N)
    print('*' * 15 + "result index" + '*' * 15)
    print(knn_idx)
    print(knn_idx.shape)
