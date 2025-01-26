import torch
import torch.linalg as LA
import random

from typing import Callable, List
import math

def data_to_blocks(x: torch.Tensor, y: torch.Tensor, blksz: int, blk_type: str = "normal"):
    if blk_type == "normal":
        return NormalBlocks(x, y, blksz)
    elif blk_type == "optim":
        return OptimizationBlocks(x, y, blksz)
    else:
        raise RuntimeError(f"Block type {blk_type} is not supported.")

class Block:
    def __init__(self, data_store: torch.Tensor, label_store: torch.Tensor, blk_ind_start: int, blk_size: int):
        self.data_store  = data_store
        self.label_store = label_store
        self.start_ind  = blk_ind_start
        self.blk_size = blk_size
        self.device   = data_store.device

    @property
    def data(self):
        return self.data_store[self.blk_ind]
    
    @property
    def label(self):
        return self.label_store[self.blk_ind]
    
    @property
    def blk_ind(self) -> torch.Tensor:
        return torch.arange(self.start_ind, self.start_ind + self.blk_size)
        
    
class NormalBlocks:
    block_list: List[Block]
    def __init__(self, data: torch.Tensor, label: torch.Tensor, blk_size: int):
        if blk_size < 0:
            raise RuntimeError("The parameter blk_size should be a positive integer")
        self.block_list = []
        self.block_size = blk_size
        self.num_blocks = math.ceil(data.size(0) / blk_size)
        self.data_num   = data.size(0)

        for idx in range(self.num_blocks):
            blk_ind_start  = idx * self.block_size
            blk_size = min(self.block_size, data.size(0) - blk_ind_start)
            self.block_list.append(Block(data, label, blk_ind_start, blk_size))

        self.index_list = list(range(self.num_blocks))
        self.cylic_idx = -1

    def shuffle(self):
        random.shuffle(self.index_list)

    def __getitem__(self, idx: int):
        return self.block_list[idx]
    
    def __len__(self):
        return self.num_blocks
    
    def __iter__(self):
        for i in self.index_list:
            yield self[i]

    def next(self):
        self.cylic_idx = (self.cylic_idx + 1) % self.num_blocks
        if self.cylic_idx == 0:
            self.shuffle()
        return self[self.cylic_idx]

class OptimizationBlocks(NormalBlocks):
    block_list: List[Block]
    def __init__(self, data: torch.Tensor, label: torch.Tensor, blk_size: int):
        super().__init__(data, label, blk_size)
        # only support uniform distribution now
        self.prob_list = [1.0 / self.num_blocks for _ in range(self.num_blocks)]

        # may include gradient information for block selection in the future...

    def random_pick(self):
        return random.choices(self.block_list, weights=self.prob_list, k=1)[0]