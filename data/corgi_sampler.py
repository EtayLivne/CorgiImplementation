from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info
import numpy as np

class CorgiSampler(Sampler):
    def __init__(self, dataset: Dataset, n: int=0, b: int=0, num_corgis: int=1, **kwargs):
        super().__init__(dataset, **kwargs)
        self.n = n
        self.b = b
        self.num_files = int(len(self.dataset) / self.b)
        self.num_shards = int(self.num_files / self.n)
        self.index_order = np.arange(len(self.dataset))
        for _ in range(num_corgis):
            self.index_order = self._corgi_shuffle(self.index_order)
        
    def _corgi_shuffle(self, array_to_shuffle: np.ndarray):
        rng = np.random.default_rng()
        
        # Each row has the indexes of all examples in one file
        files = array_to_shuffle.reshape((self.num_files, self.b))
        
        # Allocate n files to each worker by shuffling the files and slicing in order 
        shuffled_files = rng.shuffle(files)        
        shards = np.array([
            shuffled_files[i:i+self.n].flatten()
            for i in range(0, self.num_shards, self.n)
        ])
        
        # Shuffle the elements of each shard
        rng.shuffle(shards, axis=1)
        
        # return the post shuffle index order
        return shards.flatten()
        
    def __iter__(self):
        return iter(self.index_order)
    
    
# class NoShuffleDistributedSampler(Sampler):
#     def __init__(self, data) -> None:
#         self.data = data
        
#     def __len__(self) -> int:
#         return len(self.data)
    
#     def __iter__(self) -> Iterator[int]:
#         sizes = torch.tensor([len(x) for x in self.data])
#         yield from torch.argsort(sizes).tolist()