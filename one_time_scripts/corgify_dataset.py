import sys
from pathlib import Path
parent_directory = Path(__file__).parent.parent
sys.path.append(str(parent_directory))


from data.iterable_corgi_dataset import IterableCorgiDataset
import torch


def corgify(baseline_root: Path, output_dir: Path, output_file_size: int, files_per_block: int):
    print("starting")
    output_dir.mkdir(exist_ok=True, parents=True)
    files = [f for f in baseline_root.rglob("*.txt")]
    block_corgi_dataset = IterableCorgiDataset(files, files_per_block, lines_per_file=2000, output_blocks=True, output_block_size=output_file_size, local=True)
    dataloader = torch.utils.data.DataLoader(
        block_corgi_dataset,
        batch_size=1,
        num_workers=0,
        prefetch_factor=None
        # persistent_workers=True
    )
    
    print("initialized dataloaders")
    
    for i, output_file_lines in enumerate(dataloader):
        output_file_lines = [f[0] + "\n" for f in output_file_lines]
        with open(output_dir / f"{i}.txt", "w") as handler:
            handler.writelines(output_file_lines)
        if i % 250 == 0:
            print(f"created {i} files")


# corgify(
#     baseline_root=Path("/homes/etayl/code/bert/local_dataset/baseline"),
#     output_dir=Path("/homes/etayl/code/bert/local_dataset/upload_cache/corgi_b_1000_n_1350"),
#     output_file_size=1000,
#     files_per_block=1350
# )

corgify(
    baseline_root=Path("/homes/etayl/code/bert/local_dataset/baseline_2000"),
    output_dir=Path("/homes/etayl/code/bert/local_dataset/upload_cache/corgi_b_2000_n_170"),
    output_file_size=2000,
    files_per_block=170
)