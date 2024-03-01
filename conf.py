from pydantic import BaseModel

class TrainConfig(BaseModel):
    corgi: bool
    files_json: str
    batch_size: int
    num_train_workers: int
    num_val_workers: int
    prefetch_factor: int
    max_steps: int 
    train_steps_between_vals: int
    val_steps: int
    files_per_block_train: int=None
    lines_per_file: int=None
    




##### double corgi



double_corgi_1_precent_b_1000_n_1350_conf = TrainConfig(
    corgi=True,
    files_json="/homes/etayl/code/bert/single_corgi_categories.json",
    batch_size=128,
    num_train_workers=12,
    num_val_workers=2,
    prefetch_factor=2,
    max_steps=15 * int(10**4),
    train_steps_between_vals=int(10**3),
    val_steps=int(50),
    files_per_block_train=1350,
    lines_per_file=1000
)

double_corgi_quarter_precent_b_1000_n_340_conf = TrainConfig(
    corgi=True,
    files_json="/homes/etayl/code/bert/single_corgi_categories.json",
    batch_size=128,
    num_train_workers=12,
    num_val_workers=2,
    prefetch_factor=2,
    max_steps=15 * int(10**4),
    train_steps_between_vals=int(10**3),
    val_steps=int(50),
    files_per_block_train=340,
    lines_per_file=1000
)

double_corgi_1_precent_b_2000_n_675_conf = TrainConfig(
    corgi=True,
    files_json="/homes/etayl/code/bert/files_corgi_b_2000_n_675.json",
    batch_size=128,
    num_train_workers=12,
    num_val_workers=2,
    prefetch_factor=2,
    max_steps=15 * int(10**4),
    train_steps_between_vals=int(10**3),
    val_steps=int(50),
    files_per_block_train=675,
    lines_per_file=2000
)

double_corgi_quarter_precent_b_2000_n_170_conf = TrainConfig(
    corgi=True,
    files_json="/homes/etayl/code/bert/files_corgi_b_2000_n_170.json",
    batch_size=128,
    num_train_workers=12,
    num_val_workers=2,
    prefetch_factor=2,
    max_steps=15 * int(10**4),
    train_steps_between_vals=int(10**3),
    val_steps=int(50),
    files_per_block_train=170,
    lines_per_file=2000
)


##### single corgi

corgi_1_precent_b_1000_n_1350_conf = TrainConfig(
    corgi=True,
    files_json="/homes/etayl/code/bert/file_categories.json",
    batch_size=128,
    num_train_workers=12,
    num_val_workers=2,
    prefetch_factor=2,
    max_steps=15 * int(10**4),
    train_steps_between_vals=int(10**3),
    val_steps=int(50),
    files_per_block_train=1350,
    lines_per_file=1000
)

corgi_quarter_precent_b_1000_n_340_conf = TrainConfig(
    corgi=True,
    files_json="/homes/etayl/code/bert/file_categories.json",
    batch_size=128,
    num_train_workers=12,
    num_val_workers=2,
    prefetch_factor=2,
    max_steps=15 * int(10**4),
    train_steps_between_vals=int(10**3),
    val_steps=int(50),
    files_per_block_train=340,
    lines_per_file=1000
)


corgi_1_precent_b_2000_n_675_conf = TrainConfig(
    corgi=True,
    files_json="/homes/etayl/code/bert/file_categories_b_2000.json",
    batch_size=128,
    num_train_workers=12,
    num_val_workers=2,
    prefetch_factor=2,
    max_steps=15 * int(10**4),
    train_steps_between_vals=int(10**3),
    val_steps=int(50),
    files_per_block_train=675,
    lines_per_file=2000
)

corgi_quarter_precent_b_2000_n_170_conf = TrainConfig(
    corgi=True,
    files_json="/homes/etayl/code/bert/file_categories_b_2000.json",
    batch_size=128,
    num_train_workers=12,
    num_val_workers=2,
    prefetch_factor=2,
    max_steps=15 * int(10**4),
    train_steps_between_vals=int(10**3),
    val_steps=int(50),
    files_per_block_train=170,
    lines_per_file=2000
)





##### full shuffle

full_shuffle_conf = TrainConfig(
    corgi=False,
    files_json="/homes/etayl/code/bert/file_categories.json",
    batch_size=128,
    num_train_workers=12,
    num_val_workers=2,
    prefetch_factor=2,
    max_steps=15 * int(10**4),
    train_steps_between_vals=int(10**3),
    val_steps=int(50),
    files_per_block_train=340,
    lines_per_file=1000
)





##### local test

local_test_conf = TrainConfig(
    corgi=True,
    files_json="/homes/etayl/code/bert/file_categories.json",
    batch_size=8,
    num_train_workers=2,
    num_val_workers=1,
    prefetch_factor=1,
    max_steps=100,
    train_steps_between_vals=10,
    val_steps=3,
    files_per_block_train=5,
    files_per_block_val=5,
    lines_per_file=1000
)