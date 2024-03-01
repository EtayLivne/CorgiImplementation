import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from  tokenizers import Tokenizer
import lightning.pytorch as pl
import torch.optim.lr_scheduler as lr_sched
import time



def get_warmup_sch(warmup_epochs):

    def warmup_factor(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        return 1.0

    return warmup_factor


class GPT2Dataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length, return_tensors="pt")

class GPT2ForCustomPretraining(pl.LightningModule):
    def __init__(self, gpt_model: torch.nn.Module, learning_rate: float=1e-3):
        super().__init__()
        self.model = gpt_model
        self.learning_rate = learning_rate
        self._num_validations = 0
        self._validation_start_time = 0
        self._validation_end_time = 0

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs = {k: v.squeeze(1) for k, v in batch.items()}
        outputs = self(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        # self.log("train_loss", loss, sync_dist=True)
        self.log("boat_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs = {k: v.squeeze(1) for k, v in batch.items()}
        outputs = self(**inputs, labels=inputs["input_ids"])
        val_loss = outputs.loss
        self.log("val_loss", val_loss, sync_dist=True)
        my_id = self._get_my_worker_id()
        
        with open(f"val_loss_{my_id}.log", "a") as handler:
            handler.write(f"{batch_idx}: {val_loss}\n")
             
        return val_loss

    def configure_optimizers(self):
        warmup_epochs = 5
        parameters = self.model.parameters()
        optimizer = torch.optim.AdamW(parameters,
                                        lr=self.learning_rate,
                                        weight_decay=1e-4)

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                             lr_lambda=get_warmup_sch(
                                                                 warmup_epochs))
        # lr_scheduler = lr_sched.OneCycleLR(optimizer, max_lr=0.01, total_steps=int(10**9))
        lr_scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max= 15 * int(10**4))

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_epochs])

        return [optimizer], {"scheduler": scheduler,
                             "frequency": "step"}
    
    def on_train_start(self) -> None:
        my_id = self._get_my_worker_id()
        
        print(f"worker {my_id}: STARTING TRAIN")
        with open("etay.txt", "w") as handler:
            handler.write("etay")
    
    def on_validation_start(self) -> None:
        self._num_validations += 1
        print(f"VALIDATION {self._num_validations} STARTED")
        self._validation_start_time = time.time()
    
    
    def on_validation_end(self) -> None:
        self._validation_end_time = time.time()
        total_time = self._validation_end_time - self._validation_start_time 
        print(f"VALIDATION {self._num_validations} ENDED, TOOK {total_time}S")


    def _get_my_worker_id(self):
        worker_info = get_worker_info()
        return worker_info.id if worker_info is not None else "none"
    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset, batch_size=4, shuffle=True)