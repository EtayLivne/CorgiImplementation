import torch
import lightning.pytorch as pl
from transformers import BertConfig, BertTokenizer, BertForMaskedLM


# REPLACE WITH GPT2 שמאל

class BertForCustomPretraining(pl.LightningModule):
    def __init__(self, config: BertConfig):
        super(BertForCustomPretraining, self).__init__()
        self.config = config
        self.bert = BertForMaskedLM(config)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(outputs.logits.view(-1, self.config.vocab_size), labels.view(-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer



