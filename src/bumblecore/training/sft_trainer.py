from transformers import AutoTokenizer

from .base_trainer import BaseTrainer
from ..data_processing import DataFormatter, SFTDataset, load_sft_data, DataCollator, PackingDataCollator
from ..config import TrainConfig

class SFTTrainer(BaseTrainer):
    def __init__(self, config: TrainConfig):
        self.config = config
        self.format_preprocess_fn = DataFormatter(self.config.training_stage)
        self.tokenizer, self.train_dataset = self._prepare_datasets()
        # Use PackingDataCollator when packing is enabled, otherwise use standard DataCollator
        if self.config.packing:
            self.data_collator = PackingDataCollator(self.tokenizer)
        else:
            self.data_collator = DataCollator(self.tokenizer)
        super().__init__(config, self.train_dataset, self.tokenizer, self.data_collator)
        self._print_train_parameters()

    def _get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            use_fast=self.config.tokenizer_use_fast,
            trust_remote_code=self.config.trust_remote_code,
        )

    def _prepare_datasets(self):
        dataset = load_sft_data(self.config.dataset_path)
        messages = self.format_preprocess_fn(dataset)

        tokenizer = self._get_tokenizer()
        train_dataset = SFTDataset(
            messages,
            tokenizer,
            max_length=self.config.cutoff_len,
            packing=self.config.packing,
            packing_num_proc=self.config.packing_num_proc,
        )
        return tokenizer, train_dataset
