import io

import torch
import torch.distributed as dist
from torch.utils.data import Dataset,get_worker_info
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tqdm import tqdm


def show_sample(
    input_ids, 
    labels, 
    tokenizer, 
    title="Input and Labels" ,
    left_column = "Input IDs", 
    right_column = "Labels"
):
    input_ids = input_ids.tolist()
    labels = labels.tolist()

    valid_labels_list = [token_id for token_id in labels if token_id != -100]
    decoded_input = tokenizer.decode(input_ids)
    decoded_labels = tokenizer.decode(valid_labels_list)

    table = Table(show_header=True, show_lines=True, title=title)
    table.add_column(left_column, overflow="fold")
    table.add_column(right_column, overflow="fold")

    wrapped_input = Text(decoded_input, no_wrap=False, overflow="fold")
    wrapped_labels = Text(decoded_labels, no_wrap=False, overflow="fold")

    table.add_row(str(input_ids), str(labels))
    table.add_row(wrapped_input, wrapped_labels)

    with io.StringIO() as buf:
        console = Console(file=buf, force_terminal=False)
        console.print(table)
        output = buf.getvalue()

    tqdm.write(output.rstrip())


def get_padding_value(tokenizer):
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    
    eos = tokenizer.eos_token_id
    return eos[0] if isinstance(eos, list) else eos


class PretrainDataset(Dataset):
    
    def __init__(
        self,
        train_dataset,
        tokenizer,
        max_length,
    ):
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.has_shown_sample = False
    
    def __len__(self):
        return len(self.train_dataset)
    
    def create_pretraining_dataset(self, text):

        if self.tokenizer.eos_token_id is not None:
            text = text + self.tokenizer.eos_token
        elif self.tokenizer.pad_token_id is not None:
            text = text + self.tokenizer.pad_token
        elif self.tokenizer.bos_token_id is not None:
            text = text + self.tokenizer.bos_token

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )
        
        input_ids=encoding["input_ids"].squeeze(0)
        attention_mask=encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def _show_train_sample(self, input_ids, labels):

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0 
        worker_info = get_worker_info()
        is_main_worker = (worker_info is None) or (worker_info.id == 0)
        if rank == 0 and is_main_worker and not self.has_shown_sample:
            show_sample(
                input_ids=input_ids,
                labels=labels,
                tokenizer = self.tokenizer,
                title="Pretrain Input and Labels",
                left_column="Input IDs",
                right_column="Labels"
            )
            self.has_shown_sample = True

    def __getitem__(self, idx):
        text = self.train_dataset[idx]["text"]
        sample = self.create_pretraining_dataset(text)

        self._show_train_sample(
            input_ids=sample["input_ids"],
            labels=sample["labels"],
        )

        return sample



class SFTDataset(Dataset):
    
    def __init__(
        self,
        train_dataset,
        tokenizer,
        max_length,
    ):
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.has_shown_sample = False

    def __len__(self):
        return len(self.train_dataset)
    

    def create_conversation_manually(self, messages, tools):

        full = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            truncation=True,
            max_length=self.max_length,
            tools=tools if tools else None,
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        assistant_masks = [0] * len(input_ids)
        current_pos = 0

        for i, message in enumerate(messages):
            if message["role"] == "assistant":

                context_with_reply = messages[: i + 1]
                full_tokens = self.tokenizer.apply_chat_template(
                    context_with_reply,
                    tokenize=True,
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=self.max_length,
                    tools=tools if tools else None,
                )
                reply_end_pos = len(full_tokens)

                assistant_masks[current_pos:reply_end_pos] = [1] * (reply_end_pos - current_pos)

            else:

                prompt_context = messages[: i + 1]

                if message["role"] == "system":
                    continue

                else:
                    prompt_tokens = self.tokenizer.apply_chat_template(
                        prompt_context,
                        tokenize=True,
                        add_generation_prompt=True,
                        truncation=True,
                        max_length=self.max_length,
                        tools=tools if tools else None,
                    )
                    current_pos = len(prompt_tokens)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = input_ids.clone()

        labels[torch.tensor(assistant_masks, dtype=torch.bool) == 0] = -100

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    

    def _show_train_sample(self, input_ids, labels):

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0 
        worker_info = get_worker_info()
        is_main_worker = (worker_info is None) or (worker_info.id == 0)
        if rank == 0 and is_main_worker and not self.has_shown_sample:
            show_sample(
                input_ids=input_ids,
                labels=labels,
                tokenizer = self.tokenizer,
                title="SFT Input and Labels",
                left_column="Input IDs",
                right_column="Labels"
            )
            self.has_shown_sample = True
    
    def __getitem__(self, idx):
        messages = self.train_dataset[idx]["messages"]
        tools = self.train_dataset[idx]["tools"]
        sample = self.create_conversation_manually(messages, tools)

        self._show_train_sample(
            input_ids=sample["input_ids"],
            labels=sample["labels"],
        )

        return sample


class DPODataset(Dataset):
    def __init__(self, train_dataset, tokenizer, max_length):
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.has_shown_sample = False
    
    def __len__(self):
        return len(self.train_dataset)

    def create_dpo_dataset(self, messages, tools):
        full = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            truncation=True,
            max_length=self.max_length,
            tools=tools if tools else None,
        )

        prompt_messages = messages[:-1]
        prompt_input_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=self.max_length,
            tools=tools if tools else None,
        )
        input_ids = torch.tensor(full["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full["attention_mask"], dtype=torch.long)
        prompt_len = len(prompt_input_ids)
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    def _show_train_sample(
            self, 
            chosen_input_ids, 
            chosen_labels, 
            rejected_input_ids, 
            rejected_labels, 
        ):

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0 
        worker_info = get_worker_info()
        is_main_worker = (worker_info is None) or (worker_info.id == 0)
        if rank == 0 and is_main_worker and not self.has_shown_sample:
            show_sample(
                input_ids=chosen_input_ids,
                labels=chosen_labels,
                tokenizer = self.tokenizer,
                title="DPO Chosen Input IDs and Labels",
                left_column="Chosen Input IDs",
                right_column="Chosen Labels"
            )
            show_sample(
                input_ids=rejected_input_ids,
                labels=rejected_labels,
                tokenizer = self.tokenizer,
                title="DPO Rejected Input IDs and Labels",
                left_column="Rejected Input IDs",
                right_column="Rejected Labels"
            )
            self.has_shown_sample = True
    
    def __getitem__(self, idx):
        chosen_messages = self.train_dataset[idx]["chosen_messages"]
        rejected_messages = self.train_dataset[idx]["rejected_messages"]

        chosen_data = self.create_dpo_dataset(chosen_messages["messages"], chosen_messages["tools"])
        rejected_data = self.create_dpo_dataset(rejected_messages["messages"], rejected_messages["tools"])

        self._show_train_sample(
            chosen_input_ids=chosen_data["input_ids"],
            chosen_labels=chosen_data["labels"],
            rejected_input_ids=rejected_data["input_ids"],
            rejected_labels=rejected_data["labels"],
        )

        return dict(
            chosen_input_ids=chosen_data["input_ids"],
            chosen_attention_mask=chosen_data["attention_mask"],
            chosen_labels=chosen_data["labels"],
            rejected_input_ids=rejected_data["input_ids"],
            rejected_attention_mask=rejected_data["attention_mask"],
            rejected_labels=rejected_data["labels"],
        )


class DataCollator:
    def __init__(self, tokenizer):
        self.input_ids_padding_value = get_padding_value(tokenizer=tokenizer)

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids = self.right_pad_sequence(
            input_ids, padding_value=self.input_ids_padding_value
        )
        attention_mask = self.right_pad_sequence(attention_mask, padding_value=0)
        labels = self.right_pad_sequence(labels, padding_value=-100)

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    @staticmethod
    def right_pad_sequence(sequences, padding_value):
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=padding_value
        )
        return padded


class DPOCollator:
    def __init__(self, tokenizer):
        self.input_ids_padding_value = get_padding_value(tokenizer=tokenizer)

    def __call__(self, batch):
        chosen_input_ids = [item["chosen_input_ids"] for item in batch]
        chosen_attention_mask = [item["chosen_attention_mask"] for item in batch]
        chosen_labels = [item["chosen_labels"] for item in batch]

        rejected_input_ids = [item["rejected_input_ids"] for item in batch]
        rejected_attention_mask = [item["rejected_attention_mask"] for item in batch]
        rejected_labels = [item["rejected_labels"] for item in batch]

        all_lengths = [
            len(x) for x in chosen_input_ids + rejected_input_ids
        ]
        max_length = max(all_lengths)

        return dict(
            chosen_input_ids=self._right_pad_to_len(
            chosen_input_ids, max_length, self.input_ids_padding_value
        ),
            chosen_attention_mask=self._right_pad_to_len(
            chosen_attention_mask, max_length, 0
        ),
            chosen_labels=self._right_pad_to_len(
            chosen_labels, max_length, -100
        ),
            rejected_input_ids=self._right_pad_to_len(
            rejected_input_ids, max_length, self.input_ids_padding_value
        ),
            rejected_attention_mask=self._right_pad_to_len(
            rejected_attention_mask, max_length, 0
        ),
            rejected_labels=self._right_pad_to_len(
            rejected_labels, max_length, -100
        ),
        )

    @staticmethod
    def _right_pad_to_len(sequences, max_length, padding_value):
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=padding_value
        )
        if padded.size(1) < max_length:
            diff = max_length - padded.size(1)
            pad_tensor = torch.full(
                (padded.size(0), diff),
                padding_value,
                dtype=padded.dtype,
                device=padded.device
            )
            padded = torch.cat([padded, pad_tensor], dim=1)
        return padded