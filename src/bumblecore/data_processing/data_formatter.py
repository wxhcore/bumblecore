import json

class DataFormatter:

    def __init__(self, training_stage: str):
        self.training_stage = training_stage
        self.default_system = "You are a helpful assistant."
    
    def _handle_pretrain_data(self, dataset):
        if dataset[0].get("text"):
            return dataset
        else:
            raise ValueError("Pretrain data format is invalid")
    
    def _set_alpaca_system(self, dataset):
        for record in dataset:
            if not record.get("system"):
                record["system"] = self.default_system
        return dataset
    
    def _set_sharegpt_system(self, dataset):
        for record in dataset:
            conversations = record["conversations"]
            if  conversations[0].get("from") != "system":
                conversations.insert(0, {"from": "system", "value": self.default_system})
        return dataset

    def _handle_sft_data(self, dataset):

        if dataset[0].get("conversations"):
            dataset = self._set_sharegpt_system(dataset)
            return self.build_sft_sharegpt_samples(dataset)
        elif dataset[0].get("instruction"):
            dataset = self._set_alpaca_system(dataset)
            return self.build_sft_alpaca_samples(dataset)
        else:
            raise ValueError("SFT data format is invalid")
    
    def _handle_dpo_data(self, dataset):

        if dataset[0].get("conversations"):
            dataset = self._set_sharegpt_system(dataset)
            return self.build_dpo_sharegpt_samples(dataset)
        elif dataset[0].get("instruction"):
            dataset = self._set_alpaca_system(dataset)
            return self.build_dpo_alpaca_samples(dataset)
        else:
            raise ValueError("DPO data format is invalid")


    def build_sft_alpaca_samples(self, dataset):

        samples: list[dict] = []

        for item in dataset:
            system = item.get("system")
            instruction = item.get("instruction")
            input_text = item.get("input") or None
            output = item.get("output")
            tools = item.get("tools") or None
            if isinstance(tools, str):
                tools = json.loads(tools) or None

            if input_text:
                instruction += "\n" + input_text

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]

            samples.append({"messages": messages, "tools": tools})

        return samples

    def build_sft_sharegpt_samples(self, dataset):

        samples: list[dict] = []

        for item in dataset:
            turns = item["conversations"]

            assert len(turns) % 2 != 0

            system = turns[0]["value"] 
            turns.pop(0)

            tools = item.get("tools") or None
            if isinstance(tools, str):
                tools = json.loads(tools) or None

            messages = [{"role": "system", "content": system}]

            for i in range(0, len(turns), 2):

                human_turn = turns[i]
                gpt_turn = turns[i + 1]

                messages.append({"role": "user", "content": human_turn["value"]})
                messages.append({"role": "assistant", "content": gpt_turn["value"]})

            samples.append({"messages": messages, "tools": tools})

        return samples

    def build_dpo_alpaca_samples(self, dataset):
        samples: list[dict] = []

        for item in dataset:
            system = item.get("system")
            instruction = item.get("instruction")
            input_text = item.get("input") or None
            if input_text:
                instruction += "\n" + input_text
            chosen = item.get("chosen")
            rejected = item.get("rejected")
            tools = item.get("tools") or None
            if isinstance(tools, str):
                tools = json.loads(tools) or None
            
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
            ]

            chosen_messages = messages + [{"role": "assistant", "content": chosen}]
            rejected_messages = messages + [{"role": "assistant", "content": rejected}]

            samples.append(
                {
                    "chosen_messages":{
                        "messages":chosen_messages, 
                        "tools": tools
                    },
                    "rejected_messages":{
                        "messages":rejected_messages, 
                        "tools": tools
                    }
                }
            )

        return samples

    def build_dpo_sharegpt_samples(self, dataset):
        samples: list[dict] = []

        for item in dataset:
            turns = item["conversations"]

            assert len(turns) % 2 == 0
            system = turns[0]["value"] 
            turns.pop(0)

            chosen = item.get("chosen")
            rejected = item.get("rejected")
            tools = item.get("tools") or None
            if isinstance(tools, str):
                tools = json.loads(tools) or None

            messages = [{"role": "system", "content": system}]

            for idx, conversation in enumerate(turns):

                if idx % 2 == 0:
                    messages.append({"role": "user", "content": conversation["value"]})
                else:
                    messages.append({"role": "assistant", "content": conversation["value"]})

            chosen_messages = messages + [{"role": "assistant", "content": chosen["value"]}]
            rejected_messages = messages + [{"role": "assistant", "content": rejected["value"]}]

            samples.append(
                {
                    "chosen_messages":{
                        "messages":chosen_messages, 
                        "tools": tools
                    },
                    "rejected_messages":{
                        "messages":rejected_messages, 
                        "tools": tools
                    }
                }
            )
            
        return samples

    def __call__(self, dataset):

        if self.training_stage in ["pretrain", "continue_pretrain"]:
            return self._handle_pretrain_data(dataset)
        elif self.training_stage in ["sft"]:
            return self._handle_sft_data(dataset)
        elif self.training_stage in ["dpo"]:
            return self._handle_dpo_data(dataset)
        else:
            raise ValueError(f"Unsupported training stage: {self.training_stage}")