import json
import warnings


DEFAULT_SYSTEM = "You are Bumblebee, a helpful AI assistant."

_ALLOWED_OPENAI_ROLES = {"system", "user", "assistant", "tool"}


class DataFormatter:

    def __init__(self, training_stage: str):
        self.training_stage = training_stage
        self.default_system = DEFAULT_SYSTEM

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

    def _ensure_system_message(self, conversations):
        if conversations and conversations[0].get("from") == "system":
            return conversations
        return [{"from": "system", "value": self.default_system}, *conversations]

    def _ensure_system_message_openai(self, messages):
        if messages and messages[0].get("role") == "system":
            return messages
        return [{"role": "system", "content": self.default_system}, *messages]

    def _validate_openai_messages(self, messages):
        if not messages:
            raise ValueError("'messages' is empty")
        for m in messages:
            role = m.get("role")
            if role not in _ALLOWED_OPENAI_ROLES:
                raise ValueError(f"Unsupported message role: {role!r}")
            if role == "assistant":
                if not m.get("content") and not m.get("tool_calls"):
                    raise ValueError("'assistant' message must have 'content' or 'tool_calls'")
            else:
                if "content" not in m:
                    raise ValueError(f"{role!r} message must have a 'content' field")

    def _parse_tools(self, tools):
        if not tools:
            return None
        if isinstance(tools, str):
            tools = json.loads(tools)
        return tools or None

    def _convert_sharegpt_messages(self, conversations):
        messages: list[dict] = []

        for turn in conversations:
            role = turn.get("from")
            value = turn.get("value")

            if role == "system":
                messages.append({"role": "system", "content": value})

            elif role == "human":
                messages.append({"role": "user", "content": value})

            elif role == "gpt":
                messages.append({"role": "assistant", "content": value})

            elif role == "function_call":
                call = json.loads(value) if isinstance(value, str) else value
                args = call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": args,
                            },
                        }
                    ],
                })

            elif role == "observation":
                if not messages:
                    raise ValueError("'observation' appears at the start of conversation")
                prev = messages[-1]
                prev_is_tool_call = (
                    prev["role"] == "assistant" and prev.get("tool_calls")
                )
                prev_is_tool = prev["role"] == "tool"
                if not (prev_is_tool_call or prev_is_tool):
                    raise ValueError(
                        "'observation' is not preceded by a 'function_call' or another 'observation'"
                    )
                messages.append({"role": "tool", "content": value})

            else:
                raise ValueError(f"Unsupported conversation role: {role!r}")

        return messages

    def _handle_sft_data(self, dataset):
        if dataset[0].get("messages"):
            return self.build_sft_messages_samples(dataset)
        elif dataset[0].get("conversations"):
            return self.build_sft_sharegpt_samples(dataset)
        elif dataset[0].get("instruction"):
            dataset = self._set_alpaca_system(dataset)
            return self.build_sft_alpaca_samples(dataset)
        else:
            raise ValueError("SFT data format is invalid")

    def _handle_dpo_data(self, dataset):
        if dataset[0].get("messages"):
            return self.build_dpo_messages_samples(dataset)
        elif dataset[0].get("conversations"):
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

            if input_text:
                instruction += "\n" + input_text

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]

            samples.append({"messages": messages, "tools": None})

        return samples

    def build_sft_sharegpt_samples(self, dataset):
        samples: list[dict] = []

        for idx, item in enumerate(dataset):
            conversations = self._ensure_system_message(item["conversations"])

            try:
                messages = self._convert_sharegpt_messages(conversations)
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                warnings.warn(f"Skipping malformed ShareGPT sample at idx={idx}: {e}")
                continue

            tools = self._parse_tools(item.get("tools"))

            samples.append({"messages": messages, "tools": tools})

        return samples

    def build_sft_messages_samples(self, dataset):
        samples: list[dict] = []

        for idx, item in enumerate(dataset):
            messages = self._ensure_system_message_openai(item["messages"])

            try:
                self._validate_openai_messages(messages)
            except (ValueError, KeyError) as e:
                warnings.warn(f"Skipping malformed messages sample at idx={idx}: {e}")
                continue

            tools = self._parse_tools(item.get("tools"))

            samples.append({"messages": list(messages), "tools": tools})

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

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
            ]

            chosen_messages = messages + [{"role": "assistant", "content": chosen}]
            rejected_messages = messages + [{"role": "assistant", "content": rejected}]

            samples.append(
                {
                    "chosen_messages": {"messages": chosen_messages, "tools": None},
                    "rejected_messages": {"messages": rejected_messages, "tools": None},
                }
            )

        return samples

    def build_dpo_sharegpt_samples(self, dataset):
        samples: list[dict] = []

        for idx, item in enumerate(dataset):
            conversations = self._ensure_system_message(item["conversations"])

            try:
                messages = self._convert_sharegpt_messages(conversations)
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                warnings.warn(f"Skipping malformed ShareGPT DPO sample at idx={idx}: {e}")
                continue

            chosen = item.get("chosen")
            rejected = item.get("rejected")
            tools = self._parse_tools(item.get("tools"))

            chosen_messages = messages + [{"role": "assistant", "content": chosen["value"]}]
            rejected_messages = messages + [{"role": "assistant", "content": rejected["value"]}]

            samples.append(
                {
                    "chosen_messages": {"messages": chosen_messages, "tools": tools},
                    "rejected_messages": {"messages": rejected_messages, "tools": tools},
                }
            )

        return samples

    def build_dpo_messages_samples(self, dataset):
        samples: list[dict] = []

        def build_candidate_message(candidate):
            if isinstance(candidate, str):
                return {"role": "assistant", "content": candidate}

            message = {
                "role": "assistant",
                "content": (candidate or {}).get("content", ""),
            }
            if candidate and candidate.get("tool_calls"):
                message["tool_calls"] = candidate["tool_calls"]
            return message

        for idx, item in enumerate(dataset):
            messages = self._ensure_system_message_openai(item["messages"])

            try:
                self._validate_openai_messages(messages)
            except (ValueError, KeyError) as e:
                warnings.warn(f"Skipping malformed messages DPO sample at idx={idx}: {e}")
                continue

            chosen = item.get("chosen")
            rejected = item.get("rejected")
            tools = self._parse_tools(item.get("tools"))

            chosen_messages = list(messages) + [build_candidate_message(chosen)]
            rejected_messages = list(messages) + [build_candidate_message(rejected)]

            samples.append(
                {
                    "chosen_messages": {"messages": chosen_messages, "tools": tools},
                    "rejected_messages": {"messages": rejected_messages, "tools": tools},
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
