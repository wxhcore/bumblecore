from threading import Thread
from typing import Optional, Generator, List,AsyncGenerator

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextIteratorStreamer,AsyncTextIteratorStreamer

from ..bumblebee import BumblebeeConfig, BumblebeeForCausalLM
AutoModelForCausalLM.register(BumblebeeConfig, BumblebeeForCausalLM)
AutoConfig.register("bumblebee", BumblebeeConfig)



class StreamTextGenerator:

    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or model.device
        self.vocab_size = model.config.vocab_size
        
        self.special_tokens = set()
        for token_id in [tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id]:
            if token_id is not None:
                self.special_tokens.add(token_id)

    def _apply_repetition_penalty(self, logits: torch.Tensor, generated: List[int], penalty: float) -> torch.Tensor:
        if penalty == 1.0:
            return logits
            
        for token_id in set(generated):
            if token_id in self.special_tokens or token_id >= self.vocab_size:
                continue
                
            if logits[0, token_id] < 0:
                logits[0, token_id] *= penalty
            else:
                logits[0, token_id] /= penalty
                
        return logits

    def _apply_top_k(
        self, 
        logits: torch.Tensor, 
        top_k: int,
        filter_value: float = -float('inf')
    ) -> torch.Tensor:
        if top_k is None or top_k <= 0:
            return logits
            
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)
        return logits

    def _apply_top_p(
        self, 
        logits: torch.Tensor, 
        top_p: float,
        min_tokens_to_keep: int = 1,
        filter_value: float = -float('inf')
    ) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
        return logits

    def _apply_sampling_filters(
        self, 
        logits: torch.Tensor, 
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None
    ) -> torch.Tensor:
        if not do_sample:
            return logits
            
        if top_p is not None and top_p < 1.0:
            logits = self._apply_top_p(logits, top_p)
        
        if top_k is not None and top_k > 0:
            logits = self._apply_top_k(logits, top_k)
            
        return logits
    

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: Optional[float],
        do_sample: Optional[bool],
    ) -> Generator[int, None, None]:

        if input_ids.shape[0] != 1:
            raise ValueError("仅支持 batch_size=1")
            
        generated = input_ids[0].tolist()
        past_key_values = None
        rng = None
        
        if temperature <= 0.0:
            temperature = 0.01

        outputs = self.model(
            input_ids=input_ids.to(self.device), 
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        generated_count = 0
        
        for step in range(max_new_tokens):
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)

            logits = self._apply_sampling_filters(logits, top_k, top_p, do_sample)

            if do_sample:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            yield token_id
            
            generated.append(token_id)
            generated_count += 1

            if token_id == self.tokenizer.eos_token_id or token_id == self.tokenizer.pad_token_id:
                break

            outputs = self.model(
                input_ids=next_token.to(self.device),
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values


    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: Optional[float],
        do_sample: Optional[bool],
    ) -> str:
        prompt_tokens = input_ids[0].tolist()
        completion_tokens = []
        
        for token_id in self.generate_stream(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        ):
            completion_tokens.append(token_id)
            
        return dict(
            response=self.tokenizer.decode(completion_tokens, skip_special_tokens=True),
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(completion_tokens),
            total_tokens=len(prompt_tokens) + len(completion_tokens),
        )


class BumblebeeChat:

    def __init__(
        self, 
        model_path: str, 
        device_map: str, 
        dtype: str | torch.dtype,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.generation_config = self.model.generation_config
        self.generator = StreamTextGenerator(self.model, self.tokenizer)

    def _build_prompt(self, messages: List[dict], system_prompt: Optional[str]) -> str:
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages  
        return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    

    def _prepare_generation_inputs(
        self,
        messages: List[dict] | str,
        max_new_tokens: Optional[int],
        system_prompt: Optional[str],
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: Optional[float],
        do_sample: Optional[bool],
    ):
        system_prompt = system_prompt if system_prompt is not None else "You are a helpful assistant."
        max_new_tokens = max_new_tokens if max_new_tokens is not None else 1024
        temperature = temperature if temperature is not None else self.generation_config.temperature
        top_k = top_k if top_k is not None else self.generation_config.top_k
        top_p = top_p if top_p is not None else self.generation_config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.generation_config.repetition_penalty
        do_sample = do_sample if do_sample is not None else self.generation_config.do_sample

        prompt_text = self._build_prompt(messages, system_prompt) if isinstance(messages, list) else messages
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        
        return {
            "inputs": inputs,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
        }

    def chat(
        self,
        messages: List[dict]|str,
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> str:
        
        prep = self._prepare_generation_inputs(
            messages = messages,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        return self.generator.generate(
            prep["inputs"].input_ids.to(self.model.device),
            max_new_tokens=prep["max_new_tokens"],
            temperature=prep["temperature"],
            top_k=prep["top_k"],
            top_p=prep["top_p"],
            repetition_penalty=prep["repetition_penalty"],
            do_sample=prep["do_sample"],
        )
    
    def _is_chinese_char(self, cp: int) -> bool:
        """Check whether CP is the codepoint of a CJK character."""
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3400 and cp <= 0x4DBF) or
            (cp >= 0x20000 and cp <= 0x2A6DF) or
            (cp >= 0x2A700 and cp <= 0x2B73F) or
            (cp >= 0x2B740 and cp <= 0x2B81F) or
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or
            (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):
            return True
        return False

    def stream_chat(
        self,
        messages: List[dict] | str,
        max_new_tokens: Optional[int],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> Generator[str, None, None]:
        prep = self._prepare_generation_inputs(
            messages=messages,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        
        buffer: List[int] = []
        last_decoded = ""

        for token_id in self.generator.generate_stream(
            prep["inputs"].input_ids.to(self.model.device),
            max_new_tokens=prep["max_new_tokens"],
            temperature=prep["temperature"],
            top_k=prep["top_k"],
            top_p=prep["top_p"],
            repetition_penalty=prep["repetition_penalty"],
            do_sample=prep["do_sample"],
        ):
            buffer.append(token_id)
            current_decoded = self.tokenizer.decode(buffer, skip_special_tokens=True)
            new_text = current_decoded[len(last_decoded):]

            if not new_text:
                continue

            should_yield = False
            printable_text = ""

            # 情况1: 遇到换行符 → 立即 flush
            if current_decoded.endswith("\n"):
                printable_text = new_text
                should_yield = True
                last_decoded = current_decoded

            # 情况2: 最后一个字符是中文（CJK）→ 立即输出
            elif len(current_decoded) > 0 and self._is_chinese_char(ord(current_decoded[-1])):
                printable_text = new_text
                should_yield = True
                last_decoded = current_decoded

            # 情况3: 其他语言（如英文）→ 只输出到最后一个空格
            else:
                complete_part = current_decoded[: current_decoded.rfind(" ") + 1]
                if len(complete_part) > len(last_decoded):
                    printable_text = complete_part[len(last_decoded):]
                    should_yield = True
                    last_decoded = complete_part

            if should_yield and printable_text:
                yield printable_text

        if buffer:
            final_decoded = self.tokenizer.decode(buffer, skip_special_tokens=True)
            remaining = final_decoded[len(last_decoded):]
            if remaining:
                yield remaining


class HFStreamChat:
    def __init__(self, model_path: str, device_map: str, dtype: str | torch.dtype):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generation_config = self.model.generation_config

    def _build_prompt(self, messages: list[dict], system_prompt: str | None) -> str:
        if messages and messages[0]["role"] != "system":
            messages = [{"role": "system", "content": system_prompt or "You are a helpful assistant."}] + messages
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def stream_chat(
        self,
        messages: List[dict] | str,
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> Generator[str, None, None]:

        prompt = self._build_prompt(messages, system_prompt) if isinstance(messages, list) else messages
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else 1024,
            temperature = temperature if temperature is not None else self.generation_config.temperature,
            top_k = top_k if top_k is not None else self.generation_config.top_k,
            top_p = top_p if top_p is not None else self.generation_config.top_p,
            repetition_penalty = repetition_penalty if repetition_penalty is not None else self.generation_config.repetition_penalty,
            do_sample = do_sample if do_sample is not None else self.generation_config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(**gen_kwargs, streamer=streamer)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()
    
    async def async_stream_chat(
        self,
        messages: List[dict] | str,
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> AsyncGenerator[str, None]:

        prompt = self._build_prompt(messages, system_prompt) if isinstance(messages, list) else messages
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else 1024,
            temperature = temperature if temperature is not None else self.generation_config.temperature,
            top_k = top_k if top_k is not None else self.generation_config.top_k,
            top_p = top_p if top_p is not None else self.generation_config.top_p,
            repetition_penalty = repetition_penalty if repetition_penalty is not None else self.generation_config.repetition_penalty,
            do_sample = do_sample if do_sample is not None else self.generation_config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )

        streamer = AsyncTextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(**gen_kwargs, streamer=streamer)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        async for new_text in streamer:
            yield new_text