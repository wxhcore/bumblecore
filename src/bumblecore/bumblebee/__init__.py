from transformers import AutoModelForCausalLM,AutoConfig

from .modeling_bumblebee import BumblebeeForCausalLM,BumblebeeConfig
AutoModelForCausalLM.register(BumblebeeConfig, BumblebeeForCausalLM)
AutoConfig.register("bumblebee", BumblebeeConfig)


__all__ = ["BumblebeeForCausalLM","BumblebeeConfig"]
