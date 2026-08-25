"""Utility functions for dataset processing."""

import io
from typing import List, Tuple

import torch
import torch.distributed as dist
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tqdm import tqdm


def show_sample(
    input_ids,
    labels,
    tokenizer,
    title="Input and Labels",
    left_column="Input IDs",
    right_column="Labels"
):
    """Display a sample with input_ids and labels in a formatted table."""
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
    """Get the padding token id from tokenizer.

    If pad_token_id is not set, use eos_token_id as fallback.
    """
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id

    eos = tokenizer.eos_token_id
    return eos[0] if isinstance(eos, list) else eos


def calculate_matched_group(sequences: List[Tuple[int, int]], packing_length: int, is_finished: bool = True):
    """Bin-packing via First Fit Decreasing (https://arxiv.org/pdf/2404.10830).

    Args:
        sequences: List of (index, length) tuples.
        packing_length: Maximum length for each pack.
        is_finished: Whether this is the last batch.

    Returns:
        Tuple of (packed_sequences, remaining_sequences).
        packed_sequences is a list of lists, each containing (index, length) tuples.
    """
    if len(sequences) == 0:
        return [], []
    import binpacking

    # sequences 是 [(index, length), ...] 列表
    # weight_pos=1 表示长度在元组第二个位置
    # 将一组物品分配到多个容量固定的箱子（bins）中，使得每个箱子的总容量不超过指定的最大值。
    sequences = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)

    # sequences 是列表的列表，每个子列表包含多个 (index, length) 元组
    # 如果不是最后一批，保留最后一个不完整组用于下一批
    if sequences and not is_finished:
        sequences, ret_sequences = sequences[:-1], sequences[-1]
    else:
        ret_sequences = []
    return sequences, ret_sequences


def split_list(lst: list, n: int) -> List[list]:
    """Split a list into n sublists as evenly as possible.

    Args:
        lst: The list to split.
        n: Number of parts to split into.

    Returns:
        List of n sublists.
    """
    # 划分列表为n个子列表，对应n个子进程处理
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def is_master() -> bool:
    """Check if current process is the master process in distributed training."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def is_distributed() -> bool:
    """Check if running in distributed training mode."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
