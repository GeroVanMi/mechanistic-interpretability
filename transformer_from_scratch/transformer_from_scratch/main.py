import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import math
import sys
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import datasets
import einops
import numpy as np
import torch
import wandb
from ModelConfig import Config
from rich import print as rprint
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from Transformer.LayerNorm import LayerNorm
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

from tests.LayerNormTests import load_gpt2_test, rand_float_test

# Make sure exercises are in the path
chapter = r"chapter1_transformer_interp"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_transformer_from_scratch"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

# import part1_transformer_from_scratch.solutions as solutions
# from plotly_utils import imshow


MAIN = __name__ == "__main__"

if MAIN:
    config = Config()

    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small",
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,
    )

    reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
    tokens = reference_gpt2.to_tokens(reference_text).to(config.device)
    # print(tokens)
    # print(tokens.shape)
    # print(reference_gpt2.to_str_tokens(tokens))

    logits, cache = reference_gpt2.run_with_cache(tokens)
    probs = logits.softmax(dim=-1)
    print(probs.shape)

    most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(
        logits.argmax(dim=-1)[0]
    )

    print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))

    next_token = logits[0, -1].argmax(dim=-1)
    next_char = reference_gpt2.to_string(next_token)
    print(repr(next_char))

    print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")

    for i in range(10):
        print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
        # Define new input sequence, by appending the previously generated token
        tokens = torch.cat([tokens, next_token[None, None]], dim=-1)
        # Pass our new sequence through the model, to get new output
        logits = reference_gpt2(tokens)
        # Get the predicted token at the end of our sequence
        next_token = logits[0, -1].argmax(dim=-1)
        # Decode and print the result
        next_char = reference_gpt2.to_string(next_token)

    print(config)

    rand_float_test(LayerNorm, [2, 4, 768])
    load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])
