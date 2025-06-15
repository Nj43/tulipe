from typing import List, Dict
import torch
from transformers import PreTrainedTokenizerBase

class PairwiseDataCollator:
    """
    Collate a batch of examples that each have:
      - 'chosen_input_ids' (List[int])
      - 'chosen_attention_mask' (List[int])
      - 'rejected_input_ids' (List[int])
      - 'rejected_attention_mask' (List[int])
    into padded PyTorch tensors.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: bool = True,
        max_length: int = None,
        pad_to_multiple_of: int = None,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Separate chosen vs rejected
        chosen_batch = {
            "input_ids":      [f["chosen_input_ids"]      for f in features],
            "attention_mask": [f["chosen_attention_mask"] for f in features],
        }
        rejected_batch = {
            "input_ids":      [f["rejected_input_ids"]      for f in features],
            "attention_mask": [f["rejected_attention_mask"] for f in features],
        }

        # Use the tokenizer to pad each side
        c_padded = self.tokenizer.pad(
            chosen_batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        r_padded = self.tokenizer.pad(
            rejected_batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Return a single dict with both
        return {
            "input_ids_chosen":      c_padded["input_ids"],
            "attention_mask_chosen": c_padded["attention_mask"],
            "input_ids_rejected":     r_padded["input_ids"],
            "attention_mask_rejected":r_padded["attention_mask"],
        }
