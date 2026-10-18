"""
Polymer SMILES Tokenizer for Sequence-to-Sequence / VAE Modeling.
Encodes polymer strings containing wildcards (*) into integer indices and decodes them back.
"""

import torch

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

class PolymerTokenizer:
    def __init__(self, vocabulary=None):
        if vocabulary is None:
            # Default canonical tokens observed in polymer datasets
            self.chars = sorted([
                '#', '(', ')', '*', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '=', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', ']', 'c', 'i', 'l',
                'n', 'o', 'r', 's'
            ])
        else:
            self.chars = sorted(list(set(vocabulary)))

        self.vocab = SPECIAL_TOKENS + self.chars
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}

        self.pad_idx = self.char_to_idx["<pad>"]
        self.sos_idx = self.char_to_idx["<sos>"]
        self.eos_idx = self.char_to_idx["<eos>"]
        self.unk_idx = self.char_to_idx["<unk>"]
        self.vocab_size = len(self.vocab)

    def encode(self, smiles: str, max_len: int = 120) -> list:
        """Converts a SMILES string into a padded integer token list with <sos> and <eos>."""
        tokens = [self.sos_idx]
        for c in smiles:
            tokens.append(self.char_to_idx.get(c, self.unk_idx))
        tokens.append(self.eos_idx)

        # Pad or truncate
        if len(tokens) < max_len:
            tokens = tokens + [self.pad_idx] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len - 1] + [self.eos_idx]
        return tokens

    def decode(self, token_indices) -> str:
        """Converts an integer token array/list back into a clean SMILES string."""
        chars = []
        for idx in token_indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx == self.eos_idx or idx == self.pad_idx:
                break
            if idx != self.sos_idx and idx != self.unk_idx:
                chars.append(self.idx_to_char.get(idx, ""))
        return "".join(chars)
