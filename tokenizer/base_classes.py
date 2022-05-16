from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tokenizer import Tokenizer


class UsesTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def char(self):
        return self.tokenizer.char

    @property
    def state(self):
        return self.tokenizer.state
