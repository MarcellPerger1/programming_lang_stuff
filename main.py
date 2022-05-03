# TODO: fix git clone not working!!!
# TODO: files!!

from __future__ import annotations

import string
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Type

NAME_CHARS = string.ascii_letters + string.digits + '_'
VALID_CHARS = NAME_CHARS + ''


class EndToken(Exception):
    pass


class Tokens(Enum):
    INT = auto()
    NAME = auto()  # todo rename to IDENTIFIER
    WHITESPACE = auto()


class State(Enum):
    NONE = auto()

    @classmethod
    def is_none(cls, state):
        return state is None or state == cls.NONE


class UsesTokenizer:
    def __init__(self, tokenizer: Tokenizer2):
        self.tokenizer = tokenizer

    @property
    def char(self):
        return self.tokenizer.char


@dataclass
class Token2(UsesTokenizer):
    tokenizer: Tokenizer2 = field(repr=False)
    text: str = ''
    type: TokenType = None

    def set_type(self, t: TokenType):
        self.type = t

    def set_class(self, c: Type[TokenType]):
        self.set_type(c(self.tokenizer, self))

    def accept(self):
        self.text += self.char


class TokenType(UsesTokenizer, ABC):
    def __init__(self, tokenizer: Tokenizer2, token: Token2):
        super().__init__(tokenizer)
        self.token = token

    def __repr__(self):
        return f'<{type(self).__qualname__}>'

    # region props
    @property
    def text(self):
        return self.token.text
    # endregion

    def accept(self):
        self.token.accept()

    # region start
    @abstractmethod
    def do_start(self) -> bool:
        return False

    def start(self):
        return self.do_start()

    # endregion

    # region continue
    @abstractmethod
    def do_cont(self) -> bool:
        return False

    def cont(self) -> bool:
        return self.do_cont()
    # endregion


class IntToken(TokenType):
    def do_cont(self) -> bool:
        return self.char in string.digits

    def do_start(self) -> bool:
        return self.char in string.digits


class NameToken(TokenType):
    def do_start(self) -> bool:
        return self.char in NAME_CHARS

    def do_cont(self) -> bool:
        return self.char in NAME_CHARS


class WhitespaceToken(TokenType):
    def do_start(self) -> bool:
        return self.char.isspace()

    def do_cont(self) -> bool:
        return self.char.isspace()


TOKEN_TYPES: 'list[Type[TokenType]]' = [IntToken, NameToken, WhitespaceToken]


class Tokenizer2:
    def __init__(self, text):
        self.text: str = text

        self.index = 0
        self.char = ''

        self.token: Token2 | None = None
        self.state = State.NONE

        self.tokens = []

    @property
    def next_index(self):
        return self.index + 1

    @next_index.setter
    def next_index(self, value):
        self.index = value - 1

    def tokenize(self):
        while self.index < len(self.text):
            self.char = self.text[self.index]
            self.next_char()
            self.index += 1
        if self.token is not None:
            self.tokens.append(self.token)

    def next_char(self):
        if self.token is None:
            return self.new_token()
        if not self.cont_token():
            return self.new_token()

    def new_token(self):
        self.token = Token2(self)
        self.token.set_type(self.new_token_type())

    def new_token_type(self) -> TokenType:
        for t in TOKEN_TYPES:
            tok = t(self, self.token)
            if tok.start():
                tok.accept()
                return tok
        raise SyntaxError("Unknown character")

    def _cont_token(self):
        return self.token.type.cont()

    def cont_token(self):
        cont = self._cont_token()
        if cont:
            self.token.accept()
        else:
            self.tokens.append(self.token)
            self.token: Token2 | None = None
        return cont


if __name__ == '__main__':
    _t = Tokenizer2('123a_abc 764')
    _t.tokenize()
    print(_t.tokens)
