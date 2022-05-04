# TODO: fix git clone not working!!!
# TODO: files!!

from __future__ import annotations

import string
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from inspect import isclass
from typing import Type, overload, TypeVar, Callable, List

NAME_CHARS = string.ascii_letters + string.digits + '_'
OP_CHARS = '+-*/='
VALID_CHARS = NAME_CHARS + OP_CHARS


class EndToken(Exception):
    pass


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

    @property
    def state(self):
        return self.tokenizer.state


@dataclass
class Token(UsesTokenizer):
    tokenizer: Tokenizer2 = field(repr=False)
    text: str = ''
    type: TokenType = None

    def set_type(self, t: TokenType | Type[TokenType]):
        if isclass(t):
            self.set_class(t)
        else:
            self.type = t

    def set_class(self, c: Type[TokenType]):
        self.set_type(c(self.tokenizer, self))

    def accept(self):
        self.text += self.char


class TokenType(UsesTokenizer, ABC):
    def __init__(self, tokenizer: Tokenizer2, token: Token):
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

    def set_type(self, c: TokenType | Type[TokenType]):
        if isclass(c):
            self.token.set_class(c)
        else:
            self.token.set_type(c)

    def make_token(self, text: str = None, t=None):
        tok = Token(self.tokenizer, text if text is not None else self.text)
        tok.set_type(t if t is not None else self)
        return tok

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

    def end(self) -> List[Token] | None:
        pass


TOKEN_TYPES: 'list[Type[TokenType]]' = []

TT = TypeVar('TT', bound=Type[TokenType])


@overload
def register_token(cls: TT, /) -> TT: ...


@overload
def register_token(at: int | None, /) -> Callable[[TT], TT]: ...


def register_token(arg: Type[TokenType] | int | None):
    at: int | None = arg if not isclass(arg) else None

    def decor(cls: Type[TokenType]):
        if at is None:
            TOKEN_TYPES.append(cls)
        else:
            TOKEN_TYPES.insert(at, cls)
        return cls

    if isclass(arg):
        return decor(arg)
    return decor


@register_token
class IntToken(TokenType):
    def do_cont(self) -> bool:
        return self.char in string.digits

    def do_start(self) -> bool:
        return self.char in string.digits


@register_token
class NameToken(TokenType):
    def do_start(self) -> bool:
        return self.char in NAME_CHARS

    def do_cont(self) -> bool:
        return self.char in NAME_CHARS


@register_token
class WhitespaceToken(TokenType):
    def do_start(self) -> bool:
        return self.char.isspace()

    def do_cont(self) -> bool:
        return self.char.isspace()


class OpToken(TokenType):
    op_sym: str | None = None

    def do_start(self) -> bool:
        return (self.char in OP_CHARS
                and (self.op_sym is None
                     or self.op_sym.startswith(self.text)))

    def do_cont(self) -> bool:
        if self.char not in OP_CHARS:
            return False
        return True

    def end(self):
        c = OPS.get(self.text)
        if c is None:
            return self.separate_tokens()
        self.set_type(c)

    def separate_tokens(self):
        tokens = []
        text = self.text
        while text:
            op_symbols = sorted(OPS.keys(), key=lambda sym: len(sym), reverse=True)
            for sym in op_symbols:
                if text.startswith(sym):
                    tokens.append(self.make_token(sym, OPS[sym]))
                    text = text[len(sym):]
                    break
            else:
                raise SyntaxError(f"Invalid operator: {self.text!r}")
        return tokens


OPS: 'dict[str, Type[OpToken]]' = {}


def op_token(sym: str):
    assert isinstance(sym, str), ("@op_token() arg must be str "
                                  "(Did you remember to put symbol text in decorator?)")

    def decor(cls: Type[OpToken]):
        OPS[sym] = cls
        cls.op_sym = sym
        cls = register_token(cls)
        return cls

    return decor


@op_token("+")
class AddToken(OpToken):
    pass


@op_token("-")
class SubToken(OpToken):
    pass


@op_token("*")
class MulToken(OpToken):
    pass


@op_token("/")
class DivToken(OpToken):
    pass


@op_token("=")
class EqToken(OpToken):
    pass


class Tokenizer2:
    def __init__(self, text):
        self.text: str = text

        self.index = 0
        self.char = ''

        self.token: Token | None = None
        self.state = State.NONE

        self.tokens: 'list[Token]' = []

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
            self.end_token()

    def next_char(self):
        if self.token is None:
            return self.new_token()
        if not self.cont_token():
            return self.new_token()

    def new_token(self):
        self.token = Token(self)
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
            self.end_token()
        return cont

    def end_token(self):
        tokens = self.token.type.end()
        if tokens:
            self.tokens.extend(tokens)
        else:
            self.tokens.append(self.token)
        self.token: Token | None = None


def _remove_suffix(s: str, suffix: str):
    if s.endswith(suffix):
        return s[:-len(suffix)]
    return s


if __name__ == '__main__':
    _t = Tokenizer2('123a_abc+764- obj/62 *+9 az++')
    _t.tokenize()
    for _tok in _t.tokens:
        if isinstance(_tok.type, WhitespaceToken):
            continue
        print(
            _remove_suffix(type(_tok.type).__name__, 'Token').upper().rjust(15),
            ' | ',
            _tok.text if not isinstance(_tok.type, WhitespaceToken) else repr(_tok.text),
        )
