# TODO: fix git clone not working!!!
# TODO: files!!

from __future__ import annotations

import string
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from inspect import isclass
from typing import Type, overload, TypeVar, Callable, List, Iterable


def _remove_suffix(s: str, suffix: str):
    if s.endswith(suffix):
        return s[:-len(suffix)]
    return s


def _to_snake_case(s: str, upper=False):
    out = ''
    last = None
    for c in s:
        if last is None:
            out += c.lower()
        else:
            if last not in '- _' and last.islower() and c.isupper():
                out += '_' + c.lower()
            else:
                out += c.lower()
        last = c
    if upper:
        out = out.upper()
    return out


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
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def char(self):
        return self.tokenizer.char

    @property
    def state(self):
        return self.tokenizer.state


@dataclass
class Token(UsesTokenizer):
    tokenizer: Tokenizer = field(repr=False)
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
    def __init__(self, tokenizer: Tokenizer, token: Token):
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

    @classmethod
    def aug_assign(cls, name: str = None, bases: 'tuple[Type[OpToken]]' = None,
                   module: str = None):
        if name is None:
            cls_name = _remove_suffix(cls.__name__, 'Token')
            name = f"{cls_name}AssignToken"
        if bases is None:
            bases = (OpToken,)

        @op_token(cls.op_sym + '=')
        class New(*bases):
            pass
        assert issubclass(New, OpToken)

        New.__name__ = name
        New.__qualname__ = name
        New.__module__ = module or cls.__module__  # can't really do better than that
        return New


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


AssignAddToken = AddToken.aug_assign()
AssignSubToken = SubToken.aug_assign()
AssignMulToken = MulToken.aug_assign()
AssignDivToken = DivToken.aug_assign()


class Tokenizer:
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


def print_token_stream(tks: Iterable[Token], print_ws=False):
    for _tok in tks:
        if not print_ws and isinstance(_tok.type, WhitespaceToken):
            continue
        print(
            _to_snake_case(
                _remove_suffix(type(_tok.type).__name__, 'Token')).upper().rjust(15),
            ' | ',
            _tok.text if not isinstance(_tok.type, WhitespaceToken) else repr(_tok.text),
        )


if __name__ == '__main__':
    _t = Tokenizer('123a_abc+764 =- obj/62 *+9 az++ or s=90-7 and q+=-8 or w/=98-i\n'
                    ' bc*=8 or ee+q  -=4gt9')
    _t.tokenize()
    print_token_stream(_t.tokens, True)
