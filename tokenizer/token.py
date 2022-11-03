from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from inspect import isclass
from typing import Type, TYPE_CHECKING, Callable, overload, TypeVar, List

from .base_classes import UsesTokenizer
from utils.str_utils import to_snake_case, remove_suffix

if TYPE_CHECKING:
    from .tokenizer import Tokenizer


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
    invalid = False

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

    def set_type(self, c: TokenType | type[TokenType]):
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

    def text_repr(self) -> str:
        return self.text

    @classmethod
    def type_repr(cls, upper=False) -> str:
        return to_snake_case(remove_suffix(cls.__name__, 'Token'), upper)


TOKEN_TYPES: list[type[TokenType]] = []
INVALID_TOKEN_TYPES: list[type[TokenType]] = []

TT = TypeVar('TT', bound=Type[TokenType])


@overload
def register_token(cls: TT, /) -> TT: ...


@overload
def register_token(at: int | None, /) -> Callable[[TT], TT]: ...


def register_token(arg: type[TokenType] | int | None):
    at: int | None = arg if not isclass(arg) else None

    def decor(cls: type[TokenType]):
        target = INVALID_TOKEN_TYPES if getattr(cls, 'invalid', False) else TOKEN_TYPES
        if at is None:
            target.append(cls)
        else:
            target.insert(at, cls)
        return cls

    if isclass(arg):
        return decor(arg)
    return decor
