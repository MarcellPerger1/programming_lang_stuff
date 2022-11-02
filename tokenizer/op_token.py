from __future__ import annotations

from typing import Type

from utils.str_utils import remove_suffix
from .token import TokenType, register_token
from .const import OP_CHARS

OPS: 'dict[str, Type[OpToken]]' = {}


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
    def aug_assign(cls, inherit_self=True, name: str = None,
                   bases: 'tuple[Type[OpToken]]' = None,
                   module: str = None):
        if name is None:
            cls_name = remove_suffix(cls.__name__, 'Token')
            name = f"{cls_name}AssignToken"
        if bases is None:
            bases = (cls, AugAssignBaseToken) if inherit_self else (AugAssignBaseToken,)

        @op_token(cls.op_sym + '=')
        class New(*bases):
            pass

        assert issubclass(New, OpToken)

        New.__name__ = name
        New.__qualname__ = name
        New.__module__ = module or cls.__module__  # can't really do better than that
        return New


def op_token(sym: str):
    assert isinstance(sym, str), ("@op_token() arg must be str "
                                  "(Did you remember to put symbol text in decorator?)")

    def decor(cls: Type[OpToken]):
        OPS[sym] = cls
        cls.op_sym = sym
        cls = register_token(cls)
        return cls

    return decor


class AugAssignBaseToken(OpToken):
    pass
