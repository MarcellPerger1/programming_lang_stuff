from __future__ import annotations

import string

from .const import NAME_CHARS
from .op_token import op_token, OpToken
from .token import register_token, TokenType


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

    def text_repr(self) -> str:
        return repr(self.text)


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


@op_token('%')
class ModToken(OpToken):
    pass


@op_token("=")
class EqToken(OpToken):
    pass


AssignAddToken = AddToken.aug_assign()
AssignSubToken = SubToken.aug_assign()
AssignMulToken = MulToken.aug_assign()
AssignDivToken = DivToken.aug_assign()
AssignModToken = ModToken.aug_assign()


# TODO InvalidToken - copy from `pymeowlib/dev-new-parser branch`
#  and also post-processor to join adjeacent invalidToekn s


def init_tokens():
    """This can be imported then called to make sure that all tokens are initialised"""
    pass
