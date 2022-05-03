import string
from enum import Enum, auto
from abc import ABC, abstractmethod

NAME_START = string.ascii_letters

class EndToken(Exception):
  pass

class Tokens(Enum):
  INT = auto()
  NAME = auto() # todo rename to IDENTIFIER

class Token(ABC):
  @abstractmethod
  def do_start(self, char: str) -> bool:
    pass

  @abstractmethod
  def do_continue(self, char: str) -> bool:
    pass

class SpaceSepToken(Token, ABC):
  # todo tokens separated by spaces
  pass

class IntToken(Token):
  def __init__(self):
    pass

  def do_start(self, char: str) -> bool:
    return char in string.digits






class Tokenizer:
  def __init__(self, text):
    self.text: str = text

  def tokenize(self):
    self.tokens = []
    self.tok_str = ''
    self.tok_type = None
    for char in self.text:
      if self.tok_type is None:
        if char in string.digits:
          self.tok_type = Tokens.INT
        elif char.isalpha():
          self.tok_type = Tokens.NAME


