# NOTES


### `X` and `do_X`

The reason for 2 methods on TokenType: `do_X` and `X` (eg. `do_start` and `start`):
- normal token subclasses:
  - can just implement `do_X`
- classes adding extra functionality for subclasses:
  - override `X` to call `do_X` (and do some other logic)
  - subclasses of that can just implement `do_X` and can use the same name independent of inheritance