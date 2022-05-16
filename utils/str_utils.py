def remove_suffix(s: str, suffix: str):
    if s.endswith(suffix):
        return s[:-len(suffix)]
    return s


def remove_prefix(s: str, prefix: str):
    if s.startswith(prefix):
        return s[len(prefix):]
    return s


def is_cased(c: str):
    return c.isupper() or c.islower()


def to_snake_case(s: str, upper=False, sep='_'):
    out = ''
    last = None
    for c in s:
        if c in '_ -' + sep:
            out += sep
        elif last is None:
            out += c.lower()
        else:
            # last is letter and on boundary
            if is_cased(last) and last.islower() and c.isupper():
                out += sep + c.lower()
            else:
                out += c.lower()
        last = c
    if upper:
        out = out.upper()
    return out
