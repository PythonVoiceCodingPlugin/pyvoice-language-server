import functools
import re

__all__ = ["speak_single_item", "speak_items"]

pattern = re.compile(r"[A-Z][a-z]+|[A-Z]+|[a-z]+|\d")
digits_to_names = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "fife",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

pronunciation_dict = {
    "casefold": "case fold",
    "endswith": "ends with",
    "expandtabs": "expand tabs",
    "format_map": "format map",
    "isalnum": "is alnum",
    "isalpha": "is alpha",
    "isascii": "is ascii",
    "isdecimal": "is decimal",
    "isdigit": "is digit",
    "isidentifier": "is identifier",
    "islower": "is lower",
    "isnumeric": "is numeric",
    "isprintable": "is printable",
    "isspace": "is space",
    "istitle": "is title",
    "isupper": "is upper",
    "join": "join",
    "ljust": "L just",
    "lstrip": "L strip",
    "maketrans": "make trans",
    "removeprefix": "remove prefix",
    "removesuffix": "remove suffix",
    "rfind": "R find",
    "rindex": "R index",
    "rjust": "R just",
    "rpartition": "R partition",
    "rsplit": "R split",
    "rstrip": "R strip",
    "splitlines": "split lines",
    "startswith": "starts with",
    "swapcase": "swap case",
    "zfill": "Z fill",
    "str": "string",
    "int": "integer",
    "isinstance": "is instance",
    "issubclass": "is subclass",
}

digits_to_names.update(pronunciation_dict)


@functools.lru_cache(maxsize=8192)
def speak_single_item(text):
    return " ".join(
        [
            digits_to_names.get(w, w.upper() if len(w) < 3 else w.lower())
            for w in pattern.findall(text)
            if w
        ]
    )


def speak_items(items_list):
    return {speak_single_item(x): x for x in items_list}
