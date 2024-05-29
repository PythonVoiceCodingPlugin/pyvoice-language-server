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
