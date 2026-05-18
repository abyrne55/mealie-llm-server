from __future__ import annotations

import string
from importlib.resources import files


def tokenize(text: str) -> set[str]:
    tokens = set()
    for word in text.split():
        stripped = word.strip(string.punctuation).lower()
        if stripped:
            tokens.add(stripped)
    return tokens


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class Router:
    def __init__(self, threshold: float = 0.6):
        self._threshold = threshold
        self._prompts: dict[str, set[str]] = {}
        prompts_dir = files("mealie_llm_server.prompts")
        for item in prompts_dir.iterdir():
            if hasattr(item, "name") and item.name.endswith(".txt"):
                name = item.name.removesuffix(".txt")
                self._prompts[name] = tokenize(item.read_text())

    def match(self, system_message: str) -> str | None:
        truncated = system_message.split("\n###\n", 1)[0]
        tokens = tokenize(truncated)
        best_name = None
        best_score = 0.0
        for name, ref_tokens in self._prompts.items():
            score = jaccard_similarity(tokens, ref_tokens)
            if score > best_score:
                best_score = score
                best_name = name
        if best_score >= self._threshold:
            return best_name
        return None
