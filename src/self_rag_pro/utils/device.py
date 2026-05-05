from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> str:
    preference = (preference or "auto").lower()
    if preference == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
