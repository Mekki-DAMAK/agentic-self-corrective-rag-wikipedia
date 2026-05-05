from __future__ import annotations

import os
from typing import Any


class TelemetryLogger:
    """Small optional W&B wrapper that keeps local execution independent from external services."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.enabled = bool(cfg.get("enabled", False)) or os.getenv("WANDB_MODE") not in {None, "disabled", "off"}
        self.run = None
        if not self.enabled:
            return
        try:
            import wandb

            self.run = wandb.init(
                project=cfg.get("project", "self-rag-wikipedia-demo"),
                entity=cfg.get("entity"),
                mode=cfg.get("mode", os.getenv("WANDB_MODE", "offline")),
                config=cfg.get("config", {}),
                reinit=True,
            )
        except Exception:
            self.run = None
            self.enabled = False

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            import wandb

            wandb.log(payload, step=step)
        except Exception:
            return

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            return
