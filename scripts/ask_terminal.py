from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_rag_pro.pipeline import SelfRAGPipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    pipe = SelfRAGPipeline(args.config, progress=lambda m: print("[Self-RAG]", m))
    result = pipe.ask(args.question, with_ragas=False)
    print("\nANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"][:3]:
        print(f"- {s['title']} | score={s.get('score',0):.3f} | {s['url']}")


if __name__ == "__main__":
    main()
