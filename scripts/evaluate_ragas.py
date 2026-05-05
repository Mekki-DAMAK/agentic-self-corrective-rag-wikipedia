from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_rag_pro.pipeline import SelfRAGPipeline  # noqa: E402

DEFAULT_QUESTIONS = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is a neural network?",
    "What is overfitting?",
    "What is the difference between classification and regression?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="data/eval/ragas_results.csv")
    args = parser.parse_args()

    pipeline = SelfRAGPipeline(args.config, progress=lambda message: print("[Eval]", message))
    rows = []
    for question in DEFAULT_QUESTIONS:
        result = pipeline.ask(question, with_ragas=True)
        ragas_scores = result.get("ragas") or {}
        rows.append(
            {
                "question": question,
                "answer": result["answer"],
                "accepted": result["accepted"],
                "attempts": result["attempts"],
                "faithfulness": ragas_scores.get("faithfulness", 0.0),
                "answer_relevancy": ragas_scores.get("answer_relevancy", 0.0),
                "evaluation_mode": ragas_scores.get("mode", "not_available"),
                "source": result["sources"][0]["title"] if result["sources"] else "",
            }
        )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(df)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
