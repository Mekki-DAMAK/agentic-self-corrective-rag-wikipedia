from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml
from openpyxl.styles import Alignment

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_rag_pro.evaluation.ragas_eval import evaluate_batch
from self_rag_pro.pipeline import SelfRAGPipeline

CONFIG_PATH = ROOT / "configs" / "default.yaml"
CHECK_ICON = "\u2705"
BRAIN_ICON = "&#129504;"


st.set_page_config(page_title="Self-RAG Wikipedia", page_icon="\U0001F9E0", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 1800px; }
      .hero-title { font-size: 2.8rem; font-weight: 800; margin-bottom: 0.2rem; }
      .hero-subtitle { color: #8c92a3; font-weight: 600; margin-bottom: 1.2rem; }
      .step-card {
        border: 1px solid #3b3f4a;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        min-height: 58px;
        background: #151821;
        font-weight: 700;
      }
      .stMetric {
        background: #151821;
        border: 1px solid #303542;
        border-radius: 8px;
        padding: 0.75rem;
      }
      code {
        color: #f8f8f2;
        background: #151821;
        border-radius: 6px;
        padding: 0.1rem 0.35rem;
      }
      .attempt-summary {
        display: flex;
        gap: 0.6rem;
        align-items: center;
        flex-wrap: wrap;
        margin: 0.15rem 0 0.65rem 0;
      }
      .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.82rem;
        font-weight: 800;
        border: 1px solid transparent;
      }
      .status-accepted {
        color: #35f07f;
        background: rgba(34, 197, 94, 0.12);
        border-color: rgba(34, 197, 94, 0.35);
      }
      .status-rejected {
        color: #ff7b86;
        background: rgba(255, 75, 86, 0.12);
        border-color: rgba(255, 75, 86, 0.35);
      }
      .status-warning {
        color: #ffd166;
        background: rgba(245, 158, 11, 0.12);
        border-color: rgba(245, 158, 11, 0.35);
      }
      .detail-label {
        color: #9ca3af;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin-bottom: 0.25rem;
      }
      .detail-box {
        border: 1px solid #303542;
        border-radius: 8px;
        background: #11141c;
        padding: 0.75rem 0.9rem;
        min-height: 76px;
      }
      div[data-testid="stExpander"] {
        border: 1px solid #3b3f4a;
        border-radius: 8px;
        background: #0f1219;
        margin-bottom: 0.75rem;
      }
      div[data-testid="stExpander"] summary {
        font-weight: 800;
      }
      .answer-panel {
        border: 1px solid rgba(34, 197, 94, 0.35);
        border-left: 5px solid #22c55e;
        border-radius: 8px;
        background: linear-gradient(90deg, rgba(34, 197, 94, 0.12), rgba(17, 20, 28, 0.92));
        padding: 1.05rem 1.15rem;
        margin: 0.75rem 0 1rem 0;
        font-size: 1.08rem;
        line-height: 1.65;
        font-weight: 650;
      }
      .final-query {
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 8px;
        background: rgba(14, 165, 233, 0.1);
        padding: 0.75rem 0.9rem;
        margin: 0.7rem 0 1rem 0;
      }
      .section-kicker {
        color: #9ca3af;
        font-size: 0.85rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.4rem;
      }
      .score-note {
        color: #9ca3af;
        font-size: 0.92rem;
        margin-top: -0.25rem;
        margin-bottom: 0.8rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_pipeline(config_path: str) -> SelfRAGPipeline:
    return SelfRAGPipeline(config_path)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def progress_collector():
    box = st.status("Pipeline initialization", expanded=True)

    def _log(message: str) -> None:
        box.write(f"{CHECK_ICON} {message}")

    return box, _log


def render_steps() -> None:
    labels = [
        "1. User question",
        "2. Multi-query",
        "3. Hybrid Search BM25 + FAISS",
        "4. Rerank + Judge",
        "5. Verified answer",
    ]
    cols = st.columns(5)
    for col, label in zip(cols, labels, strict=False):
        col.markdown(f"<div class='step-card'>{label}</div>", unsafe_allow_html=True)


def source_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for idx, source in enumerate(result.get("sources", []), start=1):
        rows.append(
            {
                "rank": idx,
                "title": source.get("title", ""),
                "score": round(float(source.get("score", 0.0)), 3),
                "rerank_score": round(float(source.get("rerank_score", source.get("score", 0.0))), 3),
                "retrieval_score": round(float(source.get("retrieval_score", 0.0)), 3),
                "snippet": source.get("text", "")[:260],
            }
        )
    return pd.DataFrame(rows)


def debug_scores_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for attempt in result.get("timeline", []):
        for chunk in attempt.get("top_chunks", []):
            rows.append(
                {
                    "attempt": attempt.get("attempt"),
                    "query": attempt.get("query"),
                    "title": chunk.get("title", ""),
                    "bm25": round(float(chunk.get("bm25_score", 0.0)), 3),
                    "vector": round(float(chunk.get("vector_score", 0.0)), 3),
                    "retrieval": round(float(chunk.get("retrieval_score", 0.0)), 3),
                    "rerank_raw": round(float(chunk.get("rerank_raw_score", chunk.get("rerank_score", 0.0))), 3),
                    "rerank_norm": round(float(chunk.get("rerank_score", 0.0)), 3),
                    "final_score": round(float(chunk.get("score", 0.0)), 3),
                    "judge_useful": bool(chunk.get("judge_useful", False)),
                    "judge_hits": int(chunk.get("judge_lexical_hits", 0)),
                    "hit_terms": chunk.get("judge_hit_terms", ""),
                    "title_match": bool(chunk.get("judge_title_match", False)),
                }
            )
    return pd.DataFrame(rows)


def attempts_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for attempt in result.get("timeline", []):
        rows.append(
            {
                "attempt": attempt.get("attempt"),
                "status": attempt.get("status"),
                "query_sent_to_retriever": attempt.get("query"),
                "next_reformulation": attempt.get("suggested_query") or "",
                "judge_reason": attempt.get("reason", ""),
            }
        )
    return pd.DataFrame(rows)


def status_class(status: str) -> str:
    normalized = status.lower()
    if "accepted" in normalized and "warning" not in normalized:
        return "status-accepted"
    if "warning" in normalized:
        return "status-warning"
    return "status-rejected"


def attempt_chunks_table(attempt: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for idx, chunk in enumerate(attempt.get("top_chunks", []), start=1):
        rows.append(
            {
                "rank": idx,
                "title": chunk.get("title", ""),
                "final_score": round(float(chunk.get("score", 0.0)), 3),
                "retrieval": round(float(chunk.get("retrieval_score", 0.0)), 3),
                "rerank": round(float(chunk.get("rerank_score", 0.0)), 3),
                "judge_useful": bool(chunk.get("judge_useful", False)),
            }
        )
    return pd.DataFrame(rows)


def render_attempt_summary(result: dict[str, Any]) -> None:
    st.subheader("Agent Attempts")
    st.markdown(
        "<div class='score-note'>Open each attempt to inspect the query, reformulation, judge decision, retrieval variants, trace, and inspected chunks.</div>",
        unsafe_allow_html=True,
    )
    for idx, attempt in enumerate(result.get("timeline", []), start=1):
        status = attempt.get("status", "unknown")
        query = attempt.get("query", "")
        next_query = attempt.get("suggested_query")
        expanded = idx == 1 or "accepted" in status
        label = f"Attempt {attempt.get('attempt')} - {status} - query: {query}"
        with st.expander(label, expanded=expanded):
            st.markdown(
                f"""
                <div class="attempt-summary">
                  <span class="status-pill {status_class(status)}">{status}</span>
                  <span><strong>Query:</strong> <code>{query}</code></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='detail-label'>Query sent to retriever</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='detail-box'><code>{query}</code></div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='detail-label'>Next reformulation</div>", unsafe_allow_html=True)
                if next_query:
                    st.markdown(f"<div class='detail-box'><code>{next_query}</code></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='detail-box'>No reformulation needed: sources accepted or maximum attempts reached.</div>", unsafe_allow_html=True)

            st.markdown("<div class='detail-label'>Judge decision</div>", unsafe_allow_html=True)
            st.info(attempt.get("reason", ""))

            if attempt.get("queries"):
                st.markdown("<div class='detail-label'>Multi-queries sent</div>", unsafe_allow_html=True)
                st.code("\n".join(attempt.get("queries", [])), language="text")

            if attempt.get("thinking_trace"):
                st.markdown("<div class='detail-label'>Thinking trace</div>", unsafe_allow_html=True)
                for note in attempt.get("thinking_trace", []):
                    st.write(f"- {note}")

            chunks_df = attempt_chunks_table(attempt)
            if not chunks_df.empty:
                st.markdown("<div class='detail-label'>Top chunks inspected by the judge</div>", unsafe_allow_html=True)
                st.dataframe(chunks_df, width="stretch", hide_index=True)

    if not result.get("accepted"):
        st.warning("No relevant source was accepted. A refusal answer was generated.")
    elif result.get("sources"):
        st.success("Sources accepted, final answer generated.")


def ragas_chart(scores: dict[str, Any]) -> None:
    if not scores:
        return
    df = pd.DataFrame(
        [
            {"metric": "Faithfulness", "score": float(scores.get("faithfulness", 0.0))},
            {"metric": "Answer relevancy", "score": float(scores.get("answer_relevancy", 0.0))},
        ]
    )
    spec = {
        "mark": {"type": "bar", "cornerRadiusEnd": 5, "color": "#7cc7ff"},
        "encoding": {
            "y": {
                "field": "metric",
                "type": "nominal",
                "title": None,
                "sort": None,
                "axis": {"labelLimit": 180, "labelPadding": 10},
            },
            "x": {
                "field": "score",
                "type": "quantitative",
                "title": "Score",
                "scale": {"domain": [0, 1]},
                "axis": {"format": ".2f"},
            },
            "tooltip": [
                {"field": "metric", "title": "Metric"},
                {"field": "score", "title": "Score", "format": ".2f"},
            ],
        },
        "height": 120,
    }
    st.vega_lite_chart(df, spec, width="stretch")
    if scores.get("mode"):
        st.caption(f"Evaluation mode: {scores.get('mode')}")


def shorten_question(question: str, idx: int) -> str:
    return f"Q{idx:02d}"


def batch_long_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position, (_, row) in enumerate(df.reset_index(drop=True).iterrows(), start=1):
        label = shorten_question(str(row["question"]), position)
        for metric, col in [
            ("Global score", "score_global"),
            ("Faithfulness", "faithfulness"),
            ("Answer relevancy", "answer_relevancy"),
        ]:
            rows.append(
                {
                    "question_id": label,
                    "question": row["question"],
                    "metric": metric,
                    "score": float(row.get(col, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def score_bar_chart(
    data: pd.DataFrame,
    score_col: str,
    title: str,
    height: int,
    color: str = "#22c55e",
) -> None:
    chart_data = data[["question_id", "question", score_col]].rename(columns={score_col: "score"})
    spec = {
        "mark": {"type": "bar", "cornerRadiusEnd": 4, "color": color},
        "encoding": {
            "y": {
                "field": "question_id",
                "type": "ordinal",
                "title": None,
                "sort": None,
                "axis": {"labelLimit": 80},
            },
            "x": {
                "field": "score",
                "type": "quantitative",
                "title": title,
                "scale": {"domain": [0, 1]},
            },
            "tooltip": [
                {"field": "question_id", "title": "ID"},
                {"field": "question", "title": "Question"},
                {"field": "score", "title": "Score", "format": ".2f"},
            ],
        },
        "height": height,
    }
    st.vega_lite_chart(chart_data, spec, width="stretch")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
        worksheet = writer.sheets["results"]
        widths = {
            "A": 46,
            "B": 14,
            "C": 14,
            "D": 18,
            "E": 12,
            "F": 10,
            "G": 28,
            "H": 90,
        }
        for column, width in widths.items():
            worksheet.column_dimensions[column].width = width
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
    return buffer.getvalue()


def run_agent(question: str) -> dict[str, Any]:
    pipeline = get_pipeline(str(CONFIG_PATH))
    status_box, progress = progress_collector()
    result = pipeline.ask(question, progress=progress, with_ragas=True)
    status_box.update(label="Execution complete", state="complete", expanded=True)
    return result


st.markdown(f"<div class='hero-title'>{BRAIN_ICON} Agentic & Self-Corrective RAG</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-subtitle'>Hybrid Search, Multi-query, Cross-Encoder Reranking, Self-RAG Judge, LLM Reformulation, Grounding Check, RAGAS</div>",
    unsafe_allow_html=True,
)
render_steps()

tab_single, tab_debug, tab_batch = st.tabs(["Single Question", "Score Debug", "Batch Test"])

with tab_single:
    st.subheader("Single-question Self-RAG agent")
    question = st.text_input("Question", value="What is machine learning?")
    col_run, col_hint = st.columns([1, 3])
    with col_run:
        run_clicked = st.button("Run Self-RAG agent", type="primary", width="stretch")
    with col_hint:
        st.caption(
            "The pipeline displays internal steps: multi-query generation, hybrid retrieval, reranking, judging, reformulation, answer generation, and verification."
        )

    if run_clicked:
        try:
            result = run_agent(question)
            st.session_state["last_result"] = result

            if result.get("accepted"):
                st.success("Final answer generated.")
            else:
                st.warning("I could not find any relevant information about this topic in my dataset.")

            st.markdown("<div class='section-kicker'>Generated response</div>", unsafe_allow_html=True)
            st.subheader("Final Answer")
            st.markdown(f"<div class='answer-panel'>{result.get('answer', '')}</div>", unsafe_allow_html=True)

            st.subheader("Main Scores")
            c1, c2, c3, c4, c5 = st.columns(5)
            sources = result.get("sources", [])
            verification = result.get("verification", {})
            best_source_score = max((float(source.get("score", 0.0)) for source in sources), default=0.0)
            c1.metric("Confidence", f"{float(result.get('confidence', 0.0)):.2f}")
            c2.metric("Accepted", str(bool(result.get("accepted", False))))
            c3.metric("Attempts", str(result.get("attempts", 0)))
            c4.metric("Grounding", f"{float(verification.get('grounding_ratio', 0.0)):.2f}")
            c5.metric("Best source score", f"{best_source_score:.2f}")
            st.markdown(
                f"<div class='final-query'><strong>Final query:</strong> <code>{result.get('final_query', question)}</code></div>",
                unsafe_allow_html=True,
            )

            st.subheader("RAGAS / Local Fallback Scores")
            ragas_scores = result.get("ragas", {})
            r1, r2 = st.columns(2)
            r1.metric("Faithfulness", f"{float(ragas_scores.get('faithfulness', 0.0)):.2f}")
            r2.metric("Answer relevancy", f"{float(ragas_scores.get('answer_relevancy', 0.0)):.2f}")
            ragas_chart(ragas_scores)

            render_attempt_summary(result)

            st.subheader("Accepted Sources")
            sources_df = source_table(result)
            if sources_df.empty:
                st.info("No source was accepted for this question.")
            else:
                st.dataframe(sources_df, width="stretch", hide_index=True)

        except Exception as exc:  # pragma: no cover - Streamlit UI guard
            st.error(str(exc))
            st.code(
                "python scripts/download_wikipedia_subset.py --config configs/default.yaml\n"
                "python scripts/build_index.py --config configs/default.yaml"
            )

with tab_debug:
    st.subheader("Score Debug")
    with st.expander("Active configuration", expanded=False):
        config = load_config()
        runtime = config.get("runtime", {})
        retrieval = config.get("retrieval", {})
        query_rewriting = config.get("query_rewriting", {})
        models = config.get("models", {})
        generation = config.get("generation", {})
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Runtime**")
        c1.code(f"offline: {runtime.get('offline')}\ndevice: {runtime.get('device')}")
        c2.markdown("**Retrieval**")
        c2.code(
            "\n".join(
                [
                    f"require_reranker: {retrieval.get('require_reranker')}",
                    f"max_attempts: {retrieval.get('max_attempts')}",
                    f"bm25_top_k: {retrieval.get('bm25_top_k')}",
                    f"vector_top_k: {retrieval.get('vector_top_k')}",
                    f"fused_top_k: {retrieval.get('fused_top_k')}",
                    f"rerank_top_k: {retrieval.get('rerank_top_k')}",
                ]
            )
        )
        c3.markdown("**Agent**")
        c3.code(
            "\n".join(
                [
                    f"LLM rewriter: {query_rewriting.get('use_llm')}",
                    f"require LLM: {query_rewriting.get('require_llm')}",
                    f"model: {models.get('query_rewriter')}",
                    f"local generator: {generation.get('use_local_generator')}",
                ]
            )
        )
        st.markdown("**Full YAML**")
        st.code(yaml.safe_dump(config, sort_keys=False), language="yaml")

    last_result = st.session_state.get("last_result")
    if not last_result:
        st.info("Run a question in the Single Question tab to display detailed scores.")
    else:
        df_debug = debug_scores_table(last_result)
        if df_debug.empty:
            st.info("No candidate scores are available.")
        else:
            st.dataframe(df_debug, width="stretch", hide_index=True)

        st.markdown("### Score Definitions")
        st.write(
            "- **BM25**: lexical score based on exact term matching.\n"
            "- **Vector**: semantic similarity score from the FAISS embedding search.\n"
            "- **Retrieval**: normalized hybrid score after BM25 + vector fusion and bonuses.\n"
            "- **Rerank raw**: original Cross-Encoder score, which can be negative.\n"
            "- **Rerank norm**: Cross-Encoder score normalized between 0 and 1.\n"
            "- **Final score**: weighted score used by the judge.\n"
            "- **Judge useful**: whether the Self-RAG judge considered the source relevant enough."
        )

with tab_batch:
    st.subheader("Batch test with RAGAS / local fallback scores")
    default_questions = "\n".join(
        [
            "What is artificial intelligence?",
            "What is machine learning?",
            "What is a neural network?",
            "What is overfitting?",
            "What is the difference between classification and regression?",
        ]
    )
    questions_text = st.text_area("Questions, one per line", value=default_questions, height=180)
    run_batch = st.button("Run batch evaluation", type="primary", key="batch_run")

    if run_batch:
        questions = [line.strip() for line in questions_text.splitlines() if line.strip()]
        if not questions:
            st.warning("Add at least one question.")
        else:
            try:
                pipeline = get_pipeline(str(CONFIG_PATH))
                progress = st.progress(0, text="Starting batch evaluation")
                results = []
                for idx, batch_question in enumerate(questions, start=1):
                    progress.progress((idx - 1) / len(questions), text=f"Question {idx}/{len(questions)}: {batch_question}")
                    result = pipeline.ask(batch_question, with_ragas=False)
                    results.append(result)
                progress.progress(1.0, text="Batch evaluation complete")

                evaluation = evaluate_batch(results, questions, load_config())
                rows = []
                for idx, (batch_question, result, score) in enumerate(
                    zip(questions, results, evaluation["per_question"], strict=False), start=1
                ):
                    accepted_sources = result.get("sources", [])
                    rows.append(
                        {
                            "question_id": shorten_question(batch_question, idx),
                            "question": batch_question,
                            "score_global": float(score.get("score_global", 0.0)),
                            "faithfulness": float(score.get("faithfulness", 0.0)),
                            "answer_relevancy": float(score.get("answer_relevancy", 0.0)),
                            "accepted": bool(result.get("accepted", False)),
                            "attempts": int(result.get("attempts", 0)),
                            "main_source": accepted_sources[0].get("title", "") if accepted_sources else "",
                            "answer": result.get("answer", ""),
                        }
                    )
                df = pd.DataFrame(rows)

                st.metric("Average faithfulness", f"{evaluation['mean_faithfulness']:.3f}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Average answer relevancy", f"{evaluation['mean_answer_relevancy']:.3f}")
                c2.metric("Global score", f"{evaluation['score_global']:.3f}")
                c3.metric("Questions evaluated", str(len(questions)))

                st.subheader("Global Gauges")
                g1, g2, g3 = st.columns(3)
                g1.progress(float(evaluation["mean_faithfulness"]), text=f"Faithfulness {evaluation['mean_faithfulness']:.2f}")
                g2.progress(
                    float(evaluation["mean_answer_relevancy"]),
                    text=f"Answer relevancy {evaluation['mean_answer_relevancy']:.2f}",
                )
                g3.progress(float(evaluation["score_global"]), text=f"Global score {evaluation['score_global']:.2f}")

                st.subheader("Charts")
                chart_df = df.copy()
                score_bar_chart(chart_df, "score_global", "Global score", max(180, 38 * len(chart_df)))
                left, right = st.columns(2)
                with left:
                    st.markdown("**Faithfulness**")
                    score_bar_chart(chart_df, "faithfulness", "Faithfulness", max(160, 34 * len(chart_df)))
                with right:
                    st.markdown("**Answer relevancy**")
                    score_bar_chart(chart_df, "answer_relevancy", "Answer relevancy", max(160, 34 * len(chart_df)))

                with st.expander("Question ID mapping", expanded=False):
                    st.dataframe(df[["question_id", "question"]], width="stretch", hide_index=True)

                st.subheader("Detailed Table")
                st.dataframe(df, width="stretch", hide_index=True)

                excel_bytes = to_excel_bytes(df)
                st.download_button(
                    "Download Excel results (.xlsx)",
                    data=excel_bytes,
                    file_name="ragas_batch_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as exc:  # pragma: no cover - Streamlit UI guard
                st.error(str(exc))
                st.code(
                    "python scripts/download_wikipedia_subset.py --config configs/default.yaml\n"
                    "python scripts/build_index.py --config configs/default.yaml"
                )
