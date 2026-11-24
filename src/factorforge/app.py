
from __future__ import annotations

import os
from copy import deepcopy

import pandas as pd
import streamlit as st

from .config import load_config
from .llm import CURRENT_DEEPSEEK_FLASH_MODEL, DASHSCOPE_DEFAULT_MODEL
from .pipeline import DemoPipeline


def main() -> None:
    st.set_page_config(
        page_title="QuantAgent",
        page_icon="⚒️",
        layout="wide",
    )

    st.title("QuantAgent · AI Quantitative Factor Research")
    st.caption(
        "From structured hypotheses and restricted DSL to IC analysis, "
        "cost-aware Top-K backtesting, and feedback-driven iteration."
    )

    st.warning(
        "The default demo uses deterministic simulated A-share-style data. "
        "All numerical results are for pipeline validation only, not historical "
        "performance, investment advice, or evidence of tradability."
    )

    with st.sidebar:
        st.header("Experiment Settings")

        mode = st.selectbox(
            "LLM Mode",
            ["mock", "deepseek", "dashscope"],
            index=0,
        )

        model_options = (
            [CURRENT_DEEPSEEK_FLASH_MODEL, "deepseek-v4-pro"]
            if mode == "deepseek"
            else [DASHSCOPE_DEFAULT_MODEL, "deepseek-v4-pro"]
        )

        model = st.selectbox(
            "Model",
            model_options,
            disabled=mode == "mock",
        )

        periods = st.slider("Trading Days", 100, 500, 260, 20)
        symbols = st.slider("Number of Stocks", 10, 100, 30, 5)
        fee_bps = st.slider("One-way Trading Fee (bps)", 0.0, 30.0, 10.0, 1.0)
        top_k = st.slider("Top-K", 2, min(20, symbols), min(5, symbols), 1)

        required_key = {
            "deepseek": "DEEPSEEK_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
        }.get(mode)

        if required_key and not os.environ.get(required_key):
            st.error(
                f"{required_key} not detected. Switch to Mock or configure "
                "the environment variable."
            )

        run = st.button(
            "Run Full Research Pipeline",
            type="primary",
            use_container_width=True,
        )

    if not run:
        _landing()
        return

    if required_key and not os.environ.get(required_key):
        st.stop()

    config = deepcopy(load_config())
    config["data"]["periods"] = periods
    config["data"]["symbols"] = symbols
    config["backtest"]["fee_bps"] = fee_bps
    config["backtest"]["top_k"] = top_k
    config["llm"]["mode"] = mode
    config["llm"]["model"] = model

    with st.spinner(
        "Generating factors, validating expressions, and running research "
        "and backtests..."
    ):
        try:
            results, report = DemoPipeline(config).run(
                output_dir="artifacts/streamlit",
                llm_mode=mode,
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

    comparison = pd.DataFrame([item.summary() for item in results])

    tab_overview, tab_ic, tab_backtest, tab_hypotheses = st.tabs(
        [
            "Experiment Comparison",
            "IC & Quantile Analysis",
            "Strategy Backtest",
            "Hypotheses & Feedback",
        ]
    )

    with tab_overview:
        required_categories = [
            "classic",
            "random",
            "llm_single",
            "llm_feedback",
        ]

        representatives = (
            comparison.sort_values(["category", "name"])
            .groupby("category", as_index=False)
            .first()
        )

        st.subheader("All Four Research Baseline Groups")

        cols = st.columns(4)

        labels = {
            "classic": "Hand-crafted Factors",
            "random": "Random Factor",
            "llm_single": "Single-pass LLM",
            "llm_feedback": "Feedback-driven LLM",
        }

        for column, category in zip(cols, required_categories):
            row = representatives[
                representatives["category"] == category
            ]

            with column:
                st.metric(
                    labels[category],
                    "Completed" if not row.empty else "Missing",
                )

                if not row.empty:
                    st.caption(str(row.iloc[0]["name"]))

        display_columns = [
            "category",
            "name",
            "coverage",
            "ic_mean",
            "rank_ic_mean",
            "rank_icir",
            "mean_turnover",
            "annual_return",
            "max_drawdown",
        ]

        st.dataframe(
            comparison[display_columns].style.format(
                {
                    key: "{:.4f}"
                    for key in display_columns
                    if key not in {"category", "name"}
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab_ic:
        chosen_name = st.selectbox(
            "Select Factor",
            comparison["name"].tolist(),
            index=len(comparison) - 1,
            key="ic_factor",
        )

        chosen = next(
            item for item in results if item.name == chosen_name
        )

        st.line_chart(chosen.daily_ic[["ic", "rank_ic"]])

        if not chosen.quantile_returns.empty:
            st.bar_chart(
                chosen.quantile_returns.filter(regex=r"^Q\d+$").mean()
            )

        st.caption(
            "IC measures the cross-sectional correlation between factor "
            "values at day t and next-period returns. Quantile returns "
            "are not annualized."
        )

    with tab_backtest:
        names = st.multiselect(
            "Compare Net Asset Values",
            comparison["name"].tolist(),
            default=[
                results[0].name,
                results[-2].name,
                results[-1].name,
            ],
        )

        if names:
            curves = pd.concat(
                {
                    item.name: item.equity_curve["net_value"]
                    for item in results
                    if item.name in names
                },
                axis=1,
            )

            st.line_chart(curves)

        st.caption(
            f"Equal-weight Top-{top_k} strategy. Signals are formed after "
            f"the close on day t; returns use t to t+1. Rebalancing costs "
            f"are deducted at {fee_bps:.1f} bps."
        )

    with tab_hypotheses:
        proposals = [
            item for item in results if item.proposal is not None
        ]

        for item in proposals:
            with st.expander(
                f"{item.category} · {item.name}",
                expanded=False,
            ):
                proposal = item.proposal

                st.code(proposal.expression, language="text")
                st.markdown(f"**Hypothesis:** {proposal.hypothesis}")
                st.markdown(
                    f"**Economic Rationale:** {proposal.economic_rationale}"
                )
                st.markdown(
                    f"**Iteration Notes:** {proposal.iteration_note}"
                )
                st.markdown(
                    "**Risks:** " + "; ".join(proposal.risk_notes)
                )

    st.success(
        f"Report saved to {report['paths']['summary']}. "
        f"Completed {len(results)} experiments."
    )


def _landing() -> None:
    col1, col2, col3 = st.columns(3)

    col1.info(
        "① LLM / Mock generates structured financial hypotheses "
        "and factor DSL expressions."
    )

    col2.info(
        "② Validates expressions and calculates IC, Rank IC, "
        "quantile returns, and turnover."
    )

    col3.info(
        "③ Runs cost-aware Top-K backtests and feeds diagnostics "
        "into the next factor iteration."
    )

    st.subheader("Built-in Factor DSL")

    st.code(
        "Rank(Mean(close / Delay(close, 1) - 1, 20))\n"
        "- 0.5 * Rank(Std(close / Delay(close, 1) - 1, 20))",
        language="text",
    )

    st.caption(
        "Supported operators: Rank, Delay, Delta, Mean, Std, Corr, "
        "Sum, Min, Max, Abs, Sign. Fields and functions are "
        "restricted to an explicit whitelist."
    )


if __name__ == "__main__":
    main()
