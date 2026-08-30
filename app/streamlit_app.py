"""
AegisRisk AI
Merchant Risk Command Center

Streamlit dashboard using the existing frozen model,
frozen decision policy, and SHAP explainability system.
"""

import sys
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.scorer import MerchantRiskScorer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AegisRisk - Merchant Risk Command Center",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AegisRisk AI - Merchant Risk Command Center")
st.markdown(
    "Explainable fraud-risk assessment for merchants"
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = ROOT / "data" / "processed" / "transactions_features.csv"
DECISION_LOG_PATH = ROOT / "logs" / "merchant_decisions.csv"


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource
def load_scorer():
    """Load the existing frozen model scorer."""
    return MerchantRiskScorer()


@st.cache_data
def load_processed_data():
    """Load existing processed feature data."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH, nrows=2000)


@st.cache_data
def load_scored_data():
    """
    Score the existing processed dataset using the existing
    inference pipeline.
    """
    data = load_processed_data()
    scorer = load_scorer()

    return scorer.score_transactions(
        data,
        include_explanation=True,
        top_k=5,
    )


def load_decision_log():
    """Load merchant decisions if the prototype log exists."""
    if not DECISION_LOG_PATH.exists():
        return pd.DataFrame(
            columns=[
                "transaction_id",
                "ai_recommendation",
                "merchant_action",
                "merchant_action_timestamp",
                "risk_score",
                "risk_level",
            ]
        )

    return pd.read_csv(DECISION_LOG_PATH)


def save_decision(
    transaction_id,
    ai_recommendation,
    merchant_action,
    risk_score,
    risk_level,
):
    """Save one merchant decision without duplicate transaction decisions."""

    DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "transaction_id",
        "ai_recommendation",
        "merchant_action",
        "merchant_action_timestamp",
        "risk_score",
        "risk_level",
    ]

    if DECISION_LOG_PATH.exists():
        log = pd.read_csv(DECISION_LOG_PATH)
    else:
        log = pd.DataFrame(columns=columns)

    # Prevent duplicate decision rows for the same transaction.
    if not log.empty and (
        log["transaction_id"].astype(str) == str(transaction_id)
    ).any():
        return False

    new_record = pd.DataFrame(
        [
            {
                "transaction_id": transaction_id,
                "ai_recommendation": ai_recommendation,
                "merchant_action": merchant_action,
                "merchant_action_timestamp": datetime.now().isoformat(),
                "risk_score": risk_score,
                "risk_level": risk_level,
            }
        ]
    )

    log = pd.concat([log, new_record], ignore_index=True)

    log = log[columns]

    log.to_csv(DECISION_LOG_PATH, index=False)

    return True


# ============================================================
# LOAD SYSTEM
# ============================================================

try:
    scorer = load_scorer()
    scored = load_scored_data()
    metrics = scorer.get_metrics()

except Exception as exc:
    st.error("AegisRisk could not load the required system components.")
    st.exception(exc)
    st.stop()


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "transaction_id",
    "amount",
    "fraud_probability",
    "risk_decision",
    "risk_action",
]

missing_columns = [
    column for column in required_columns
    if column not in scored.columns
]

if missing_columns:
    st.error(
        "Required dashboard columns are missing: "
        + ", ".join(missing_columns)
    )
    st.stop()


probability_valid = (
    scored["fraud_probability"].between(0, 1).all()
)

if not probability_valid:
    st.error("Invalid fraud probabilities detected.")
    st.stop()


valid_risk_levels = {
    "LOW_RISK",
    "MEDIUM_RISK",
    "HIGH_RISK",
}

invalid_risk_levels = set(
    scored["risk_decision"].dropna().unique()
) - valid_risk_levels

if invalid_risk_levels:
    st.error(
        f"Invalid risk levels detected: {invalid_risk_levels}"
    )
    st.stop()


valid_actions = {
    "ALLOW",
    "REVIEW",
    "HOLD_FOR_VERIFICATION",
}

invalid_actions = set(
    scored["risk_action"].dropna().unique()
) - valid_actions

if invalid_actions:
    st.error(
        f"Invalid risk actions detected: {invalid_actions}"
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## Navigation")

page = st.sidebar.radio(
    "Select view:",
    [
        "Overview",
        "Risk Queue",
        "Investigation",
        "Decisions Log",
        "System Status",
    ],
)

st.sidebar.markdown("---")

st.sidebar.markdown("## Model Information")

st.sidebar.write(
    f"**Model:** {metrics.get('model_type', 'Logistic Regression')}"
)

st.sidebar.write(
    f"**Input features:** {metrics.get('input_features', 'N/A')}"
)

st.sidebar.write(
    f"**Review threshold:** "
    f"{metrics.get('policy_review_threshold', 0.35):.0%}"
)

st.sidebar.write(
    f"**Hold threshold:** "
    f"{metrics.get('policy_hold_threshold', 0.70):.0%}"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Prototype uses generated/scored experimental transaction data. "
    "It does not use Razorpay production transaction data."
)


# ============================================================
# PAGE 1 - OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Merchant Portfolio Overview")

    total_transactions = len(scored)

    high_risk = scored[
        scored["risk_decision"] == "HIGH_RISK"
    ]

    medium_risk = scored[
        scored["risk_decision"] == "MEDIUM_RISK"
    ]

    low_risk = scored[
        scored["risk_decision"] == "LOW_RISK"
    ]

    high_risk_value = high_risk["amount"].sum()
    high_risk_count = len(high_risk)
    high_risk_pct = (high_risk_count / total_transactions * 100) if total_transactions > 0 else 0

    medium_risk_value = medium_risk["amount"].sum()
    medium_risk_count = len(medium_risk)

    average_probability = scored["fraud_probability"].mean()

    # ========================================================
    # Priority Attention Section
    # ========================================================

    st.markdown("### 🚨 Priority Attention")

    if high_risk_count > 0:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "High-Risk Transactions",
                f"{high_risk_count:,}",
                f"{high_risk_pct:.1f}% of portfolio"
            )

        with col2:
            st.metric(
                "High-Risk Value at Stake",
                f"${high_risk_value:,.2f}",
            )

        with col3:
            st.metric(
                "Avg Risk Score (High-Risk)",
                f"{high_risk['fraud_probability'].mean():.1%}",
            )

        st.warning(
            f"**{high_risk_count:,} high-risk transactions** require immediate review. "
            f"These transactions represent ${high_risk_value:,.2f} in potential exposure. "
            f"Use the Risk Queue to prioritize investigation."
        )
    else:
        st.success("✅ No high-risk transactions detected.")

    st.markdown("---")

    # ========================================================
    # Portfolio Summary Section
    # ========================================================

    st.markdown("### 📊 Portfolio Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Transactions",
            f"{total_transactions:,}",
        )

    with col2:
        st.metric(
            "Medium-Risk Count",
            f"{medium_risk_count:,}",
        )

    with col3:
        st.metric(
            "Low-Risk Count",
            f"{len(low_risk):,}",
        )

    with col4:
        st.metric(
            "Avg Fraud Risk",
            f"{average_probability:.1%}",
        )

    # ========================================================
    # Risk Distribution
    # ========================================================

    st.markdown("### Risk Distribution by Level")

    risk_distribution = (
        scored["risk_decision"]
        .value_counts()
        .reindex(
            ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"],
            fill_value=0,
        )
    )

    # Rename for better display
    risk_distribution_display = risk_distribution.copy()
    risk_distribution_display.index = ["Low Risk", "Medium Risk", "High Risk"]

    st.bar_chart(
        risk_distribution_display,
        use_container_width=True,
    )

    st.caption("Number of transactions by risk level classification.")

    # ========================================================
    # Top Risk Transactions
    # ========================================================

    st.markdown("### Top 10 Highest-Risk Transactions")

    top_columns = [
        column
        for column in [
            "transaction_id",
            "amount",
            "fraud_probability",
            "risk_decision",
            "risk_action",
        ]
        if column in scored.columns
    ]

    top_transactions = (
        scored
        .nlargest(10, "fraud_probability")
        [top_columns]
        .copy()
    )

    # Rename columns for display
    display_df = top_transactions.copy()
    display_df.columns = ["Transaction ID", "Amount", "Fraud Risk", "Risk Level", "Recommendation"]
    display_df["Fraud Risk"] = display_df["Fraud Risk"].apply(lambda x: f"{x:.1%}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Click Risk Queue tab to investigate these transactions.")


# ============================================================
# PAGE 2 - RISK QUEUE
# ============================================================

elif page == "Risk Queue":

    st.header("Risk Queue")

    st.caption(
        "Transactions requiring merchant review, prioritized by fraud risk."
    )

    queue = scored[
        scored["risk_action"].isin(
            ["REVIEW", "HOLD_FOR_VERIFICATION"]
        )
    ].copy()

    # ========================================================
    # Filters
    # ========================================================

    st.markdown("### Filter Transactions")

    col1, col2, col3 = st.columns(3)

    with col1:
        risk_filter = st.multiselect(
            "Risk Level",
            options=[
                "HIGH_RISK",
                "MEDIUM_RISK",
                "LOW_RISK",
            ],
            default=[
                "HIGH_RISK",
                "MEDIUM_RISK",
            ],
            help="Filter by AI-assigned risk level"
        )

    if risk_filter:
        queue = queue[
            queue["risk_decision"].isin(risk_filter)
        ]

    with col2:
        if "amount" in queue.columns and not queue.empty:
            min_amount = float(queue["amount"].min())
            max_amount = float(queue["amount"].max())

            amount_range = st.slider(
                "Transaction Amount ($)",
                min_value=min_amount,
                max_value=max_amount,
                value=(min_amount, max_amount),
                help="Filter by transaction amount range"
            )

            queue = queue[
                queue["amount"].between(
                    amount_range[0],
                    amount_range[1],
                )
            ]

    with col3:
        if "risk_action" in queue.columns:
            action_filter = st.multiselect(
                "Recommendation",
                options=[
                    "REVIEW",
                    "HOLD_FOR_VERIFICATION",
                ],
                default=[
                    "REVIEW",
                    "HOLD_FOR_VERIFICATION",
                ],
                help="Filter by AI recommendation type"
            )

            if action_filter:
                queue = queue[
                    queue["risk_action"].isin(action_filter)
                ]

    # ========================================================
    # Sort highest risk first (automatically)
    # ========================================================

    queue = queue.sort_values(
        "fraud_probability",
        ascending=False,
    )

    st.markdown("---")

    st.metric(
        "Transactions in Queue",
        f"{len(queue):,}",
    )

    # ========================================================
    # Queue Display - Merchant-Friendly Columns Only
    # ========================================================

    if not queue.empty:
        display_columns = [
            "transaction_id",
            "amount",
            "fraud_probability",
            "risk_decision",
            "risk_action",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in queue.columns
        ]

        queue_display = queue[
            available_columns
        ].head(100).copy()

        # Rename columns for merchant-friendly display
        queue_display_renamed = queue_display.copy()
        queue_display_renamed.columns = [
            "Transaction ID",
            "Amount",
            "Fraud Risk",
            "Risk Level",
            "Recommendation"
        ]
        queue_display_renamed["Fraud Risk"] = queue_display_renamed["Fraud Risk"].apply(lambda x: f"{x:.1%}")

        st.dataframe(
            queue_display_renamed,
            use_container_width=True,
            hide_index=True,
            height=600,
        )

        st.caption(
            f"Showing {len(queue_display):,} of "
            f"{len(queue):,} matching transactions. "
            "Click 'Investigation' tab and select a transaction ID to view details and make a decision."
        )
    else:
        st.info("No transactions match the selected filters.")


# ============================================================
# PAGE 3 - INVESTIGATION
# ============================================================

elif page == "Investigation":

    st.header("Transaction Investigation")

    st.caption(
        "Review transaction details and make a merchant decision."
    )

    transaction_ids = scored[
        "transaction_id"
    ].astype(str).tolist()

    selected_id = st.selectbox(
        "Select Transaction to Review",
        transaction_ids,
        help="Choose a transaction ID to investigate in detail"
    )

    selected_rows = scored[
        scored["transaction_id"].astype(str)
        == selected_id
    ]

    if selected_rows.empty:

        st.warning(
            "The selected transaction could not be found."
        )

    else:

        transaction = selected_rows.iloc[0]

        st.subheader(
            f"Transaction: {selected_id}"
        )

        # ====================================================
        # Core Transaction Facts
        # ====================================================

        st.markdown("### Transaction Details")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Amount",
                f"${transaction.get('amount', 0):,.2f}",
            )

        with col2:
            st.metric(
                "Risk Level",
                transaction["risk_decision"].replace("_", " "),
            )

        with col3:
            st.metric(
                "Fraud Risk Score",
                f"{transaction['fraud_probability']:.1%}",
            )

        with col4:
            action_label = {
                "ALLOW": "Allow",
                "REVIEW": "Review",
                "HOLD_FOR_VERIFICATION": "Hold for Verification"
            }.get(transaction["risk_action"], transaction["risk_action"])
            
            st.metric(
                "AI Recommendation",
                action_label,
            )

        # ====================================================
        # Timestamp and Payment Method
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:
            if "timestamp" in transaction.index:
                st.write(
                    f"**Transaction Time:** {transaction['timestamp']}"
                )

        with col2:
            if "payment_method" in transaction.index:
                st.write(
                    f"**Payment Method:** {transaction['payment_method'].title() if transaction['payment_method'] else 'Unknown'}"
                )

        st.markdown("---")

        # ====================================================
        # AI Explanation (Merchant-Friendly)
        # ====================================================

        st.markdown("### Why Did AI Recommend This?")

        explanation_summary = transaction.get(
            "explanation_summary",
            None,
        )

        if explanation_summary:
            st.info(str(explanation_summary))
        else:
            st.warning("Detailed explanation is unavailable for this transaction.")

        # ====================================================
        # Top Risk Factors
        # ====================================================

        contributors = transaction.get(
            "top_contributors",
            None,
        )

        if contributors:
            st.markdown("### Risk Factors Analysis")

            risk_increasing = [
                c for c in contributors
                if c.get("direction") == "INCREASES_FRAUD_RISK"
            ]

            risk_decreasing = [
                c for c in contributors
                if c.get("direction") == "DECREASES_FRAUD_RISK"
            ]

            if risk_increasing:
                st.markdown("#### 📈 Factors Increasing Fraud Risk")
                for contributor in risk_increasing:
                    display_name = contributor.get(
                        "display_name",
                        "Risk factor",
                    )
                    st.write(f"• {display_name}")

            if risk_decreasing:
                st.markdown("#### 📉 Protective Factors (Reducing Risk)")
                for contributor in risk_decreasing:
                    display_name = contributor.get(
                        "display_name",
                        "Protective factor",
                    )
                    st.write(f"• {display_name}")

            # Optional: Expanded technical details
            with st.expander("View Detailed Feature Contributions (Technical)"):
                for index, contributor in enumerate(contributors, start=1):
                    display_name = contributor.get(
                        "display_name",
                        "Risk factor",
                    )

                    direction = contributor.get(
                        "direction",
                        "",
                    )

                    if direction == "INCREASES_FRAUD_RISK":
                        direction_text = "↑ Increases fraud risk"
                    else:
                        direction_text = "↓ Reduces fraud risk"

                    st.write(
                        f"**{index}. {display_name}**"
                    )

                    st.caption(
                        f"{direction_text} | "
                        f"Contribution: {float(contributor.get('magnitude', 0)):.4f}"
                    )

        st.markdown("---")

        # ====================================================
        # Transaction Context (Optional Expanded View)
        # ====================================================

        with st.expander("View Transaction Context & History (Optional)"):
            st.markdown("#### Transaction Context")

            context_columns = [
                "amount",
                "timestamp",
                "payment_method",
                "account_age_days",
                "failed_attempts",
                "customer_txn_count_before",
                "customer_amount_mean_before",
                "amount_vs_customer_mean",
                "customer_amount_zscore",
                "customer_txn_count_10m",
                "customer_txn_count_1h",
                "customer_txn_count_24h",
                "merchant_txn_count_before",
                "merchant_amount_mean_before",
                "merchant_txn_count_1h",
            ]

            available_context = [
                column
                for column in context_columns
                if column in transaction.index
            ]

            context_data = pd.DataFrame(
                {
                    "Field": available_context,
                    "Value": [
                        str(transaction[column])
                        for column in available_context
                    ],
                }
            )

            st.dataframe(
                context_data,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")

        # ====================================================
        # Merchant Decision Section
        # ====================================================

        st.markdown("### Your Merchant Decision")

        st.info(
            f"**AI Recommendation:** {action_label} "
            f"(Fraud Risk: {transaction['fraud_probability']:.1%})"
        )

        st.markdown(
            "Based on your investigation, what should we do with this transaction?"
        )

        merchant_action = st.radio(
            "Your Decision",
            [
                "APPROVE",
                "HOLD",
                "ESCALATE",
            ],
            horizontal=True,
            key=f"merchant_action_{selected_id}",
            help="APPROVE: Let it through. HOLD: Pause for verification. ESCALATE: Flag for urgent review."
        )

        decision_saved = False

        if st.button(
            "Save Your Decision",
            type="primary",
            use_container_width=True,
        ):

            saved = save_decision(
                transaction_id=selected_id,
                ai_recommendation=transaction[
                    "risk_action"
                ],
                merchant_action=merchant_action,
                risk_score=float(
                    transaction["fraud_probability"]
                ),
                risk_level=transaction[
                    "risk_decision"
                ],
            )

            if saved:
                decision_saved = True
                st.success(
                    f"✅ Your decision '{merchant_action}' has been recorded for transaction {selected_id}."
                )

            else:

                st.warning(
                    f"A decision has already been recorded for transaction {selected_id}. "
                    f"Each transaction should be decided only once."
                )


# ============================================================
# PAGE 4 - DECISIONS LOG
# ============================================================

elif page == "Decisions Log":

    st.header("Merchant Decision Log")

    st.caption(
        "Local record of merchant decisions made through the AegisRisk dashboard. "
        "Not production-grade audit storage."
    )

    decision_log = load_decision_log()

    if decision_log.empty:

        st.info(
            "No merchant decisions have been logged yet. "
            "Visit the Investigation tab to review transactions and make decisions."
        )

    else:

        # ========================================================
        # Summary Statistics
        # ========================================================

        st.markdown("### Decision Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Decisions Logged",
                f"{len(decision_log):,}",
            )

        with col2:
            if "merchant_action" in decision_log.columns:
                unique_txns = decision_log["transaction_id"].nunique()
                st.metric(
                    "Unique Transactions",
                    f"{unique_txns:,}",
                )

        with col3:
            if "risk_level" in decision_log.columns:
                high_risk_decided = len(
                    decision_log[decision_log["risk_level"] == "HIGH_RISK"]
                )
                st.metric(
                    "High-Risk Decided",
                    f"{high_risk_decided:,}",
                )

        # ========================================================
        # Merchant Action Distribution
        # ========================================================

        st.markdown("### Decisions by Your Action")

        if "merchant_action" in decision_log.columns:

            action_counts = (
                decision_log["merchant_action"]
                .value_counts()
            )

            # Create display with better labels
            action_labels = {
                "APPROVE": "Approve",
                "HOLD": "Hold",
                "ESCALATE": "Escalate"
            }

            action_counts_display = action_counts.copy()
            action_counts_display.index = [
                action_labels.get(action, action)
                for action in action_counts_display.index
            ]

            st.bar_chart(
                action_counts_display,
                use_container_width=True,
            )

        # ========================================================
        # Full Decision Log Table
        # ========================================================

        st.markdown("### Complete Decision Log")

        # Rename columns for display
        display_log = decision_log.copy()
        if "merchant_action_timestamp" in display_log.columns:
            display_log = display_log.rename(
                columns={
                    "transaction_id": "Transaction ID",
                    "ai_recommendation": "AI Recommendation",
                    "merchant_action": "Your Action",
                    "merchant_action_timestamp": "Timestamp",
                    "risk_score": "Risk Score",
                    "risk_level": "Risk Level",
                }
            )

        # Format risk score as percentage
        if "Risk Score" in display_log.columns:
            display_log["Risk Score"] = display_log["Risk Score"].apply(lambda x: f"{x:.1%}")

        # Format action labels
        if "Your Action" in display_log.columns:
            action_labels = {
                "APPROVE": "Approve",
                "HOLD": "Hold",
                "ESCALATE": "Escalate"
            }
            display_log["Your Action"] = display_log["Your Action"].apply(
                lambda x: action_labels.get(x, x)
            )

        # Format AI recommendation labels
        if "AI Recommendation" in display_log.columns:
            action_labels = {
                "ALLOW": "Allow",
                "REVIEW": "Review",
                "HOLD_FOR_VERIFICATION": "Hold for Verification"
            }
            display_log["AI Recommendation"] = display_log["AI Recommendation"].apply(
                lambda x: action_labels.get(x, x)
            )

        st.dataframe(
            display_log,
            use_container_width=True,
            hide_index=True,
            height=600,
        )

        st.caption(
            f"Showing all {len(decision_log)} logged decisions."
        )


# ============================================================
# PAGE 5 - SYSTEM STATUS
# ============================================================

elif page == "System Status":

    st.header("🔧 System Status (Developer/Operations)")

    st.markdown(
        "This page displays system health, model status, and configuration verification. "
        "For merchants: contact support. For developers/ops: verify system components below."
    )

    st.markdown("---")

    # ========================================================
    # Frozen Model Status
    # ========================================================

    st.subheader("1. Model Verification")

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            f"**Model Type:** {metrics.get('model_type', 'Logistic Regression')}"
        )
        st.write(
            f"**Model Path:** {metrics.get('model_path', 'N/A')}"
        )

    with col2:
        st.write(
            f"**Input Features:** {metrics.get('input_features', 'N/A')}"
        )
        st.write(
            f"**Status:** ✅ Loaded and verified"
        )

    st.markdown("---")

    # ========================================================
    # Decision Policy Status
    # ========================================================

    st.subheader("2. Decision Policy")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(
            f"**Review Threshold:** "
            f"{metrics.get('policy_review_threshold', 0.35):.0%}"
        )

    with col2:
        st.write(
            f"**Hold Threshold:** "
            f"{metrics.get('policy_hold_threshold', 0.70):.0%}"
        )

    with col3:
        st.write(
            f"**Actions:** ALLOW, REVIEW, HOLD_FOR_VERIFICATION"
        )

    st.markdown("---")

    # ========================================================
    # Data Validation Status
    # ========================================================

    st.subheader("3. Data Validation")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Rows Scored:** {len(scored):,}")
        st.write(
            f"**Fraud Probabilities Valid:** "
            f"{'✅ Yes' if probability_valid else '❌ No'}"
        )
        st.write(
            f"**Required Columns Present:** "
            f"{'✅ Yes' if len(missing_columns) == 0 else '❌ No'}"
        )

    with col2:
        st.write(
            f"**Risk Levels Valid:** "
            f"{'✅ Yes' if not invalid_risk_levels else '❌ No'}"
        )
        st.write(
            f"**Recommended Actions Valid:** "
            f"{'✅ Yes' if not invalid_actions else '❌ No'}"
        )

    if not probability_valid:
        st.error("⚠️ Invalid fraud probabilities detected.")

    if len(missing_columns) > 0:
        st.error(f"⚠️ Missing columns: {', '.join(missing_columns)}")

    if invalid_risk_levels:
        st.error(f"⚠️ Invalid risk levels: {invalid_risk_levels}")

    if invalid_actions:
        st.error(f"⚠️ Invalid actions: {invalid_actions}")

    st.markdown("---")

    # ========================================================
    # Explanation System Status
    # ========================================================

    st.subheader("4. Explanation System")

    st.write(
        "**SHAP LinearExplainer** is used for transaction-level fraud risk explanations."
    )
    st.write(
        "**Status:** ✅ Active and verified"
    )

    st.markdown("---")

    # ========================================================
    # Prototype Limitations
    # ========================================================

    st.subheader("⚠️ Prototype Limitations")

    st.warning(
        "✓ Uses generated/scored experimental transaction data.\n"
        "✓ Does NOT use Razorpay production transaction data.\n"
        "✓ Decision log is stored locally (not production audit storage).\n"
        "✓ For research and evaluation purposes only."
    )