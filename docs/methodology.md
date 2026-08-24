# AegisRisk AI — Methodology

## 1. Methodology Overview

AegisRisk AI is designed as a temporal, explainable, and cost-aware payment/transaction fraud risk management prototype.

The planned methodology follows this high-level workflow:

1. Define the fraud-risk decision problem
2. Configure the synthetic temporal simulation
3. Generate transaction behaviour over time
4. Create behavioural features using only information available at transaction time
5. Split data chronologically into training, validation, and test periods
6. Establish a Logistic Regression baseline
7. Evaluate tree-based models
8. Compare models using fraud-appropriate metrics
9. Select decision thresholds using a cost-aware framework
10. Explain investigated transactions
11. Monitor temporal changes in fraud risk
12. Map risk outputs to merchant actions
13. Record decisions for auditability

Day 1 establishes the configuration, documentation, and architecture required before these later stages are implemented.

---

## 2. Phase 1 — Problem Definition and Scope

The primary loss class is **payment/transaction fraud risk**.

The system is intended to help merchants make transaction-level decisions using the following action framework:

* `APPROVE`
* `REVIEW`
* `HOLD`
* `REJECT`
* `ESCALATE`

The initial project does not build separate full models for:

* Return fraud
* Chargeback fraud
* Account takeover
* Other unrelated loss classes

The architecture may remain extensible, but the modelling scope remains focused.

---

## 3. Phase 2 — Research-Calibrated Temporal Simulation

A synthetic transaction dataset will be generated in a future implementation phase.

Simulation parameters will be classified as:

* `calibrated` — supported by verified public evidence
* `assumed` — explicit design assumptions
* `derived` — calculated from other documented parameters

The simulation configuration will be stored separately from implementation code.

The dataset must never be described as real Razorpay production data.

---

## 4. Phase 3 — Temporal Transaction Generation

The future simulator will generate transactions over an ordered time period.

The design is expected to represent:

* Repeated customer behaviour
* Merchant activity
* Transaction timestamps
* Transaction amounts
* Changes in behaviour over time
* Historical activity
* Configurable fraud scenarios
* Temporally concentrated suspicious activity

The final generator is intentionally outside the Day 1 scope.

---

## 5. Phase 4 — Behavioural Feature Engineering

Features will be constructed from information available at the transaction decision time.

Potential feature categories include:

### Transaction Features

* Transaction amount
* Transaction time
* Time-based patterns
* Merchant-related attributes

### Historical Behavioural Features

* Number of previous transactions
* Historical transaction frequency
* Time since previous transaction
* Historical amount statistics
* Deviation from previous behaviour

### Scenario Features

Scenario-specific indicators may be generated where justified by the documented simulation design.

The exact feature definitions will be finalized before model training.

---

## 6. Temporal Leakage Prevention

Temporal leakage prevention is a mandatory requirement.

For a transaction at time **T**, every predictive feature must be constructed using only information available at or before **T**.

The pipeline must not use:

* Future transactions
* Future fraud labels
* Post-transaction outcomes
* Future behavioural aggregates
* Future-derived statistics

This rule applies to feature engineering, data preparation, evaluation, and model development.

---

## 7. Phase 5 — Chronological Evaluation

The future dataset will be divided according to time rather than using a random split as the primary evaluation method.

The intended ordering is:

```text
Earlier period → Training
Middle period  → Validation
Later period   → Test
```

The purpose is to simulate a more realistic deployment setting in which the model is trained using historical information and evaluated on later transactions.

The exact split proportions will be selected and documented before model implementation.

---

## 8. Phase 6 — Baseline Model

The first planned model is:

### Logistic Regression

This model will provide an interpretable baseline against which more complex models can be compared.

The baseline will help answer whether increased model complexity provides meaningful improvement.

Model implementation is not part of Day 1.

---

## 9. Phase 7 — Candidate Advanced Models

The planned candidate models are:

### Random Forest

A tree-based ensemble model intended for comparison with the baseline.

### XGBoost

A gradient-boosted tree model intended for comparison with Logistic Regression and Random Forest.

These models will not be implemented during Day 1.

---

## 10. Phase 8 — Model Evaluation

The planned evaluation metrics are:

* Precision
* Recall
* F1 Score
* Precision-Recall AUC (PR-AUC)

Accuracy alone is not intended to be the primary evaluation metric because fraud-risk datasets may have class imbalance.

The final model comparison will use results from a held-out chronological test period.

---

## 11. Phase 9 — Cost-Aware Decision Thresholds

A fraud probability or risk score alone does not directly define the best merchant action.

The future system will evaluate decision thresholds using a configurable cost-aware framework.

The framework may consider:

* Cost of missed fraud
* Cost of unnecessary review
* Cost of holding legitimate transactions
* Cost of rejecting legitimate transactions
* Operational investigation burden

Specific monetary values will not be presented as real-world facts unless supported by appropriate evidence or explicitly labelled as assumptions.

---

## 12. Phase 10 — Explainability

For transactions requiring investigation, the future prototype plans to provide local explanations.

The planned explainability approach is:

* SHAP-based feature contribution analysis

Explanations are intended to help users understand which model inputs contributed to a particular risk assessment.

SHAP implementation is outside the Day 1 scope.

---

## 13. Phase 11 — Temporal Monitoring

The future system will monitor changes in fraud-risk behaviour over time.

Potential monitoring outputs include:

* Risk score trends
* Fraud-label trends within the synthetic evaluation environment
* Transaction volume changes
* Scenario-related spikes
* Changes in behavioural patterns

Monitoring is intended to identify temporal changes rather than assume that fraud behaviour remains constant.

---

## 14. Phase 12 — Merchant Decision Workflow

The final workflow is intended to map model outputs and configured decision policies into merchant actions.

Conceptually:

```text
Transaction
    ↓
Transaction-Time Features
    ↓
Fraud Risk Score
    ↓
Risk Level
    ↓
Decision Policy
    ↓
APPROVE / REVIEW / HOLD / REJECT / ESCALATE
```

The exact policy and threshold mapping will be validated in a later phase.

---

## 15. Phase 13 — Audit Trail

The future system should record relevant decision information for prototype auditability.

Potential audit fields include:

* Transaction identifier
* Decision timestamp
* Model version
* Configuration version
* Risk score
* Risk level
* Recommended action
* Important explanatory factors, where applicable

The audit trail must distinguish simulated information from any claims about real production systems.

---

## 16. Reproducibility

The project will support reproducibility through:

* Configurable simulation seed
* Version-controlled source code
* Separate configuration files
* Documented assumptions
* Parameter classification
* Chronological evaluation methodology
* Recorded configuration versions

A reproducible result should be traceable to the relevant code, configuration, seed, and software environment.

---

## 17. Day 1 Scope Boundary

Day 1 establishes the foundation only.

Completed or planned foundation activities include:

* Repository architecture
* Simulation configuration structure
* Parameter governance
* Problem definition
* Data strategy
* Methodology documentation
* Leakage-prevention requirements

The following are explicitly deferred:

* Final dataset generation
* Feature implementation
* ML model training
* Model comparison
* SHAP implementation
* Streamlit dashboard
* Final merchant workflow implementation
* Monitoring implementation
* Audit trail implementation

This separation is intentional to avoid premature implementation before the data and evaluation foundation is defensible.
