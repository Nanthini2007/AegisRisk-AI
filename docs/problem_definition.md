# AegisRisk AI — Problem Definition

## 1. Business Problem

Merchants need to make fast and defensible decisions when a payment transaction appears risky. A poor decision can create financial loss, operational cost, and customer friction:

* Approving a fraudulent transaction may result in financial loss.
* Holding or reviewing a legitimate transaction may create unnecessary friction.
* Rejecting a legitimate transaction may reduce customer trust and revenue.
* Investigating every transaction manually is not operationally scalable.

AegisRisk AI is a proposed fraud risk management prototype designed to support merchant decision-making at the transaction level.

The primary scope of this project is **payment/transaction fraud risk**.

The project does not claim to use or represent real Razorpay production transaction data.

---

## 2. Target User

The primary target users are:

* Merchant risk teams
* Fraud and operations analysts
* Payment operations teams
* Risk managers

The system is intended to provide decision support rather than replace all human judgement.

---

## 3. Transaction-Level Prediction Objective

For each transaction at time **T**, the future system will estimate the transaction's fraud risk using only information available at or before time **T**.

The prediction objective is:

> Estimate the likelihood or risk level of payment/transaction fraud and recommend an appropriate merchant action.

The initial action framework supports:

* `APPROVE`
* `REVIEW`
* `HOLD`
* `REJECT`
* `ESCALATE`

The exact policy mapping from model risk score to these actions will be designed and validated in later project phases.

---

## 4. Expected Inputs

The final prototype may use transaction-time information and behavioural features derived from historical information available before or at the current transaction time.

Potential input categories include:

* Transaction amount
* Transaction timestamp
* Merchant identifier or merchant segment
* Customer identifier represented using privacy-safe synthetic identifiers
* Historical transaction behaviour available before time T
* Transaction frequency over defined historical windows
* Time since previous transaction
* Historical amount patterns
* Behavioural deviations from prior activity
* Scenario-specific risk indicators

The exact feature set will be defined during the feature engineering phase.

### Leakage-Prevention Requirement

No future information may be used to predict a current transaction.

For a transaction occurring at time **T**, the system must not use:

* Future transactions
* Future fraud labels
* Post-transaction outcomes
* Future aggregates
* Information that becomes available only after time **T**

---

## 5. Expected Outputs

The final system is expected to produce:

1. A transaction risk score or estimated fraud probability
2. A risk level
3. A recommended merchant action
4. An explanation of important factors for investigated transactions
5. Audit information describing the decision process

The exact output schema will be finalized in later implementation phases.

---

## 6. Risk Levels

The future system may use configurable risk levels such as:

* `LOW`
* `MEDIUM`
* `HIGH`
* `CRITICAL`

These levels are currently design categories and do not yet represent calibrated real-world thresholds.

Thresholds will be selected later using validation results and cost-aware decision analysis.

---

## 7. Merchant Actions

### APPROVE

The transaction appears to have sufficiently low estimated risk according to the configured decision policy.

### REVIEW

The transaction requires additional investigation before a final decision.

### HOLD

The transaction should be temporarily paused while additional verification or analysis is performed.

### REJECT

The transaction should not proceed according to the configured risk policy.

### ESCALATE

The transaction requires further investigation by a specialized risk, fraud, or operations process.

These actions represent the intended merchant workflow. Their final mapping to model outputs will be implemented and evaluated in later phases.

---

## 8. Success Metrics

The future fraud-risk models will be evaluated using:

* Precision
* Recall
* F1 Score
* Precision-Recall AUC (PR-AUC)

The project will also evaluate decision quality using a cost-aware framework because classification errors may have different business consequences.

Potential evaluation considerations include:

* Cost of missed fraud
* Cost of unnecessary review
* Cost of holding or rejecting legitimate transactions
* Operational investigation burden

No specific financial cost values are assumed as real-world facts during Day 1.

---

## 9. Non-Goals

The Day 1 foundation and initial project scope do not aim to build separate full models for:

* Return fraud
* Chargeback fraud
* Account takeover
* Credit underwriting
* General customer credit risk
* AML compliance
* All forms of merchant risk

The architecture may be extended in the future, but the primary project scope remains **payment/transaction fraud risk**.

---

## 10. Current Limitations

The project has the following known limitations:

* The transaction-level dataset will be synthetic.
* The dataset must not be described as real Razorpay production data.
* Simulation parameters may include calibrated, assumed, and derived values.
* Parameters without verified evidence must remain explicitly labelled as assumptions or `TODO: source required`.
* Synthetic simulation cannot guarantee representation of all real-world fraud patterns.
* Real production fraud systems may use additional proprietary data, operational signals, and verification processes unavailable to this prototype.
* Final model performance on synthetic data may not directly transfer to a real production environment.

These limitations will be documented and revisited throughout the project.
