# AegisRisk AI — Model Card

## 1. Model Name

**AegisRisk AI**

Current project status: **Foundation phase — no trained model exists yet.**

---

## 2. Intended Purpose

AegisRisk AI is intended to become a prototype payment/transaction fraud risk management system.

For each transaction, the future system will estimate fraud risk and support merchant decision-making.

The intended action framework is:

* `APPROVE`
* `REVIEW`
* `HOLD`
* `REJECT`
* `ESCALATE`

---

## 3. Model Status

As of the Day 1 foundation phase:

* No final dataset has been generated.
* No features have been implemented.
* No machine learning model has been trained.
* No model performance metrics are available.
* No production deployment is claimed.

Any future model results must be documented separately after actual training and evaluation.

---

## 4. Planned Candidate Models

The planned model comparison includes:

1. Logistic Regression
2. Random Forest
3. XGBoost

Logistic Regression is intended to serve as an interpretable baseline.

More complex models will be evaluated against the baseline rather than assumed to be better.

---

## 5. Intended Inputs

The future models may use transaction-time and historical behavioural features.

Potential categories include:

* Transaction amount
* Transaction timestamp
* Time-based features
* Historical transaction frequency
* Time since previous transaction
* Historical transaction amount patterns
* Behavioural deviations
* Other documented scenario-specific indicators

The final feature set has not yet been implemented.

---

## 6. Intended Outputs

The future system is expected to produce:

* A fraud risk score or estimated probability
* A configurable risk level
* A recommended merchant action
* Local explanations for investigated transactions
* Audit information

The exact output schema will be defined after the modelling and decision-policy stages.

---

## 7. Training Data

The future transaction-level dataset will be synthetic and generated using a research-calibrated behavioural simulation approach.

The dataset:

* Is not real Razorpay production data.
* Must not be presented as real payment-provider production data.
* Will contain simulated entities and transaction behaviour.
* May contain calibrated, assumed, and derived parameters.

Synthetic-data limitations must be considered when interpreting model results.

---

## 8. Evaluation Strategy

The intended evaluation design is chronological:

```text
Earlier transactions → Training
Later transactions   → Validation
Latest held-out period → Test
```

The exact temporal split configuration will be documented before implementation.

The future evaluation metrics include:

* Precision
* Recall
* F1 Score
* PR-AUC

A held-out chronological test set is intended to provide the final reported model evaluation.

---

## 9. Temporal Leakage Prevention

For a transaction at time **T**, the model must use only information available at or before **T**.

The model pipeline must not use:

* Future transactions
* Future fraud labels
* Post-transaction outcomes
* Future behavioural aggregates
* Any feature unavailable at transaction decision time

This requirement applies to data preparation, feature engineering, training, validation, and testing.

---

## 10. Decision Thresholds

The future system will not assume that a default classification threshold is automatically optimal.

Decision thresholds will be evaluated using validation data and a cost-aware decision framework.

Potential considerations include:

* Missed fraud
* Unnecessary review
* Legitimate transactions incorrectly held
* Legitimate transactions incorrectly rejected
* Operational investigation cost

Any cost values used without verified evidence must be explicitly labelled as assumptions.

---

## 11. Explainability

The planned explainability method is SHAP-based local explanation for transactions requiring investigation.

Explainability implementation is not part of Day 1.

Explanations should support investigation and decision understanding; they should not be interpreted as proof of causation.

---

## 12. Known Limitations

* No trained model exists during Day 1.
* No real production transaction data is used.
* Synthetic fraud patterns may not represent all real-world fraud behaviour.
* Good synthetic-data performance may not transfer to production.
* Some simulation parameters may remain assumptions.
* The prototype does not replace human judgement or operational controls.
* Model outputs should be interpreted as decision-support signals.

---

## 13. Out-of-Scope Uses

AegisRisk AI is not currently intended to:

* Provide credit scoring
* Perform AML compliance decisions
* Replace regulatory fraud controls
* Serve as a production payment authorization system
* Act as the sole basis for irreversible high-impact decisions

---

## 14. Future Updates

This model card will be updated after:

1. The simulator is implemented
2. The dataset is generated
3. Features are implemented
4. Models are trained
5. Chronological evaluation is completed
6. Threshold selection is evaluated
7. Explainability is implemented
8. Limitations are reassessed using actual results
