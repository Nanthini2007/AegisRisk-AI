# AegisRisk AI — Data Strategy

## 1. Purpose

AegisRisk AI requires transaction-level data with temporal and behavioural structure in order to study payment/transaction fraud risk.

The project will use a **research-calibrated behavioural simulation approach** to create a synthetic transaction dataset in a later phase.

The synthetic dataset is intended for prototype development, experimentation, and evaluation. It must **never be described as real Razorpay production data**.

---

## 2. Why Production Payment Data Is Unavailable

Real payment transaction data can contain commercially sensitive, confidential, and privacy-sensitive information.

This project does not have access to real Razorpay production transaction data. Therefore, the prototype must not make claims that its dataset contains:

* Real Razorpay transactions
* Real Razorpay customer identities
* Real merchant identities
* Real proprietary fraud labels
* Real internal risk signals

Instead, the project will use synthetic identifiers and simulated transaction behaviour.

---

## 3. Why a Static Public Dataset Alone May Not Be Sufficient

Public fraud datasets can be useful for experimentation and benchmarking. However, a static dataset alone may not provide all the requirements needed for this project's intended temporal behavioural design.

AegisRisk AI requires the ability to represent:

* Transactions occurring over time
* Customer and merchant behavioural histories
* Changes in behaviour
* Historical aggregates available at transaction time
* Temporal fraud scenarios
* Fraud spikes
* Chronological train, validation, and test periods

A static dataset may also contain an unknown or fixed feature-generation process that cannot be adapted to enforce the project's transaction-time feature availability rules.

Therefore, the project will not rely on a static public dataset as the only data source for the behavioural simulation requirement.

---

## 4. Research-Calibrated Behavioural Simulation

The future simulator will generate transaction behaviour using three categories of parameters.

### 4.1 Calibrated Parameters

A parameter is labelled `calibrated` when its value or range is supported by verified public evidence relevant to the simulation design.

Each calibrated parameter should include:

* The parameter name
* The value or range used
* The source
* A short explanation of how the evidence was translated into the simulation

A calibrated parameter must not be used without recording its source.

---

### 4.2 Assumed Parameters

A parameter is labelled `assumed` when reliable public evidence is unavailable, insufficient, or not directly transferable to the simulation.

Assumptions must be explicitly documented and must not be presented as real-world facts.

Examples may include:

* Simulated population size
* Initial simulation duration
* Certain behavioural distribution choices
* Scenario intensity
* Prototype-specific operating assumptions

Any assumption requiring future evidence should be marked:

`TODO: source required`

---

### 4.3 Derived Parameters

A parameter is labelled `derived` when it is calculated from one or more other parameters.

Examples include:

* Simulation duration calculated from start and end dates
* A population allocation calculated from total population and segment proportions
* A scenario volume calculated from baseline activity and a configured multiplier

Derived parameters must document their calculation or dependency.

---

## 5. Parameter Governance

Every important simulation parameter should have clear metadata.

The preferred structure is conceptually:

* `value`
* `parameter_type`
* `rationale`
* `source`, when applicable
* `derivation`, when applicable

The allowed primary labels are:

* `calibrated`
* `assumed`
* `derived`

No arbitrary value should be silently presented as a verified real-world fraud statistic.

---

## 6. Temporal Simulation Strategy

The future dataset generator will model transactions across a defined time period rather than treating every transaction as an independent static row.

The simulation is expected to include:

* Ordered transaction timestamps
* Repeated behaviour for simulated customers
* Merchant-level activity
* Changes in transaction frequency over time
* Changes in transaction amounts over time
* Behavioural deviations
* Fraud scenarios occurring at specific periods

The exact implementation is intentionally deferred beyond Day 1.

---

## 7. Transaction-Time Information Rule

A core requirement is that the future ML pipeline must use only information available when the transaction decision is made.

For a transaction at time **T**:

* Historical transactions at or before **T** may be used.
* Historical aggregates must be calculated using only valid prior information.
* Future transactions must not be used.
* Future fraud labels must not be used.
* Post-transaction outcomes must not be used.
* Future aggregates must not be used.

This rule is designed to reduce temporal data leakage.

---

## 8. Fraud Scenario Strategy

The primary scope is **payment/transaction fraud risk**.

The future simulator may define multiple scenarios within this single loss class, such as different patterns of abnormal or suspicious transaction behaviour.

Potential scenarios may involve:

* Unusual transaction timing
* Unusual transaction frequency
* Sudden behavioural deviation
* Abnormal transaction amounts
* Temporally concentrated suspicious activity

These are currently simulation design categories, not claims about actual Razorpay fraud patterns.

Scenario parameters must be explicitly labelled as:

* Calibrated, when supported by verified evidence
* Assumed, when used as a documented design choice
* Derived, when calculated from other configured values

The project will not expand return fraud, chargeback fraud, or account takeover into separate full models during the Day 1 foundation phase.

---

## 9. Reproducibility Strategy

The project will support reproducibility through:

* A fixed and configurable simulation seed
* Configuration stored separately from implementation
* Version-controlled source code
* Explicit parameter labels
* Recorded simulation configuration
* Chronological evaluation design
* Documented assumptions and limitations

The exact generated dataset should be reproducible from the same code, configuration, seed, and supported software environment, subject to deterministic behaviour of the implementation.

---

## 10. Validation Strategy

The future data and modelling pipeline will include validation at multiple levels.

### Configuration Validation

* YAML configuration must parse successfully.
* Required configuration sections must exist.
* Important parameters must contain a valid parameter label.
* Derived parameters must document their dependencies.

### Simulation Validation

Before relying on generated data, the project should check:

* Required fields are present
* Timestamps are valid and ordered
* Population relationships are valid
* Scenario configuration is reproducible
* Fraud labels follow the documented simulation logic

### Temporal Validation

The future pipeline should verify that:

* Training data occurs before validation data
* Validation data occurs before test data
* Features use only information available at prediction time
* No future information leaks into current-time features

### Model Validation

Later project phases will evaluate models using:

* Precision
* Recall
* F1 Score
* PR-AUC
* Cost-aware decision analysis

Model implementation is outside the Day 1 scope.

---

## 11. Limitations

The synthetic simulation has important limitations:

* It cannot guarantee representation of real-world payment behaviour.
* It cannot reproduce proprietary Razorpay fraud detection systems.
* It may omit unknown fraud patterns.
* Public evidence may not always transfer directly to the intended simulation context.
* Some parameters may remain assumptions until better evidence is available.
* Good performance on synthetic data does not guarantee production performance.

These limitations must remain visible in the project documentation and final presentation.

---

## 12. Data Strategy Summary

AegisRisk AI will use a **research-calibrated behavioural simulation approach** because the project requires temporal transaction behaviour while real production payment data is unavailable.

The strategy is based on:

1. Publicly supported evidence where available
2. Explicit assumptions where reliable evidence is unavailable
3. Derived parameters calculated from documented inputs
4. Temporal ordering of transactions
5. Strict prevention of future-information leakage
6. Reproducible configuration and simulation design
7. Clear acknowledgement of synthetic-data limitations

The resulting dataset is a **prototype simulation artifact**, not a representation of actual Razorpay production data.
