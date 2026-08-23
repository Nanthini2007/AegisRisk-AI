 # ReturnShield AI

**Explainable, cost-aware return-related merchant loss detection for e-commerce.**

Built for the Razorpay AI Buildathon — Track 02: AI Risk Manager.

## Problem

> "Build a working detector, verifier or auto-responder for one class of loss,
> with measured precision and recall on a held-out test set."

## Chosen Task

Predict whether a new e-commerce order will later result in a **return-related
merchant loss** — a single, binary, held-out-evaluated classification task.

This system detects a **behavioural loss pattern**, not criminal intent. Every
output is a risk score or flag for human review; the system never takes
autonomous action against a customer (defense-only, per Track 02's bar).

## Status

🚧 **Day 1 — project scaffold + data simulation.**

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
