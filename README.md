This project implements an industry-grade toxic / abusive content detection pipeline that goes beyond static machine learning models.
The system automatically detects novel (out-of-distribution) language, discovers new open-source datasets, and re-trains itself without human intervention.

Key Capabilities

XGBoost-based content classification

Outputs: CLEAN, BORDERLINE, ABUSIVE

Out-of-Distribution (OOD) detection

Identifies novel or unseen language patterns

Autonomous open-source data discovery

Uses GitHub Search 

No hardcoded file paths

Self-updating training pipeline

Retrains only when meaningful new data is found

Negative caching & idempotency

Repositories are processed once and never retried

MongoDB memory layer

Stores novel inputs and processed repositories

Cron-safe & production-ready

Safe to run repeatedly without duplication

 ------------------OOD (Out-of-Distribution) Detection--------------------------

OOD detection flags inputs that fall outside the learned feature distribution using:

Per-feature z-score analysis

Feature-space norm deviation

When OOD is detected:

Input is stored in MongoDB

Used later to trigger autonomous retraining
--------------------Autonomous Data Discovery (No Manual Selection)------------------

OOD tokens are extracted automatically

GitHub search queries are generated dynamically

GitHub repositories are discovered

GitHub Contents API scans repository structure

Dataset files are identified (.txt, .csv, .json, .tsv)

Data is parsed and deduplicated

Model retrains only if performance improves

This removes all hardcoded dataset assumptions.

📌 Ideal Use Cases

Social media moderation

Content safety research

ML system design portfolios

Autonomous learning pipelines

Continual learning experiments
