# Dual-SAC Basal-Bolus + G2P2C-Inspired Planner (Simglucose)

This project is a research prototype for Type 1 diabetes simulation/control in Python using:

- a **basal SAC agent**,
- a **bolus SAC agent**,
- a **G2P2C-inspired model-based planner** for short-horizon safety filtering.

The planner is **not** implemented as a third RL policy and does **not** include full G2P2C PPO training. It is an inference-time predictive safety layer that evaluates candidate actions and refines/clips unsafe proposals.

## Staged Protocol

Training and evaluation follow a staged protocol aligned to the paper-inspired setup:

1. **Basal pretraining (Scenario A fasting window):**
  - scenario fixed to A,
  - episodes stop at 07:00 (fasting-only phase),
  - bolus disabled.
2. **Bolus training (Scenario B):**
  - basal policy frozen by default,
  - bolus policy updated,
  - meal timing reward window enabled.
3. **Evaluation (Scenarios A/B/C):**
  - no learning updates,
  - planner enabled by default at inference,
  - scenario-wise summaries written to checkpoint folder.

## Project Structure

```text
project/
  main.py
  config.py

  env/
    simglucose_env.py

  agents/
    sac_core.py
    basal_agent.py
    bolus_agent.py

  planner/
    g2p2c_planner.py

  controller/
    hybrid_controller.py

  utils/
    state_builder.py
    reward.py
    buffers.py
    metrics.py
    seed.py

  training/
    trainer.py
    evaluator.py

  tests/
    test_state_builder.py
    test_planner.py
    test_controller.py
    test_env_scenarios.py

  requirements.txt
  README.md
```

## Setup

```bash
cd project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Training

```bash
python main.py --mode train --basal-episodes 10 --bolus-episodes 10
```

Optional: force the lightweight fallback environment for quick debugging/tests:

```bash
python main.py --mode train --basal-episodes 2 --bolus-episodes 2 --use-mock-env
```

Run only Phase 1 (basal pretraining):

```bash
python main.py --mode pretrain-basal --basal-episodes 10
```

Run only Phase 2 (bolus training) with a pretrained basal checkpoint:

```bash
python main.py --mode train-bolus --bolus-episodes 10 --basal-checkpoint outputs/<run>/checkpoints/basal_latest.pt
```

## Run Evaluation

```bash
python main.py --mode eval --checkpoint-dir outputs/<your_run> --scenario all --eval-episodes 3
```

Optional flags:

- `--disable-planner-eval`: bypass planner during evaluation.
- `--fixed-eval-seed`: use a fixed seed each episode for deterministic replay.
- `--scenario A|B|C`: evaluate a single scenario.

Evaluation writes `eval_summary.json` under `<checkpoint-dir>/checkpoints/`.

## Tests

```bash
pytest -q
```

Scenario behavior tests are included in `tests/test_env_scenarios.py`.

## Notes on Design Choices and Simplifications

- **Controller API:** Main integration happens in `HybridController.policy(...)`, which returns `Action(basal, bolus)`.
- **State representation:** A sliding window of CGM, insulin, and meal history is used as a fixed-size state vector.
- **Basal reward:** Implemented from the provided episode formula using a binary in-range indicator for \(r^{(i)}(a,b)\).
- **Bolus reward:** Implemented exactly as specified with glucose + action terms.
- **Planner dynamics model:** Uses a transparent surrogate one-step model with meal absorption and insulin effects for short-horizon rollout.
- **Simglucose fallback:** If `simglucose` is unavailable (or `--use-mock-env` is passed), a deterministic mock environment is used so tests and scaffolding remain runnable.
