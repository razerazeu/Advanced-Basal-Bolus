# Multi-Agent RL Insulin Delivery (Simglucose)

This repository contains a minimal but complete research prototype for:

Multi-Agent Reinforcement Learning for Optimizing Insulin Delivery with a G2P2C-inspired frozen planning layer.

## Architecture

- Layer 1: Two independent SAC agents
	- Basal agent (slow timescale, one action at 07:00, held for 24h)
	- Bolus agent (fast timescale, one action every 15 minutes)
- Layer 2: Frozen predictive safety planner
	- Scores short-horizon trajectories using lightweight linear glucose prediction
	- Selects safe candidate action and clips unsafe bolus
- Layer 3: Centralized critic (training only)
	- Observes joint state-action for value estimation only
	- Not used during execution-time action generation

## Project Structure

```text
project/
├── env/
│   ├── simglucose_wrapper.py
│   └── scenarios.py
├── agents/
│   ├── sac_base.py
│   ├── sac_basal.py
│   └── sac_bolus.py
├── planner/
│   └── planner.py
├── training/
│   ├── rewards.py
│   ├── train_basal.py
│   └── train_bolus.py
├── evaluation/
│   └── evaluate.py
├── utils/
│   ├── replay_buffer.py
│   └── normalizer.py
├── checkpoints/
├── logs/
├── requirements.txt
└── main.py
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py --scenario A --patients 3 --phase all
```

## Notes

- Adult patients only are used (`adult#001` to `adult#010`).
- Default patient count is 3 and is configurable with `--patients`.
- Training phases are:
	- Phase 1: Basal pre-training (`200` episodes, Scenario A)
	- Phase 2: Bolus training (`300` episodes, Scenario A then B)
	- Phase 3: Combined fine-tuning with planner (`100` episodes, Scenarios A/B/C)
- Episode logs are saved as JSONL files in `logs/`.
- Model checkpoints are saved in `checkpoints/`.

## Expected Early Sanity (around episode 10, adult#001, Scenario A)

- TIR > 50%
- min_G > 55
- planner_overrides > 0