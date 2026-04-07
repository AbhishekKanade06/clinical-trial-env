---
title: Clinical Trial Patient Screening Environment
emoji: 🧪
colorFrom: teal
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - healthcare
  - reinforcement-learning
---

# Clinical Trial Patient Screening Environment

OpenEnv environment for clinical trial patient screening with three deterministic healthcare tasks:

- `easy`: binary eligibility screening against 5 criteria
- `medium`: ranking 3 patients by protocol fit
- `hard`: protocol deviation and exclusion detection from unstructured clinical text

This environment is designed as a real-world screening workflow rather than a toy game. It uses typed Pydantic models, deterministic programmatic graders, incremental reward shaping for correct data extraction, and terminal rewards for correct screening outcomes.

## Task Overview

### Easy
EGFR-mutated metastatic NSCLC eligibility check using structured oncology data.

### Medium
Rank 3 HER2-positive metastatic breast cancer candidates by fit for a trial.

### Hard
Detect subtle exclusions from an AML screening note, including protocol deviations such as recent investigational treatment, active infection, QTc prolongation, and CYP3A4 inhibitor exposure.

## Reward Design

- `+0.20` for each correct clinical data point or valid deviation extracted
- `+1.00` for a correct final screening decision
- `-0.50` for hallucinated fields, invalid deviation claims, or destructive actions

Each task also produces a deterministic grader score in `[0.0, 1.0]`.

## Project Structure

```text
clinical_trial_env/
├── .env
├── Dockerfile
├── README.md
├── __init__.py
├── client.py
├── env.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
└── server/
    ├── __init__.py
    ├── app.py
    └── clinical_trial_env_environment.py
```

## Build And Run

Build the container from the project root:

```bash
docker build -t clinical-trial-env:latest .
```

Run the server locally:

```bash
docker run --rm -p 8000:8000 clinical-trial-env:latest
```

Validate the environment:

```bash
openenv validate .
```

## Inference

The root [inference.py](/Users/abhishekkanade/Desktop/Hackathon/OpenEnv/clinical_trial_env/inference.py) uses the OpenAI client and emits exactly:

- `[START]`
- `[STEP]`
- `[END]`

Required environment variables:

- `HF_TOKEN`
- `LOCAL_IMAGE_NAME` or `ENV_BASE_URL`
- `API_BASE_URL` optional, defaults to Hugging Face router
- `MODEL_NAME` optional
- `CLINICAL_TRIAL_TASK` with values `easy`, `medium`, or `hard`

Example:

```bash
set -a
source .env
set +a
python3 inference.py
```

## Python Usage

```python
import asyncio

from clinical_trial_env import ClinicalTrialAction, ClinicalTrialEnv


async def main() -> None:
    env = await ClinicalTrialEnv.from_docker_image("clinical-trial-env:latest")
    try:
        result = await env.reset(task_id="easy")
        result = await env.step(
            ClinicalTrialAction(action_type="extract_data", field_name="age", value="56")
        )
        print(result.reward, result.observation.reward_details.grader_score)
    finally:
        await env.close()


asyncio.run(main())
```

## Audit Notes

- `reset()`, `step(action)`, and OpenEnv `state` access are implemented in [env.py](/Users/abhishekkanade/Desktop/Hackathon/OpenEnv/clinical_trial_env/env.py).
- Gold answers are not exposed through observation metadata.
- Graders are deterministic and bounded in `[0.0, 1.0]`.
- The hard task uses unstructured medical text and clinically realistic exclusion criteria.
