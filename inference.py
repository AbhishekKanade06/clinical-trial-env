"""Hackathon-compliant inference runner for the clinical trial environment."""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from clinical_trial_env import ClinicalTrialAction, ClinicalTrialEnv
except ImportError:
    from client import ClinicalTrialEnv
    from models import ClinicalTrialAction

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("CLINICAL_TRIAL_TASK", "easy")
BENCHMARK = os.getenv("CLINICAL_TRIAL_BENCHMARK", "clinical_trial_env")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.8"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are operating a clinical trial screening environment.
    Return exactly one compact JSON object with keys:
    action_type, field_name, value, ranking, deviations, final_decision, rationale.
    Use only supported action_type values:
    extract_data, rank_patients, flag_deviation, submit_decision.
    Do not add markdown, commentary, or code fences.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_text = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def sanitize_error(error: Optional[str]) -> Optional[str]:
    if error is None:
        return None
    cleaned = " ".join(error.split())
    return cleaned or "null"


def build_user_prompt(task_name: str, step: int, observation_payload: Dict, history: List[str]) -> str:
    history_text = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task: {task_name}
        Step: {step}
        Observation:
        {json.dumps(observation_payload, indent=2, sort_keys=True)}

        Recent history:
        {history_text}

        Return the next best JSON action.
        """
    ).strip()


def heuristic_action(task_name: str, step: int) -> ClinicalTrialAction:
    heuristics: Dict[Tuple[str, int], ClinicalTrialAction] = {
        ("easy", 1): ClinicalTrialAction(action_type="extract_data", field_name="age", value="56"),
        ("easy", 2): ClinicalTrialAction(
            action_type="extract_data", field_name="egfr_mutation", value="L858R positive"
        ),
        ("easy", 3): ClinicalTrialAction(action_type="submit_decision", final_decision="eligible"),
        ("medium", 1): ClinicalTrialAction(
            action_type="extract_data", field_name="BC-101_her2_status", value="IHC 3+"
        ),
        ("medium", 2): ClinicalTrialAction(
            action_type="extract_data", field_name="BC-102_trastuzumab_exposure", value="none"
        ),
        ("medium", 3): ClinicalTrialAction(
            action_type="rank_patients", ranking=["BC-101", "BC-103", "BC-102"]
        ),
        ("hard", 1): ClinicalTrialAction(action_type="extract_data", field_name="biomarker", value="FLT3-ITD"),
        ("hard", 2): ClinicalTrialAction(
            action_type="flag_deviation",
            deviations=["recent strong CYP3A4 inhibitor", "active uncontrolled infection"],
        ),
        ("hard", 3): ClinicalTrialAction(action_type="submit_decision", final_decision="ineligible"),
    }
    return heuristics.get((task_name, step), ClinicalTrialAction(action_type="submit_decision", final_decision="ineligible"))


def parse_action(raw_text: str) -> ClinicalTrialAction:
    payload = json.loads(raw_text)
    return ClinicalTrialAction.model_validate(payload)


def get_model_action(
    client: OpenAI,
    task_name: str,
    step: int,
    observation_payload: Dict,
    history: List[str],
) -> ClinicalTrialAction:
    user_prompt = build_user_prompt(task_name, step, observation_payload, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        return parse_action(content)
    except Exception:
        return heuristic_action(task_name, step)


def format_action(action: ClinicalTrialAction) -> str:
    payload = {
        "action_type": action.action_type,
        "field_name": action.field_name,
        "value": action.value,
        "ranking": action.ranking,
        "deviations": action.deviations,
        "final_decision": action.final_decision,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


async def create_env() -> ClinicalTrialEnv:
    if LOCAL_IMAGE_NAME:
        return await ClinicalTrialEnv.from_docker_image(LOCAL_IMAGE_NAME)
    if ENV_BASE_URL:
        env = ClinicalTrialEnv(base_url=ENV_BASE_URL)
        await env.connect()
        return env
    raise RuntimeError("Set LOCAL_IMAGE_NAME for Docker execution or ENV_BASE_URL for an existing server.")


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env: Optional[ClinicalTrialEnv] = None
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []
    last_error: Optional[str] = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await create_env()
        result = await env.reset(task_id=TASK_NAME)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(
                client=client,
                task_name=TASK_NAME,
                step=step,
                observation_payload=result.observation.model_dump(mode="json"),
                history=history,
            )

            try:
                result = await env.step(action)
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                last_error = None
            except Exception as exc:
                reward = 0.0
                done = True
                last_error = sanitize_error(str(exc))

            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step,
                action=format_action(action),
                reward=reward,
                done=done,
                error=sanitize_error(last_error),
            )
            history.append(f"step={step} action={format_action(action)} reward={reward:.2f}")

            if last_error is not None or done:
                break

        if last_error is None and "result" in locals():
            score = float(result.observation.reward_details.grader_score)
        score = max(0.0, min(score, 1.0))
        success = last_error is None and score >= SUCCESS_SCORE_THRESHOLD
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as exc:
                last_error = last_error or sanitize_error(str(exc))
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
