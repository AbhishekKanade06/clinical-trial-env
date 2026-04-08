"""Core RL environment for clinical trial patient screening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from pydantic import BaseModel, Field

try:
    from .models import (
        ClinicalTrialAction,
        ClinicalTrialObservation,
        ClinicalTrialReward,
        ClinicalTrialState,
    )
except ImportError:
    from models import (
        ClinicalTrialAction,
        ClinicalTrialObservation,
        ClinicalTrialReward,
        ClinicalTrialState,
    )

INCREMENTAL_REWARD = 0.20
FINAL_REWARD = 1.00
HALLUCINATION_PENALTY = -0.50
TASK_SEQUENCE = ["easy", "medium", "hard"]
MIN_STRICT_SCORE = 0.01
MAX_STRICT_SCORE = 0.99


def _normalize(value: Optional[str]) -> str:
    return " ".join((value or "").strip().lower().replace("_", " ").split())


class GroundTruth(BaseModel):
    """Deterministic grader targets for a scenario."""

    extracted_fields: Dict[str, str] = Field(default_factory=dict)
    ranking: List[str] = Field(default_factory=list)
    final_decision: str


class ScenarioSpec(BaseModel):
    """Scenario loaded from patient_data.json."""

    task_id: str
    difficulty: str
    title: str
    instructions: str
    context: Dict[str, Any]
    ground_truth: GroundTruth
    hidden_exclusions: List[str] = Field(default_factory=list)
    max_steps: int = 6
    grader_name: str = "deterministic_json_grader"


class ClinicalTrialEnvironment(
    Environment[ClinicalTrialAction, ClinicalTrialObservation, ClinicalTrialState]
):
    """Clinical trial screening environment backed by externalized JSON scenarios."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        data_path = Path(__file__).resolve().with_name("patient_data.json")
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        task_payload = payload.get("tasks", {})
        self._scenarios: Dict[str, ScenarioSpec] = {
            task_id: ScenarioSpec.model_validate({**scenario, "task_id": task_id})
            for task_id, scenario in task_payload.items()
        }
        self._task_cursor = -1
        self._current_scenario: Optional[ScenarioSpec] = None
        self._submitted_ranking: List[str] = []
        self._state = ClinicalTrialState(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ClinicalTrialObservation:
        del seed, kwargs
        selected_task_id = task_id or self._next_task_id()
        self._current_scenario = self._scenarios[selected_task_id]
        self._submitted_ranking = []
        self._state = ClinicalTrialState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_task_id=self._current_scenario.task_id,
            difficulty=self._current_scenario.difficulty,
            title=self._current_scenario.title,
            extracted_fields={},
            identified_deviations=[],
            final_decision=None,
            grading_score=0.0,
        )
        return self._build_observation(
            reward_details=ClinicalTrialReward(notes=["Episode reset."]),
            done=False,
        )

    def step(
        self,
        action: ClinicalTrialAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ClinicalTrialObservation:
        del timeout_s, kwargs
        if self._current_scenario is None:
            return self.reset()

        self._state.step_count += 1
        reward = ClinicalTrialReward()
        done = False
        terminal_reason: Optional[str] = None

        if action.action_type == "extract_data":
            self._handle_extraction(action, reward)
        elif action.action_type == "flag_deviation":
            self._handle_deviation_flag(action, reward)
        elif action.action_type == "rank_patients":
            self._handle_ranking(action, reward)
            if action.ranking:
                done = True
                terminal_reason = "ranking_submitted"
        elif action.action_type == "submit_decision":
            self._state.final_decision = action.final_decision
            done = True
            terminal_reason = "final_decision_submitted"
        elif action.action_type == "delete_evidence":
            reward.penalty += HALLUCINATION_PENALTY
            reward.notes.append("Destructive action: deleting evidence is not allowed.")
        else:
            reward.penalty += HALLUCINATION_PENALTY
            reward.notes.append(f"Unsupported action type: {action.action_type}")

        if self._state.step_count >= self._current_scenario.max_steps and not done:
            done = True
            terminal_reason = "max_steps_reached"

        if done:
            reward.grader_score = self.grader()
            if self._is_final_submission_correct():
                reward.final_reward = FINAL_REWARD
                reward.notes.append("Correct final screening decision.")
            reward.missing_items = self._missing_items()
            self._state.grading_score = reward.grader_score

        reward.total_reward = round(
            reward.incremental_reward + reward.final_reward + reward.penalty, 4
        )
        return self._build_observation(
            reward_details=reward,
            done=done,
            terminal_reason=terminal_reason,
        )

    @property
    def state(self) -> ClinicalTrialState:
        return self._state

    def grader(self) -> float:
        """Deterministically compare agent outputs against the current scenario ground truth."""
        assert self._current_scenario is not None
        components: List[float] = []
        truth = self._current_scenario.ground_truth

        if truth.extracted_fields:
            field_hits = sum(
                1
                for field_name, expected in truth.extracted_fields.items()
                if _normalize(self._state.extracted_fields.get(field_name)) == _normalize(expected)
            )
            components.append(field_hits / len(truth.extracted_fields))

        if self._current_scenario.hidden_exclusions:
            exclusion_hits = sum(
                1
                for exclusion in self._current_scenario.hidden_exclusions
                if exclusion in self._state.identified_deviations
            )
            components.append(exclusion_hits / len(self._current_scenario.hidden_exclusions))

        if truth.ranking:
            ranking = self._submitted_ranking
            if ranking and len(ranking) == len(truth.ranking):
                positional_hits = sum(
                    1 for actual, expected in zip(ranking, truth.ranking) if actual == expected
                ) / len(truth.ranking)
                pairwise_hits = 0
                total_pairs = 0
                for index, higher in enumerate(truth.ranking):
                    for lower in truth.ranking[index + 1 :]:
                        total_pairs += 1
                        if ranking.index(higher) < ranking.index(lower):
                            pairwise_hits += 1
                pairwise_score = pairwise_hits / max(total_pairs, 1)
                components.append((0.6 * positional_hits) + (0.4 * pairwise_score))
            else:
                components.append(0.0)

        components.append(
            1.0
            if _normalize(self._state.final_decision) == _normalize(truth.final_decision)
            else 0.0
        )

        if not components:
            return MIN_STRICT_SCORE

        raw_score = sum(components) / len(components)
        strict_score = min(max(raw_score, MIN_STRICT_SCORE), MAX_STRICT_SCORE)
        return round(strict_score, 4)

    def _next_task_id(self) -> str:
        self._task_cursor = (self._task_cursor + 1) % len(TASK_SEQUENCE)
        return TASK_SEQUENCE[self._task_cursor]

    def _handle_extraction(self, action: ClinicalTrialAction, reward: ClinicalTrialReward) -> None:
        assert self._current_scenario is not None
        if not action.field_name or action.value is None:
            reward.penalty += HALLUCINATION_PENALTY
            reward.notes.append("extract_data requires field_name and value.")
            return

        expected_value = self._current_scenario.ground_truth.extracted_fields.get(action.field_name)
        if expected_value is None:
            reward.penalty += HALLUCINATION_PENALTY
            reward.notes.append(f"Hallucinated field: {action.field_name}")
            return

        if _normalize(action.value) == _normalize(expected_value):
            if action.field_name not in self._state.extracted_fields:
                reward.incremental_reward += INCREMENTAL_REWARD
                reward.matched_items.append(action.field_name)
                reward.notes.append(f"Validated extraction for {action.field_name}.")
            self._state.extracted_fields[action.field_name] = action.value
        else:
            reward.penalty += HALLUCINATION_PENALTY
            reward.notes.append(f"Incorrect value for {action.field_name}.")

    def _handle_deviation_flag(self, action: ClinicalTrialAction, reward: ClinicalTrialReward) -> None:
        assert self._current_scenario is not None
        submitted = [_normalize(item) for item in action.deviations]
        if not submitted:
            reward.penalty += HALLUCINATION_PENALTY
            reward.notes.append("flag_deviation requires at least one deviation.")
            return

        for deviation in submitted:
            if deviation in self._current_scenario.hidden_exclusions:
                if deviation not in self._state.identified_deviations:
                    self._state.identified_deviations.append(deviation)
                    reward.incremental_reward += INCREMENTAL_REWARD
                    reward.matched_items.append(deviation)
                    reward.notes.append(f"Validated deviation: {deviation}.")
            else:
                reward.penalty += HALLUCINATION_PENALTY
                reward.notes.append(f"Unsupported deviation claim: {deviation}.")

    def _handle_ranking(self, action: ClinicalTrialAction, reward: ClinicalTrialReward) -> None:
        assert self._current_scenario is not None
        ranking = action.ranking
        valid_patients = [
            patient["patient_id"]
            for patient in self._current_scenario.context.get("patients", [])
        ]
        if sorted(ranking) != sorted(valid_patients):
            reward.penalty += HALLUCINATION_PENALTY
            reward.notes.append("Ranking must include each patient exactly once.")
            return
        self._submitted_ranking = ranking
        self._state.final_decision = "ranking_submitted"

    def _is_final_submission_correct(self) -> bool:
        assert self._current_scenario is not None
        truth = self._current_scenario.ground_truth
        if truth.ranking:
            return self._submitted_ranking == truth.ranking
        return _normalize(self._state.final_decision) == _normalize(truth.final_decision)

    def _missing_items(self) -> List[str]:
        assert self._current_scenario is not None
        truth = self._current_scenario.ground_truth
        missing_fields = [
            field_name
            for field_name, expected in truth.extracted_fields.items()
            if _normalize(self._state.extracted_fields.get(field_name)) != _normalize(expected)
        ]
        missing_fields.extend(
            exclusion
            for exclusion in self._current_scenario.hidden_exclusions
            if exclusion not in self._state.identified_deviations
        )
        if truth.ranking and self._submitted_ranking != truth.ranking:
            missing_fields.append("ranking")
        if _normalize(self._state.final_decision) != _normalize(truth.final_decision):
            missing_fields.append("final_decision")
        return missing_fields

    def _build_observation(
        self,
        reward_details: ClinicalTrialReward,
        done: bool,
        terminal_reason: Optional[str] = None,
    ) -> ClinicalTrialObservation:
        assert self._current_scenario is not None
        attempts_remaining = max(self._current_scenario.max_steps - self._state.step_count, 0)
        return ClinicalTrialObservation(
            task_id=self._current_scenario.task_id,
            difficulty=self._current_scenario.difficulty,  # type: ignore[arg-type]
            title=self._current_scenario.title,
            instructions=self._current_scenario.instructions,
            context=self._current_scenario.context,
            expected_fields=list(self._current_scenario.ground_truth.extracted_fields.keys()),
            extracted_fields=dict(self._state.extracted_fields),
            identified_deviations=list(self._state.identified_deviations),
            attempts_remaining=attempts_remaining,
            grader_name=self._current_scenario.grader_name,
            reward_details=reward_details,
            reward=reward_details.total_reward,
            done=done,
            metadata={"grading_score": self._state.grading_score},
            terminal_reason=terminal_reason,
        )


class ClinicalTrialEnv(ClinicalTrialEnvironment):
    """Compatibility alias for manifest entry points expecting env:ClinicalTrialEnv."""

    pass
