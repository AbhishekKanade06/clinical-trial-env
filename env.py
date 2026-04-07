"""Core RL environment for clinical trial patient screening."""

from __future__ import annotations

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


def _normalize(value: Optional[str]) -> str:
    return " ".join((value or "").strip().lower().replace("_", " ").split())


class TaskSpec(BaseModel):
    """Deterministic task configuration."""

    task_id: str
    difficulty: str
    title: str
    instructions: str
    context: Dict[str, Any]
    expected_fields: Dict[str, str]
    valid_deviations: List[str] = Field(default_factory=list)
    gold_ranking: List[str] = Field(default_factory=list)
    gold_final_decision: str
    max_steps: int = 6
    grader_name: str


TASKS: Dict[str, TaskSpec] = {
    "easy": TaskSpec(
        task_id="easy",
        difficulty="easy",
        title="EGFR-Mutated Metastatic NSCLC Eligibility",
        instructions=(
            "Review the structured oncology chart and determine whether the patient meets the "
            "five binary eligibility criteria for a first-line EGFR inhibitor trial. "
            "Use extract_data for clinical facts and submit_decision with eligible or ineligible."
        ),
        context={
            "trial_protocol": {
                "phase": "Phase III",
                "disease": "Metastatic non-small cell lung cancer",
                "binary_criteria": [
                    "Age >= 18 years",
                    "Pathologically confirmed metastatic NSCLC",
                    "Activating EGFR mutation present",
                    "ECOG performance status 0 or 1",
                    "No prior EGFR-targeted therapy",
                ],
            },
            "patient": {
                "patient_id": "NSCLC-204",
                "age": 56,
                "sex": "female",
                "diagnosis": "Stage IV lung adenocarcinoma with pleural metastases",
                "molecular_profile": "EGFR L858R positive; ALK negative; ROS1 negative",
                "ecog": 1,
                "prior_therapy": "Carboplatin/pemetrexed deferred; no prior EGFR TKI exposure",
                "recent_mri": "No untreated brain metastases",
            },
        },
        expected_fields={
            "age": "56",
            "diagnosis": "metastatic nsclc",
            "egfr_mutation": "l858r positive",
            "ecog": "1",
            "prior_egfr_tki": "none",
        },
        gold_final_decision="eligible",
        grader_name="grade_easy_eligibility",
    ),
    "medium": TaskSpec(
        task_id="medium",
        difficulty="medium",
        title="Rank Patients for HER2-Positive Breast Cancer Trial",
        instructions=(
            "Rank the three candidates from best to worst fit for a second-line HER2-positive "
            "metastatic breast cancer antibody-drug conjugate trial. Extract key biomarkers or "
            "performance status if useful, then submit the full ranking with rank_patients."
        ),
        context={
            "trial_protocol": {
                "phase": "Phase II",
                "disease": "HER2-positive metastatic breast cancer",
                "fit_score_drivers": [
                    "HER2 IHC 3+ or ISH amplified",
                    "Prior trastuzumab and taxane exposure required",
                    "No active CNS progression",
                    "ECOG 0-1 preferred",
                    "Adequate left ventricular ejection fraction",
                ],
            },
            "patients": [
                {
                    "patient_id": "BC-101",
                    "age": 48,
                    "her2_status": "IHC 3+",
                    "prior_therapy": "Taxane, trastuzumab, pertuzumab completed",
                    "ecog": 0,
                    "brain_mri": "Stable treated cerebellar lesion; no active CNS progression",
                    "lvef": "61%",
                },
                {
                    "patient_id": "BC-103",
                    "age": 62,
                    "her2_status": "ISH amplified",
                    "prior_therapy": "Taxane and trastuzumab completed; pertuzumab intolerant",
                    "ecog": 1,
                    "brain_mri": "No CNS disease",
                    "lvef": "55%",
                },
                {
                    "patient_id": "BC-102",
                    "age": 54,
                    "her2_status": "IHC 2+ / ISH non-amplified",
                    "prior_therapy": "Taxane only; trastuzumab naive",
                    "ecog": 2,
                    "brain_mri": "New enhancing frontal lesion on last scan",
                    "lvef": "49%",
                },
            ],
        },
        expected_fields={
            "BC-101_her2_status": "ihc 3+",
            "BC-101_ecog": "0",
            "BC-103_her2_status": "ish amplified",
            "BC-103_ecog": "1",
            "BC-102_cns_status": "active cns progression",
            "BC-102_trastuzumab_exposure": "none",
        },
        gold_ranking=["BC-101", "BC-103", "BC-102"],
        gold_final_decision="ranking_submitted",
        grader_name="grade_medium_ranking",
    ),
    "hard": TaskSpec(
        task_id="hard",
        difficulty="hard",
        title="Protocol Deviations in AML Screening Note",
        instructions=(
            "Read the unstructured hematology note, identify protocol deviations or exclusions, "
            "and determine if the patient is screen-failed. Use extract_data for critical facts, "
            "flag_deviation for protocol violations, then submit_decision with eligible or ineligible."
        ),
        context={
            "trial_protocol": {
                "phase": "Phase Ib",
                "disease": "Relapsed/refractory FLT3-mutated AML",
                "key_exclusions": [
                    "QTc > 470 ms",
                    "Strong CYP3A4 inhibitor within 7 days",
                    "Active uncontrolled infection",
                    "Last investigational therapy within 14 days",
                ],
            },
            "screening_note": (
                "Mr. J.R. is a 63-year-old man with relapsed AML harboring FLT3-ITD. He completed "
                "cycle 2 of investigational menin inhibitor ABX-17 nine days ago and was admitted "
                "overnight for neutropenic fever. Blood cultures from 04/06 grew Klebsiella; he remains "
                "on IV cefepime. ECG today showed QTc 486 ms. Posaconazole prophylaxis was restarted "
                "five days ago because of prior pulmonary aspergillosis. Bone marrow confirms persistent "
                "22% blasts. ECOG 1. Creatinine clearance 68 mL/min."
            ),
        },
        expected_fields={
            "age": "63",
            "biomarker": "flt3-itd",
            "ecg_qtc_ms": "486",
            "last_investigational_therapy_days": "9",
            "cyp3a4_inhibitor": "posaconazole within 7 days",
            "infection_status": "active klebsiella bacteremia",
        },
        valid_deviations=[
            "qtc greater than 470 ms",
            "recent strong cyp3a4 inhibitor",
            "active uncontrolled infection",
            "investigational therapy washout under 14 days",
        ],
        gold_final_decision="ineligible",
        grader_name="grade_hard_protocol_deviation",
    ),
}

TASK_SEQUENCE = ["easy", "medium", "hard"]


def grade_easy_eligibility(
    task: TaskSpec,
    extracted_fields: Dict[str, str],
    identified_deviations: List[str],
    final_decision: Optional[str],
    ranking: Optional[List[str]] = None,
) -> float:
    correct_fields = sum(
        1 for field, expected in task.expected_fields.items() if _normalize(extracted_fields.get(field)) == _normalize(expected)
    )
    field_score = correct_fields / max(len(task.expected_fields), 1)
    decision_score = 1.0 if _normalize(final_decision) == _normalize(task.gold_final_decision) else 0.0
    return round((0.5 * field_score) + (0.5 * decision_score), 4)


def grade_medium_ranking(
    task: TaskSpec,
    extracted_fields: Dict[str, str],
    identified_deviations: List[str],
    final_decision: Optional[str],
    ranking: Optional[List[str]] = None,
) -> float:
    ranking = ranking or []
    if not ranking:
        return 0.0
    positional_hits = sum(
        1 for actual, expected in zip(ranking, task.gold_ranking) if actual == expected
    ) / len(task.gold_ranking)
    pairwise_hits = 0
    total_pairs = 0
    for i, higher in enumerate(task.gold_ranking):
        for lower in task.gold_ranking[i + 1 :]:
            total_pairs += 1
            if ranking.index(higher) < ranking.index(lower):
                pairwise_hits += 1
    pairwise_score = pairwise_hits / max(total_pairs, 1)
    return round((0.6 * positional_hits) + (0.4 * pairwise_score), 4)


def grade_hard_protocol_deviation(
    task: TaskSpec,
    extracted_fields: Dict[str, str],
    identified_deviations: List[str],
    final_decision: Optional[str],
    ranking: Optional[List[str]] = None,
) -> float:
    deviation_hits = sum(
        1 for deviation in task.valid_deviations if deviation in identified_deviations
    )
    deviation_score = deviation_hits / max(len(task.valid_deviations), 1)
    correct_fields = sum(
        1 for field, expected in task.expected_fields.items() if _normalize(extracted_fields.get(field)) == _normalize(expected)
    )
    field_score = correct_fields / max(len(task.expected_fields), 1)
    decision_score = 1.0 if _normalize(final_decision) == _normalize(task.gold_final_decision) else 0.0
    return round((0.4 * deviation_score) + (0.3 * field_score) + (0.3 * decision_score), 4)


GRADERS = {
    "easy": grade_easy_eligibility,
    "medium": grade_medium_ranking,
    "hard": grade_hard_protocol_deviation,
}


class ClinicalTrialEnvironment(Environment[ClinicalTrialAction, ClinicalTrialObservation, ClinicalTrialState]):
    """Clinical trial screening environment with three deterministic tasks."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._task_cursor = -1
        self._current_task: Optional[TaskSpec] = None
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
        self._current_task = TASKS[selected_task_id]
        self._submitted_ranking = []
        self._state = ClinicalTrialState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_task_id=self._current_task.task_id,
            difficulty=self._current_task.difficulty,
            title=self._current_task.title,
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
        if self._current_task is None:
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

        if self._state.step_count >= self._current_task.max_steps and not done:
            done = True
            terminal_reason = "max_steps_reached"

        if done:
            reward.grader_score = GRADERS[self._current_task.task_id](
                self._current_task,
                self._state.extracted_fields,
                self._state.identified_deviations,
                self._state.final_decision,
                self._submitted_ranking,
            )
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

    def _next_task_id(self) -> str:
        self._task_cursor = (self._task_cursor + 1) % len(TASK_SEQUENCE)
        return TASK_SEQUENCE[self._task_cursor]

    def _handle_extraction(self, action: ClinicalTrialAction, reward: ClinicalTrialReward) -> None:
        if not action.field_name or action.value is None:
            reward.penalty += HALLUCINATION_PENALTY
            reward.notes.append("extract_data requires field_name and value.")
            return

        expected_value = self._current_task.expected_fields.get(action.field_name)
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
        submitted = [_normalize(item) for item in action.deviations]
        if not submitted:
            reward.penalty += HALLUCINATION_PENALTY
            reward.notes.append("flag_deviation requires at least one deviation.")
            return

        for deviation in submitted:
            if deviation in self._current_task.valid_deviations:
                if deviation not in self._state.identified_deviations:
                    self._state.identified_deviations.append(deviation)
                    reward.incremental_reward += INCREMENTAL_REWARD
                    reward.matched_items.append(deviation)
                    reward.notes.append(f"Validated deviation: {deviation}.")
            else:
                reward.penalty += HALLUCINATION_PENALTY
                reward.notes.append(f"Unsupported deviation claim: {deviation}.")

    def _handle_ranking(self, action: ClinicalTrialAction, reward: ClinicalTrialReward) -> None:
        ranking = action.ranking
        valid_patients = [patient["patient_id"] for patient in self._current_task.context.get("patients", [])]
        if sorted(ranking) != sorted(valid_patients):
            reward.penalty += HALLUCINATION_PENALTY
            reward.notes.append("Ranking must include each patient exactly once.")
            return
        self._submitted_ranking = ranking
        self._state.final_decision = "ranking_submitted"

    def _is_final_submission_correct(self) -> bool:
        if self._current_task is None:
            return False
        if self._current_task.task_id == "medium":
            return self._submitted_ranking == self._current_task.gold_ranking
        return _normalize(self._state.final_decision) == _normalize(self._current_task.gold_final_decision)

    def _missing_items(self) -> List[str]:
        if self._current_task is None:
            return []
        missing_fields = [
            field
            for field, expected in self._current_task.expected_fields.items()
            if _normalize(self._state.extracted_fields.get(field)) != _normalize(expected)
        ]
        if self._current_task.task_id == "hard":
            missing_fields.extend(
                deviation
                for deviation in self._current_task.valid_deviations
                if deviation not in self._state.identified_deviations
            )
        return missing_fields

    def _build_observation(
        self,
        reward_details: ClinicalTrialReward,
        done: bool,
        terminal_reason: Optional[str] = None,
    ) -> ClinicalTrialObservation:
        assert self._current_task is not None
        attempts_remaining = max(self._current_task.max_steps - self._state.step_count, 0)
        return ClinicalTrialObservation(
            task_id=self._current_task.task_id,
            difficulty=self._current_task.difficulty,  # type: ignore[arg-type]
            title=self._current_task.title,
            instructions=self._current_task.instructions,
            context=self._current_task.context,
            expected_fields=list(self._current_task.expected_fields.keys()),
            extracted_fields=dict(self._state.extracted_fields),
            identified_deviations=list(self._state.identified_deviations),
            attempts_remaining=attempts_remaining,
            grader_name=self._current_task.grader_name,
            reward_details=reward_details,
            reward=reward_details.total_reward,
            done=done,
            metadata={
                "grading_score": self._state.grading_score,
            },
            terminal_reason=terminal_reason,
        )
