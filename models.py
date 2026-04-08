"""Typed models for the clinical trial screening environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field


class ClinicalTrialReward(BaseModel):
    """Structured reward payload for deterministic grading and shaping."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    incremental_reward: float = Field(default=0.0, description="Reward from validated extractions.")
    final_reward: float = Field(default=0.0, description="Terminal reward for a correct final decision.")
    penalty: float = Field(default=0.0, description="Penalty for hallucinations or destructive actions.")
    total_reward: float = Field(default=0.0, description="Net reward for the step.")
    grader_score: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Deterministic task score in the strict open interval (0, 1).",
    )
    matched_items: List[str] = Field(default_factory=list, description="Correctly matched items this step.")
    missing_items: List[str] = Field(default_factory=list, description="Expected items still missing at grading time.")
    notes: List[str] = Field(default_factory=list, description="Human-readable reward rationale.")


class ClinicalTrialAction(Action):
    """Agent action for extracting evidence and making screening decisions."""

    action_type: Literal[
        "extract_data",
        "rank_patients",
        "flag_deviation",
        "submit_decision",
        "delete_evidence",
    ] = Field(..., description="Type of environment action.")
    field_name: Optional[str] = Field(default=None, description="Clinical field being extracted.")
    value: Optional[str] = Field(default=None, description="Normalized value for the extracted field.")
    patient_id: Optional[str] = Field(default=None, description="Patient identifier for ranking or extraction.")
    ranking: List[str] = Field(default_factory=list, description="Ordered patient IDs from best to worst fit.")
    deviations: List[str] = Field(default_factory=list, description="Protocol deviations or exclusions identified.")
    final_decision: Optional[str] = Field(
        default=None,
        description="Terminal decision such as eligible, ineligible, or ranking_submitted.",
    )
    rationale: Optional[str] = Field(default=None, description="Optional short reasoning trace.")


class ClinicalTrialObservation(Observation):
    """Observation returned after each environment interaction."""

    task_id: str = Field(..., description="Current task identifier.")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Task difficulty.")
    title: str = Field(..., description="Scenario title.")
    instructions: str = Field(..., description="Task instructions for the agent.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Structured and unstructured patient context.")
    expected_fields: List[str] = Field(default_factory=list, description="High-value clinical fields to extract.")
    extracted_fields: Dict[str, str] = Field(default_factory=dict, description="Validated data extracted so far.")
    identified_deviations: List[str] = Field(default_factory=list, description="Validated protocol deviations found.")
    attempts_remaining: int = Field(default=0, description="Steps left in the current episode.")
    grader_name: str = Field(default="", description="Deterministic grader assigned to the task.")
    reward_details: ClinicalTrialReward = Field(
        default_factory=ClinicalTrialReward,
        description="Structured reward breakdown for the step.",
    )
    terminal_reason: Optional[str] = Field(default=None, description="Why the episode ended, if done.")


class ClinicalTrialState(State):
    """Internal environment state exposed through the OpenEnv state endpoint."""

    current_task_id: Optional[str] = Field(default=None, description="Current task identifier.")
    difficulty: Optional[str] = Field(default=None, description="Difficulty for the current task.")
    title: Optional[str] = Field(default=None, description="Current task title.")
    extracted_fields: Dict[str, str] = Field(default_factory=dict, description="Accepted extracted fields.")
    identified_deviations: List[str] = Field(default_factory=list, description="Accepted deviations.")
    final_decision: Optional[str] = Field(default=None, description="Submitted terminal decision.")
    grading_score: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Latest grader output in the strict open interval (0, 1).",
    )

    def __call__(self) -> "ClinicalTrialState":
        """Support env.state() as a compatibility alias for the OpenEnv state property."""
        return self
