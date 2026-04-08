# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Clinical Trial Env environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import ClinicalTrialAction, ClinicalTrialObservation, ClinicalTrialState
except ImportError:
    from models import ClinicalTrialAction, ClinicalTrialObservation, ClinicalTrialState


class ClinicalTrialEnv(
    EnvClient[ClinicalTrialAction, ClinicalTrialObservation, ClinicalTrialState]
):
    """
    Client for the clinical trial screening environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ClinicalTrialEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset(task_id="easy")
        ...     print(result.observation.title)
        ...
        ...     result = await client.step(
        ...         ClinicalTrialAction(action_type="submit_decision", final_decision="eligible")
        ...     )
        ...     print(result.observation.reward_details.grader_score)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = await ClinicalTrialEnv.from_docker_image("clinical-trial-env:latest")
        >>> try:
        ...     result = await client.reset(task_id="hard")
        ...     result = await client.step(
        ...         ClinicalTrialAction(action_type="flag_deviation", deviations=["active uncontrolled infection"])
        ...     )
        ... finally:
        ...     await client.close()
    """

    def _step_payload(self, action: ClinicalTrialAction) -> Dict:
        """
        Convert ClinicalTrialAction to JSON payload for step message.

        Args:
            action: ClinicalTrialAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict) -> StepResult[ClinicalTrialObservation]:
        """
        Parse server response into StepResult[ClinicalTrialObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ClinicalTrialObservation
        """
        obs_data = payload.get("observation", {})
        observation = ClinicalTrialObservation.model_validate(
            {
                **obs_data,
                "done": obs_data.get("done", payload.get("done", False)),
                "reward": obs_data.get("reward", payload.get("reward")),
            }
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ClinicalTrialState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return ClinicalTrialState.model_validate(payload)
