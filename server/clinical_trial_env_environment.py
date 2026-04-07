"""Compatibility wrapper for the root clinical trial environment."""

from __future__ import annotations

try:
    from ..env import ClinicalTrialEnvironment
except ImportError:
    from env import ClinicalTrialEnvironment

__all__ = ["ClinicalTrialEnvironment"]
