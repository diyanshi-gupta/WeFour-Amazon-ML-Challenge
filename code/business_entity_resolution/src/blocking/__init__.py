"""Blocking modules for candidate pair generation."""
from code.business_entity_resolution.src.blocking.rule_based import (
    exact_name_country_blocking,
    evaluate_blocking_recall,
)

__all__ = [
    "exact_name_country_blocking",
    "evaluate_blocking_recall",
]
