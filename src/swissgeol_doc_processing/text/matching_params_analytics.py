"""Module for tracking matching parameter usage in material descriptions."""

import json
from collections import defaultdict
from pathlib import Path

from swissgeol_doc_processing.utils.file_utils import read_params


class MatchingParamsAnalytics:
    """Analytics tracker for matching parameters."""

    def __init__(self, material_description_params: dict):
        """Initialize analytics tracker.

        Args:
            material_description_params: Material description parameters from matching_params.yml
        """
        self.material_description_params = material_description_params
        self.usage_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

    def track_expression_match(self, expression: str, language: str, is_excluding: bool = False):
        """Track a single expression match.

        Args:
            expression: The matched expression
            language: Language code
            is_excluding: Whether this is an excluding expression
        """
        expression_type = "excluding_expressions" if is_excluding else "including_expressions"
        self.usage_stats[language][expression_type][expression] += 1

    def get_summary(self) -> dict:
        """Get analytics summary with usage stats and unused expressions."""
        summary = {"usage_stats": dict(self.usage_stats), "unused_expressions": {}}

        # Find unused expressions
        for language, expressions in self.material_description_params.items():
            unused_by_type = {}

            for expression_type in ["including_expressions", "excluding_expressions"]:
                if expression_type not in expressions:
                    continue

                used_expressions = set(self.usage_stats[language][expression_type].keys())
                all_expressions = set(expressions[expression_type])
                unused_by_type[expression_type] = list(all_expressions - used_expressions)

            if unused_by_type:
                summary["unused_expressions"][language] = unused_by_type

        return summary

    def save_analytics(self, output_path: Path):
        """Save analytics data to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        analytics_data = {
            "summary": self.get_summary(),
            "configuration": {"material_description_params": self.material_description_params},
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analytics_data, f, indent=2, ensure_ascii=False)


def create_analytics(config_path: str | None = None) -> MatchingParamsAnalytics | None:
    """Create analytics instance if parameters provided.

    Args:
        config_path: Path to user-provided config file. Defaults to None.

    Returns:
        Analytics instance or None if disabled
    """
    material_description_params = read_params("matching_params.yml", user_config_path=config_path)[
        "material_description"
    ]
    return MatchingParamsAnalytics(material_description_params) if material_description_params else None


def track_match(
    analytics: MatchingParamsAnalytics | None, expression: str, language: str, is_excluding: bool = False
) -> None:
    """Track a match if analytics is enabled.

    Args:
        analytics: Analytics instance or None
        expression: The matched expression
        language: Language code
        is_excluding: Whether this is an excluding expression
    """
    if analytics:
        analytics.track_expression_match(expression, language, is_excluding)
