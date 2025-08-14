"""Module for tracking matching parameter usage in material descriptions."""

import json
import logging
from collections import defaultdict
from pathlib import Path


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

    def _convert_tuples_to_strings(self, data):
        """Convert tuple keys to string representations for JSON serialization.

        Args:
            data: The data structure to convert

        Returns:
            The data structure with tuple keys converted to strings
        """
        if isinstance(data, dict):
            converted = {}
            for key, value in data.items():
                # Convert tuple keys to string representation
                if isinstance(key, tuple):
                    string_key = " + ".join(str(item) for item in key)
                else:
                    string_key = key
                converted[string_key] = self._convert_tuples_to_strings(value)
            return converted
        elif isinstance(data, list):
            return [self._convert_tuples_to_strings(item) for item in data]
        else:
            return data

    def save_analytics(self, output_path: Path):
        """Save analytics data to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        analytics_data = {
            "summary": self._convert_tuples_to_strings(self.get_summary()),
            "configuration": {"material_description_params": self.material_description_params},
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analytics_data, f, indent=2, ensure_ascii=False)


# Global instance
_analytics_instance: MatchingParamsAnalytics | None = None


def initialize_analytics(material_description_params: dict) -> None:
    """Initialize global analytics instance."""
    global _analytics_instance
    _analytics_instance = MatchingParamsAnalytics(material_description_params)


def get_analytics() -> MatchingParamsAnalytics | None:
    """Get global analytics instance."""
    return _analytics_instance


def track_match(expression: str, language: str, is_excluding: bool = False) -> None:
    """Convenience function to track a match if analytics is enabled."""
    if _analytics_instance:
        _analytics_instance.track_expression_match(expression, language, is_excluding)


def finalize_analytics(output_path: Path) -> None:
    """Save and cleanup analytics."""
    if _analytics_instance:
        _analytics_instance.save_analytics(output_path)
