"""Module for tracking and analyzing matching parameter usage in material descriptions."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import re
from nltk.stem.snowball import SnowballStemmer
from compound_split import char_split

logger = logging.getLogger(__name__)


class MatchingParamsAnalytics:
    """Tracks usage statistics of matching parameters across document processing."""
    
    def __init__(self, material_description_params: Dict):
        """Initialize analytics tracker.
        
        Args:
            material_description_params (Dict): The material description parameters from matching_params.yml
        """
        self.material_description_params = material_description_params
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        # Structure: stats[language][expression_type][expression] = count
        
        # Cache stemmers for performance
        self._stemmers = {}
        self._stemmed_params = {}
        self._precompute_stemmed_params()
    
    def _precompute_stemmed_params(self):
        """Precompute stemmed versions of all material description parameters."""
        for language, expressions in self.material_description_params.items():
            self._stemmed_params[language] = {}
            stemmer = self._get_stemmer(language)
            
            for expression_type in ['including_expressions', 'excluding_expressions']:
                if expression_type in expressions:
                    self._stemmed_params[language][expression_type] = {}
                    for expression in expressions[expression_type]:
                        stemmed = stemmer.stem(expression.lower())
                        self._stemmed_params[language][expression_type][expression] = stemmed
    
    def _get_stemmer(self, language: str) -> SnowballStemmer:
        """Get stemmer for language, cached for performance."""
        if language not in self._stemmers:
            stemmer_languages = {"de": "german", "fr": "french", "en": "english", "it": "italian"}
            stemmer_lang = stemmer_languages.get(language, "german")
            self._stemmers[language] = SnowballStemmer(stemmer_lang)
        return self._stemmers[language]
    
    def track_matches(self, text: str, language: str, search_excluding: bool = False) -> List[str]:
        """Track matches for a given text line and return list of matched expressions.
        
        Args:
            text (str): The text to analyze
            language (str): The language of the text
            search_excluding (bool): Whether to search for excluding expressions
            
        Returns:
            List[str]: List of matched expressions
        """
        stemmer = self._get_stemmer(language)
        
        # Tokenize and stem words in the text (same logic as original is_description)
        text_lower = text.lower()
        text_tokens = re.findall(r"\b\w+\b", text_lower)
        
        if language == "de" and not search_excluding:
            german_char_split_list = []
            for token in text_tokens:
                german_char_split = char_split.split_compound(token)[0]
                if german_char_split[0] > 0.4:
                    german_char_split_list.extend(german_char_split[1:])
                else:
                    german_char_split_list.append(token)
            text_tokens = german_char_split_list
        
        stemmed_text_tokens = {stemmer.stem(token) for token in text_tokens}
        
        # Check for matches and track them
        expression_type = "excluding_expressions" if search_excluding else "including_expressions"
        matched_expressions = []
        
        if (language in self._stemmed_params and 
            expression_type in self._stemmed_params[language]):
            
            for expression, stemmed_expression in self._stemmed_params[language][expression_type].items():
                if stemmed_expression in stemmed_text_tokens:
                    matched_expressions.append(expression)
                    self.stats[language][expression_type][expression]['matches'] += 1
        
        return matched_expressions
    
    def get_analytics_summary(self) -> Dict:
        """Get summary of analytics data."""
        summary = {
            "total_processing": {},
            "by_language": {},
            "unused_expressions": {},
        }
        
        # Calculate totals by language and expression type
        for language, lang_data in self.stats.items():
            summary["by_language"][language] = {}
            for expr_type, expr_data in lang_data.items():
                total_matches = sum(counts['matches'] for counts in expr_data.values())
                summary["by_language"][language][expr_type] = {
                    "total_matches": total_matches,
                    "unique_expressions_matched": len(expr_data),
                    "expression_details": dict(expr_data)
                }
        
        # Find unused expressions
        for language, expressions in self.material_description_params.items():
            summary["unused_expressions"][language] = {}
            for expression_type in ['including_expressions', 'excluding_expressions']:
                if expression_type in expressions:
                    unused = []
                    for expression in expressions[expression_type]:
                        if (language not in self.stats or 
                            expression_type not in self.stats[language] or 
                            expression not in self.stats[language][expression_type] or
                            self.stats[language][expression_type][expression]['matches'] == 0):
                            unused.append(expression)
                    summary["unused_expressions"][language][expression_type] = unused
        
        return summary
    
    def save_analytics(self, output_path: Path):
        """Save analytics data to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        analytics_data = {
            "summary": self.get_analytics_summary(),
            "configuration": {
                "material_description_params": self.material_description_params
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analytics_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Matching parameters analytics saved to {output_path}")


# Global analytics instance
_analytics_instance: Optional[MatchingParamsAnalytics] = None


def initialize_analytics(material_description_params: Dict):
    """Initialize the global analytics instance."""
    global _analytics_instance
    _analytics_instance = MatchingParamsAnalytics(material_description_params)


def get_analytics() -> Optional[MatchingParamsAnalytics]:
    """Get the global analytics instance."""
    return _analytics_instance


def finalize_analytics(output_path: Path):
    """Save analytics and cleanup."""
    if _analytics_instance:
        _analytics_instance.save_analytics(output_path)
