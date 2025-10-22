"""Module implementing logic for pairing elements with a dynamical programming approach."""

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from typing import TypeVar

L = TypeVar("L")
R = TypeVar("R")


Match = tuple[L, R | list[R]]


class DP(ABC):
    """Abstract base class for dynamic programming matching."""

    def __init__(self, left: list[L], right: list[R], right_affinity: list[float]):
        """Initialize the DP matcher.

        Args:
            left (list[L]): The list of left elements to be matched.
            right (list[R]): The list of right elements to be matched.
            right_affinity (list[float]): The affinity scores for the right elements. The affinity score is used to
                encourage grouping compatible right elements together when they are matched to the same left element.
                For one to one matching, this can be a list of zeros.
        """
        self.left = left
        self.right = right
        self.right_affinity = right_affinity
        self.nL, self.nR = len(self.left), len(self.right)

    def solve(self, scoring_fn: Callable[[L, R], float]) -> tuple[float, list[Match]]:
        """Solve the matching problem using dynamic programming.

        Args:
            scoring_fn: Callable[[L, R], float]: the scoring function to be used for selecting the best mapping.

        Returns:
            tuple: A tuple containing:
                - matching_score (float): The overall matching score.
                - left_to_right_mapping (list[Match]): The mapping between left and right elements.
        """
        if not self.left or not self.right:
            return 0.0, []

        # Precompute pairwise scores
        pairwise_scores = self._compute_scores(scoring_fn)

        # Build DP table and pointer
        dp, ptr = self._build_dp_table(pairwise_scores)

        # Backtrack to recover mapping
        left_to_right_mapping = self._get_mapping(ptr)

        matching_score = dp[self.nL][self.nR] / max(self.nL, self.nR)  # maximum is 1.
        return matching_score, left_to_right_mapping

    def _compute_scores(self, scoring_fn: Callable[[L, R], float]) -> list[list[float]]:
        """Compute pairwise scores between all left and right elements.

        Args:
            scoring_fn: Callable[[L, R], float]: the scoring function to be used for selecting the best mapping.

        Returns:
            list[list[float]]: scores for each left-right pair.
        """
        pair_score = [[0.0] * self.nR for _ in range(self.nL)]

        for i in range(self.nL):
            for j in range(self.nR):
                pair_score[i][j] = scoring_fn(self.left[i], self.right[j])
        return pair_score

    @abstractmethod
    def _build_dp_table(self, pair_score: list[list[float]]) -> tuple[list[list[float]], list[list[str]]]:
        """Build the dynamic programming table."""
        raise NotImplementedError

    @abstractmethod
    def _get_mapping(self, ptr: list[list[str]]) -> list[Match]:
        """Get the mapping between left and right elements by backtracking through the DP table."""
        raise NotImplementedError


class IntervalToLinesDP(DP):
    """Dynamic programming matcher for interval zone to lines matching."""

    def _build_dp_table(self, pair_score: list[list[float]]) -> tuple[list[list[float]], list[list[str]]]:
        """Build the dynamic programming table for layer matching.

        The algorithm finds the optimal alignment between interval zones and text lines while preserving their
        relative order. It uses a dynamic programming approach where:
        - Each cell dp[i][j] represents the best cumulative score for matching the first i
          intervals with the first j lines.
        - Moves can be:
          * Diagonal: Match current interval with current line, with previous lines belonging to previous intervals.
          * Up: Keep previous interval assignments for current line.
          * Left: Assign current line to the same interval as the previous line. As the lines in the same interval,
            we add the affinity bonus for the line to encourage grouping compatible lines together.
        - In case of equal scores, diagonal moves are preferred to preserve matching,
          followed by up moves, then left moves.

        Args:
            pair_score (list[list[float]]): The pairwise scores between predicted and ground truth layers.
                Each score combines text similarity and depth matching accuracy.

        Returns:
            tuple: A tuple containing:
                - dp (list[list[float]]): The dynamic programming table with cumulative scores
                - ptr (list[list[str]]): The pointer table storing move directions ('diag', 'up', 'left')
                    used for backtracking to recover the optimal matching.
        """
        dp = [[-float("inf")] * (self.nR + 1)] + [[0.0] * (self.nR + 1) for _ in range(self.nL)]
        ptr = [[None] * (self.nR + 1) for _ in range(self.nL + 1)]

        for i in range(1, self.nL + 1):
            for j in range(1, self.nR + 1):
                # up: trust what was already matched, let current line j with previous intervals
                up = dp[i - 1][j]

                # left : steal the line to belong to current interval, consider the line affinity bonus (which can
                # be negative) to take into account that lines in the same interval are compatible.
                bonus = self.right_affinity[j - 1]
                left = dp[i][j - 1] + pair_score[i - 1][j - 1] + bonus  # steal: new pairing is better

                # diag : begin a new interval with the current line, thus not considering affinity bonus
                diag = dp[i - 1][j - 1] + pair_score[i - 1][j - 1]

                # tie-break: diag > up > left
                candidates = [("diag", diag, 3), ("up", up, 2), ("left", left, 1)]
                move, score, _ = max(candidates, key=lambda x: (x[1], x[2]))  # break tie with prioritize matching
                dp[i][j] = score
                ptr[i][j] = move
        return dp, ptr

    def _get_mapping(self, ptr: list[list[str]]) -> list[Match]:
        """Get the mapping between intervals and lines by backtracking through the DP table.

        Args:
            ptr (list[list[str]]): The pointer table for backtracking.

        Returns:
            list[Match]: The mapping between intervals and and possibly many lines.
        """
        i, j = self.nL, self.nR
        mapping: OrderedDict[int, list[int]] = defaultdict(list)
        while i > 0 and j > 0:
            move = ptr[i][j]
            left_idx = i - 1
            right_idx = j - 1
            if move == "diag":
                mapping[left_idx].append(right_idx)
                mapping[left_idx].reverse()
                i -= 1
                j -= 1
            elif move == "up":
                mapping[left_idx].reverse()
                i -= 1
            elif move == "left":
                mapping[left_idx].append(right_idx)
                j -= 1
            else:
                break

        if i != 0:
            assert move == "left"
            mapping[i - 1].reverse()

        # assign remaining lines to first interval
        assert j == 0  # all first moves should be left, thanks to the -inf top row
        for k in range(j):
            mapping[0].insert(0, k)

        mapping = OrderedDict(reversed(list(mapping.items())))

        return [
            (self.left[left_idx], [self.right[right_idx] for right_idx in right_idxes])
            for left_idx, right_idxes in mapping.items()
        ]


class PredToGroundTruthLayerDP(DP):
    """Dynamic programming matcher for predicted layers to ground truth layers matching."""

    def _build_dp_table(self, pair_score: list[list[float]]) -> tuple[list[list[float]], list[list[str]]]:
        """Build the dynamic programming table for layer matching.

        The algorithm finds the optimal alignment between predicted and ground truth layers
        while preserving their relative order. It uses a dynamic programming approach where:
        - Each cell dp[i][j] represents the best cumulative score for matching the first i
          predictions with the first j ground truth layers
        - Moves can be:
          * Diagonal: Match current prediction with current ground truth (score from pair_score)
          * Up: Skip current prediction (no additional score)
          * Left: Skip current ground truth (no additional score)
        - In case of equal scores, diagonal moves are preferred to preserve matching, then up moves, then left moves.

        Args:
            pair_score (list[list[float]]): The pairwise scores between predicted and ground truth layers.
                Each score combines text similarity and depth matching accuracy.

        Returns:
            tuple: A tuple containing:
                - dp (list[list[float]]): The dynamic programming table with cumulative scores
                - ptr (list[list[str]]): The pointer table storing move directions ('diag', 'up', 'left')
                    used for backtracking to recover the optimal matching.
        """
        dp = [[0.0] * (self.nR + 1) for _ in range(self.nL + 1)]
        ptr = [[None] * (self.nR + 1) for _ in range(self.nL + 1)]

        for i in range(1, self.nL + 1):
            for j in range(1, self.nR + 1):
                diag = dp[i - 1][j - 1] + pair_score[i - 1][j - 1]
                up = dp[i - 1][j]
                left = dp[i][j - 1]
                candidates = [("diag", diag, 3), ("up", up, 2), ("left", left, 1)]
                choice = max(candidates, key=lambda x: (x[1], x[2]))  # break tie with prioritize matching
                dp[i][j] = choice[1]
                ptr[i][j] = choice[0]
        return dp, ptr

    def _get_mapping(self, ptr: list[list[str]]) -> list[Match]:
        """Get the mapping between predicted and ground truth layers by backtracking through the DP table.

        Args:
            ptr (list[list[str]]): The pointer table for backtracking.

        Returns:
            list[Match]: The mapping between predicted and ground truth layers.
        """
        i, j = self.nL, self.nR
        mapping = []
        while i > 0 and j > 0:
            move = ptr[i][j]
            if move == "diag":
                predicted_layer_index, ground_truth_layer_index = i - 1, j - 1
                mapping.append((predicted_layer_index, ground_truth_layer_index))
                i, j = i - 1, j - 1
            elif move == "up":
                i -= 1
            elif move == "left":
                j -= 1
            else:
                break
        mapping.reverse()

        return [(self.left[left_idx], self.right[right_idx]) for left_idx, right_idx in mapping]
