"""Module implementing logic for pairing elements with a dynamical programming approach."""

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from typing import TypeVar

from swissgeol_doc_processing.utils.file_utils import read_params

L = TypeVar("L")
R = TypeVar("R")


Match = tuple[L, R | list[R]]

matching_params = read_params("matching_params.yml")


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

        # Build DP table and backtrack to recover mapping
        score, left_to_right_mapping = self._get_mapping(pairwise_scores)

        matching_score = score / max(self.nL, self.nR)  # maximum is 1.
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
    def _get_mapping(self, pair_score: list[list[float]]) -> tuple[float, list[Match]]:
        """Get the mapping between left and right elements by backtracking through the DP table."""
        raise NotImplementedError


class IntervalToLinesDP(DP):
    """Dynamic programming matcher for interval zone to lines matching."""

    def _get_mapping(self, pair_score: list[list[float]]) -> tuple[float, Match]:
        """Get the mapping between intervals and text lines by backtracking through the DP table.

        The algorithm finds the optimal alignment between interval zones and text lines while preserving their
        relative order. It uses a dynamic programming approach where:
        - Each cell dp[i][j] represents the best cumulative score for matching the first i
          intervals with the first j lines.
        - Moves can be:
          * Diagonal: assign the j'th line to the i'th interval, and move to the previous interval
          * Up: move to the previous interval without assigning the j'th line yet
          * Left: assign the j'th line to the i'th interval, and stay with the same interval
        - In case of equal scores, diagonal moves are preferred to preserve matching,
          followed by up moves, then left moves.

        Forbidden transitions:
            We never allow an up move to be immediately followed by a left move. Such an up+left pair is a degenerate
            way of "re-labeling" what should instead be scored as a single diagonal step: a line that starts a new
            interval should be charged once by the diagonal move, not smeared over an up+left pair that would let the
            same line both close the skipped interval and open the next one, double-counting the empty-interval
            penalty and the affinity bonus.

            The reverse order, left followed by up, IS allowed: it is legitimate to finish a multi-line interval
            (left) and then skip the next interval (up) when a borehole has an interval with a multi-line description
            immediately above an interval that gets no description at all.

            This is enforced by tracking, for every cell dp[i][j], a separate score per *last move taken*
            (dp[i][j]["up"], dp[i][j]["diag"], dp[i][j]["left"]) together with the corresponding predecessor pointer
            (ptr[i][j][move]). Each move then only accepts a compatible previous move:
              * up   <- {diag, up, left} (an empty interval may follow anything, including a finished multi-line one)
              * left <- {diag, left}     (a continuation can only follow a fresh start or another continuation)
              * diag <- {diag, up, left} (a fresh interval start may follow anything)
            Because the predecessor set for left contains only {diag, left}, a left can never directly follow an up,
            while up remains free to follow a left.

        Args:
            pair_score (list[list[float]]): The pairwise scores between predicted and ground truth layers.
                Each score combines text similarity and depth matching accuracy.

        Returns:
            tuple: A tuple containing:
                - score (float): The optimal matching score
                - list[Match]: The mapping between intervals and possibly many lines.
        """
        EMPTY_INTERVAL_PENALTY = matching_params["empty_interval_penalty"]

        def default_state(value) -> dict:
            return {"up": value, "diag": value, "left": value}

        # Initialize row by row. Each cell must be its own dict (no list * dict aliasing). Column 0 represents the
        # state "no lines assigned yet": reachable only via up moves, costing one EMPTY_INTERVAL_PENALTY per interval.
        dp = [
            [{"up": -EMPTY_INTERVAL_PENALTY * i, "diag": -float("inf"), "left": -float("inf")}]
            + [default_state(-float("inf")) for _ in range(self.nR)]
            for i in range(self.nL + 1)
        ]
        ptr = [[default_state(None) for _ in range(self.nR + 1)] for _ in range(self.nL + 1)]

        def pick_previous_move(i: int, j: int, options: list[str]) -> tuple[str, float]:
            move = max(
                options, key=lambda move: dp[i][j][move]
            )  # in case of tie-break, the first maximal option is chosen
            return move, dp[i][j][move]

        for i in range(1, self.nL + 1):
            for j in range(1, self.nR + 1):
                # up: move to the previous interval without assigning the j'th line yet (in other words: the i'th
                # interval will remain empty)
                prev_move, prev_score = pick_previous_move(i - 1, j, ["diag", "up", "left"])
                ptr[i][j]["up"] = prev_move
                dp[i][j]["up"] = prev_score - EMPTY_INTERVAL_PENALTY

                # left : assign the j'th line to the i'th interval, and stay with the same interval (in other words:
                # this line does not start a new interval)
                prev_move, prev_score = pick_previous_move(i, j - 1, ["diag", "left"])
                ptr[i][j]["left"] = prev_move
                bonus = self.right_affinity[j - 1]
                # steal: new pairing is better
                dp[i][j]["left"] = prev_score + pair_score[i - 1][j - 1] + bonus

                # diag : assign the j'th line to the i'th interval, and move to the previous interval (in other words:
                # this line starts a new interval)
                prev_move, prev_score = pick_previous_move(i - 1, j - 1, ["diag", "up", "left"])
                ptr[i][j]["diag"] = prev_move
                dp[i][j]["diag"] = prev_score + pair_score[i - 1][j - 1]

        i, j = self.nL, self.nR
        move, max_score = pick_previous_move(i, j, ["diag", "up", "left"])
        mapping: OrderedDict[int, list[int]] = defaultdict(list)
        while i > 0 and j > 0:
            previous_move = ptr[i][j][move]
            left_idx = i - 1
            right_idx = j - 1
            if move == "diag":
                mapping[left_idx].append(right_idx)
                i -= 1
                j -= 1
            elif move == "up":
                i -= 1
            elif move == "left":
                mapping[left_idx].append(right_idx)
                j -= 1
            else:
                break
            move = previous_move

        return max_score, [
            (self.left[left_idx], [self.right[right_idx] for right_idx in reversed(mapping[left_idx])])
            for left_idx in range(self.nL)
        ]


class OrderedMatchingDP(DP):
    """Match two ordered sequences one to one, allowing elements on either side to be skipped."""

    def _get_mapping(self, pair_score: list[list[float]]) -> tuple[float, list[Match]]:
        """Get the ordered one-to-one mapping between left and right elements.

        This is doing by building a dynamic programming table and then backtracking through it.

        The algorithm finds the optimal alignment between left and right elements
        while preserving their relative order. It uses a dynamic programming approach where:
        - Each cell dp[i][j] represents the best cumulative score for matching the first i
          left elements with the first j right elements
        - Moves can be:
          * Diagonal: Match current left element with current right element (score from pair_score)
          * Up: Skip current left element (no additional score)
          * Left: Skip current right element (no additional score)
        - In case of equal scores, diagonal moves are preferred to preserve matching, then up moves, then left moves.

        Args:
            pair_score (list[list[float]]): The pairwise scores between left and right elements.

        Returns:
            tuple: A tuple containing:
                - score (float): The optimal matching score
                - list[Match]: The one-to-one mapping between left and right elements.
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

        i, j = self.nL, self.nR
        mapping = []
        while i > 0 and j > 0:
            move = ptr[i][j]
            if move == "diag":
                left_idx, right_idx = i - 1, j - 1
                mapping.append((left_idx, right_idx))
                i, j = i - 1, j - 1
            elif move == "up":
                i -= 1
            elif move == "left":
                j -= 1
            else:
                break
        mapping.reverse()

        return dp[self.nL][self.nR], [(self.left[left_idx], self.right[right_idx]) for left_idx, right_idx in mapping]
