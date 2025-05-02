from __future__ import annotations

import numpy as np


class MCTSNode:
    def __init__(
        self,
        parent: MCTSNode | None = None,
        action: int | None = None,
        prior: float = 0.0,
    ) -> None:
        self.parent = parent
        self.action = action
        self.children: dict[int, MCTSNode] = {}
        self.visits = 0
        self.value = 0.0
        self.prior = prior

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self, max_depth: int, current_depth: int) -> bool:
        return current_depth >= max_depth

    def select_child(self, c_puct: float = 0.8) -> "MCTSNode":
        # UCB formula for selection
        best_score = -float("inf")
        best_action = None

        # Calculate the square root term outside the loop for efficiency
        sqrt_parent_visits = np.sqrt(self.visits)
        for action, child in self.children.items():
            # UCB score combines exploitaton and exploration (prior * sqrt(parent_visits) / child_visits)
            exploit = child.value
            explore = c_puct * child.prior * sqrt_parent_visits / (1 + child.visits)
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_action = action

        assert best_action is not None, "Action should be selected"

        return self.children[best_action]
