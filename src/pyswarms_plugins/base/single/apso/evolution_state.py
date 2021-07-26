from enum import Enum

import numpy as np

__all__ = ["EvolutionState"]
__author__ = "Hung-Tien Huang"
__contact__ = "hungtienhuang@gmail.com"


class EvolutionState(Enum):
    S1_EXPLORATION = 1
    S2_EXPLOITATION = 2
    S3_CONVERGENCE = 3
    S4_JUMPING_OUT = 4

    @staticmethod
    def compute_exploration_numeric(evo_factor: float) -> float:
        if 0.0 <= evo_factor and evo_factor <= 0.4:
            return 0.0
        elif 0.4 < evo_factor and evo_factor <= 0.6:
            return 5.0 * evo_factor - 2.0
        elif 0.6 < evo_factor and evo_factor <= 0.7:
            return 1.0
        elif 0.7 < evo_factor and evo_factor <= 0.8:
            return -10.0 * evo_factor + 8.0
        elif 0.8 < evo_factor and evo_factor <= 1.0:
            return 0.0
        raise ValueError("evo_factor not in range [0.0, 1.0]")

    @staticmethod
    def compute_exploitation_numeric(evo_factor: float) -> float:
        if 0.0 <= evo_factor and evo_factor <= 0.2:
            return 0.0
        elif 0.2 < evo_factor and evo_factor <= 0.3:
            return 10.0 * evo_factor - 2.0
        elif 0.3 < evo_factor and evo_factor <= 0.4:
            return 1.0
        elif 0.4 < evo_factor and evo_factor <= 0.6:
            return -5.0 * evo_factor + 3.0
        elif 0.6 < evo_factor and evo_factor <= 1.0:
            return 0.0
        raise ValueError("evo_factor not in range [0.0, 1.0]")

    @staticmethod
    def compute_convergence_numeric(evo_factor: float) -> float:
        if 0.0 <= evo_factor and evo_factor <= 0.1:
            return 1.0
        elif 0.1 < evo_factor and evo_factor <= 0.3:
            return -5.0 * evo_factor + 1.5
        elif 0.3 < evo_factor and evo_factor <= 1.0:
            return 0.0
        raise ValueError("evo_factor not in range [0.0, 1.0]")

    @staticmethod
    def compute_jump_out_numeric(evo_factor: float) -> float:
        if 0.0 <= evo_factor and evo_factor <= 0.7:
            return 0.0
        elif 0.7 < evo_factor and evo_factor <= 0.9:
            return 5.0 * evo_factor - 3.5
        elif 0.9 < evo_factor and evo_factor <= 1.0:
            return 1.0
        raise ValueError("evo_factor not in range [0.0, 1.0]")

    @staticmethod
    def classify_next_state(
        state_numeric: np.ndarray,
        curr_state: "EvolutionState",
    ) -> "EvolutionState":
        """Classify next state given the state numerics and the current state.

        Use singleton method along with the single direction circular APSO sequence table. The circular APSO sequence is defined as S_1 => S_2 => S_3 => S_4 => S_1.

        Parameters
        ----------
        state_numeric : np.ndarray
            (4,) in the order of [s1_num, s2_num, s3_num, s4_num].
        curr_state : AdaptiveOptimizerPSO.EvolutionState
            the current state that APSO is in.

        Returns
        -------
        AdaptiveOptimizerPSO.EvolutionState
            the next evolutionary state.
        """
        if (curr_state is EvolutionState.S1_EXPLORATION and
                state_numeric[0] < state_numeric[1]):
            return EvolutionState.S2_EXPLOITATION
        elif (curr_state is EvolutionState.S2_EXPLOITATION and
              state_numeric[1] < state_numeric[2]):
            return EvolutionState.S3_CONVERGENCE
        elif (curr_state is EvolutionState.S3_CONVERGENCE and
              state_numeric[2] < state_numeric[3]):
            return EvolutionState.S4_JUMPING_OUT
        elif (curr_state is EvolutionState.S4_JUMPING_OUT and
              state_numeric[3] < state_numeric[0]):
            return EvolutionState.S1_EXPLORATION
        return curr_state
