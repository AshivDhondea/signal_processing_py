import src.utils.linear_algebra_functions as lin_alg
import logging
import math
import numpy as np


logging.basicConfig(level=logging.INFO)


def polynomial_state_transition_matrix(transition_span: float, num_states: int) -> np.ndarray:
    """
    Create the state transition matrix for polynomial models of degree num_states-1 over a given span of transition in
    seconds.
    Polynomial models are a subset of the class of constant-coefficient linear differential equations.
    Parameters
    ----------
    transition_span: float
          The span of transition from sample to sample in seconds.
    num_states: int
          The number of states for the chosen polynomial model.

    Returns
    -------
    state_transition_matrix: np.ndarray
         (num_states, num_states) array representing the state transition matrix.
    Notes
    -----
    The polynomial state transition matrix function is an implementation of equation xxx from [1]_.

    References
    -----
    [1] Tracking Filter Engineering, Norman Morrison. 2013.
    """
    state_transition_matrix = np.eye(num_states, dtype=np.float64)
    for y in range(num_states):
        for x in range(y, num_states):  # STM is upper triangular
            state_transition_matrix[y, x] = np.power(transition_span, x-y)/float(math.factorial(x-y))
    return state_transition_matrix


def polynomial_state_transition_matrix_3d(span_transition: float, dynamic_states: int, dimensionality: int) -> (
        np.ndarray):
    """
    Generate the full polynomial state transition matrix accounting for dimensionality.
    If dimensionality == 3, the problem is 3-dimensional. The state vector is thus defined as
        [x, x-dot, x-ddot, y, y-dot, y-ddot, z, z-dot, z-ddot]

    Parameters
    ----------
    span_transition: float
        The span of transition from sample to sample in [seconds].
    dynamic_states: int
        The number of dynamic states to keep track of.
    dimensionality: int
         The number of spatial dimensions involved in the problem.

    Returns
    -------
    state_transition_matrix_full: np.ndarray
         (num_states*dimensionality, num_states*dimensionality) state transition matrix.
    """
    stm_array = polynomial_state_transition_matrix(span_transition, dynamic_states)
    state_transition_matrix_full = lin_alg.concatenated_block_diagonal_matrix(stm_array, dimensionality-1)
    return state_transition_matrix_full
