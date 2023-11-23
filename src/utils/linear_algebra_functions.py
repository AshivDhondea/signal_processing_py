import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


def concatenated_block_diagonal_matrix(matrix_r: np.ndarray, stacking_length: int) -> np.ndarray:
    """
    Create a block diagonal matrix whose diagonal blocks are copies of matrix R.
    Parameters
    ----------
    matrix_r: np.ndarray
          (r, r) array for which the block diagonal matrix should be created.
    stacking_length: int
          The number of times that copies of matrix R should appear along the diagonal.

    Returns
    -------
    block_diagonal_matrix: np.ndarray
         (stacking_length*r, stacking_length*r) array representing the block diagonal matrix.

    Notes
    -----
    The concatenated block diagonal matrix function is an implementation of the concatenation equation from [1]_

    References
    -----
    [1] Tracking Filter Engineering, Norman Morrison. 2013.
    """
    # TODO: check if matrix_r should always be square. If yes, build logic to assert that.
    block_diagonal_array = np.kron(np.eye(stacking_length+1), matrix_r)
    return block_diagonal_array
