import numpy as np

# TODO: fix docstrings and add test strings.


def runge_kutta_4th_vector(f, dt, x, t, q_matrix=None, l_matrix=None):
    """
    fnRK4_vector implements the Runge-Kutta fourth order method for solving Initial Value Problems.
    f : dynamics function
    dt : fixed step size
    x : state vector
    t : current time instant.
    Refer to Burden, Fairies (2011) for the RK4 method.
    """

    if q_matrix is None:  # Assuming L matrix is also None.
        # Execute one RK4 integration step
        k1 = dt * f(t, x)
        k2 = dt * f(t + 0.5 * dt, x + 0.5 * k1)
        k3 = dt * f(t + 0.5 * dt, x + 0.5 * k2)
        k4 = dt * f(t + dt, x + k3)
    else:
        # Execute one RK4 integration step
        k1 = dt * f(t, x, q_matrix, l_matrix)
        k2 = dt * f(t + 0.5 * dt, x + 0.5 * k1, q_matrix, l_matrix)
        k3 = dt * f(t + 0.5 * dt, x + 0.5 * k2, q_matrix, l_matrix)
        k4 = dt * f(t + dt, x + k3, q_matrix, l_matrix)

    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


# Stochastic numerical integration

def euler_maruyama_integration(x, nonlinear_function, time_array, l_matrix, q_matrix):
    """
    fnEuler_Maruyama implements the 0.5 strong Euler-Maruyama scheme. See Sarkka, Solin (2014).
    x : state vector of dimensions dx by 1.
    fnD : nonlinear dynamics function.
    timevec : time vector for simulation duration.
    L : dispersion matrix of dimensions dx by dw.
    Qd : discretized covariance matrix of dimensions dw by dw.
    """
    dt = time_array[1] - time_array[0]
    dx = np.shape(l_matrix)[0]  # dimension of state vector.
    # dw = np.shape(L)[1]; # dimension of process noise vector
    x_state = np.zeros([dx, len(time_array)], dtype=np.float64)
    x_state[:, 0] = x

    for index in range(1, len(time_array)):
        x_state[:, index] = (x_state[:, index - 1] +
                             dt * nonlinear_function(time_array[index - 1], x_state[:, index - 1]) +
                             np.dot(l_matrix, np.random.multivariate_normal(np.zeros(np.shape(l_matrix)[1],
                                                                                     dtype=np.float64), q_matrix)))
    return x_state


def stochastic_runge_kutta_15(x, nonlinear_function, time_array, l_matrix, q_matrix):
    """
    fnSRK_Crouse implements the 1.5 strong Stochastic Runge-Kutta method in Crouse (2015).
    Note that as opposed to Crouse (2015), we have assumed that the dispersion matrix is constant, i.e. not time-varying
    and definitely not state-dependent.
    x : state vector of dimensions dx by 1.
    fnD : nonlinear dynamics function.
    timevec : time vector for simulation duration.
    L : dispersion matrix of dimensions dx by dw.
    Qd : discretized covariance matrix of dimensions dw by dw.
    Note: 27/07/16: explicit order 1.5 strong scheme in section 11.2 in Kloden, Platen (1992) and Crouse(2015).
    """
    dw = np.shape(l_matrix)[1]  # dimension of process noise vector
    dx = np.shape(l_matrix)[0]  # dimension of state vector.

    x_state = np.zeros([dx, len(time_array)], dtype=np.float64)
    x_state[:, 0] = x

    dt = time_array[1] - time_array[0]

    # Form the covariance matrix for delta_beta and delta_alpha.
    beta_beta = dt * q_matrix
    beta_alpha = 0.5 * (dt ** 2) * q_matrix
    alpha_alpha = ((dt ** 3) / 3.0) * q_matrix
    Qd_aug = np.zeros([dw + dw, dw + dw], dtype=np.float64)  # The covariance matrix in eqn 44.
    Qd_aug[0:dw, 0:dw] = beta_beta
    Qd_aug[0:dw, dw:] = beta_alpha
    Qd_aug[dw:, 0:dw] = beta_alpha
    Qd_aug[dw:, dw:] = alpha_alpha

    # Generate process noise terms according to eqn 44.
    noise_array = np.zeros([dw + dw], dtype=np.float64)  # mean vector is zero.

    y_plus = np.zeros([dx, dw], dtype=np.float64)
    y_minus = np.zeros([dx, dw], dtype=np.float64)
    fy_plus = np.zeros([dx, dw], dtype=np.float64)
    fy_minus = np.zeros([dx, dw], dtype=np.float64)
    f2 = np.zeros([dx, dw], dtype=np.float64)  # equal to F2 (eqn 39)
    # F3 == F2 because we assume L is a constant matrix.

    for index in range(1, len(time_array)):
        process_noise = np.random.multivariate_normal(noise_array, Qd_aug)
        delta_beta = process_noise[0:dw]
        delta_alpha = process_noise[dw:]
        summ = np.zeros([dx], dtype=np.float64)

        for j in range(0, dw):
            # find yj+ and yj-. eqns 42 and 43
            y_plus[:, j] = (x_state[:, index - 1] + (dt / float(dw)) * nonlinear_function(time_array[index - 1],
                                                                                          x_state[:, index - 1]) +
                            np.sqrt(dt) * l_matrix[:, j])
            y_minus[:, j] = (x_state[:, index - 1] + (dt / float(dw)) * nonlinear_function(time_array[index - 1],
                                                                           x_state[:, index - 1]) -
                             np.sqrt(dt) * l_matrix[:, j])
            # expressions in eqns 40 and 38
            fy_plus[:, j] = nonlinear_function(time_array[index - 1], y_plus[:, j])
            fy_minus[:, j] = nonlinear_function(time_array[index - 1], y_minus[:, j])
            f2[:, j] = (1 / (2 * np.sqrt(dt))) * (fy_plus[:, j] - fy_minus[:, j])  # eqn 40
            # sum term in eqn 38
            summ += (fy_plus[:, j] - 2 * nonlinear_function(time_array[index - 1], x_state[:, index - 1]) +
                     fy_minus[:, j])

        f1 = (x_state[:, index - 1] + dt * nonlinear_function(time_array[index - 1], x_state[:, index - 1]) +
              0.25 * dt * summ)  # eqn 38
        x_state[:, index] = f1 + np.dot(l_matrix, delta_beta) + np.dot(f2, delta_alpha)  # eqn 37

    return x_state
