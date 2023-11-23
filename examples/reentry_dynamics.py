"""
[1] Unscented filtering and nonlinear estimation. Julier and Uhlmann. 2004.

"""
import src.utils.numerical_integration as num_integ
import matplotlib.pyplot as plt
import numpy as np


def _reentry_function(t, x):
    r = np.linalg.norm(x[0:2])
    v = np.linalg.norm(x[2:4])
    b = b0 * np.exp(x[4])
    D = b * np.exp((R0 - r) / H0) * v

    G = -Gm0 / r ** 3
    xdot = np.zeros_like(x, dtype=np.float64)

    xdot[0] = x[2]
    xdot[1] = x[3]
    xdot[2] = D * x[2] + G * x[0]
    xdot[3] = D * x[3] + G * x[1]
    # The SDE method will add the w1,2,3 terms xdot[2,3,4] by multiplying with matrix L.
    return xdot


def radar_measurement(x, x_radar):
    y_measurement = np.zeros([2], dtype=np.float64)
    displacement_x = x[0] - x_radar[0]
    displacement_y = x[1] - x_radar[1]
    y_measurement[0] = np.sqrt(displacement_x ** 2 + displacement_y ** 2)
    y_measurement[1] = np.arctan2(displacement_y, displacement_x)
    return y_measurement


def radar_observation(y_measurement, R):
    y_measurement += np.random.multivariate_normal([0.0, 0.0], R)
    return y_measurement


if __name__ == '__main__':
    # Define time variable
    delta_time = 0.1  # in [s]
    time_end = 200.  # [s]
    time_array = np.arange(0, time_end, delta_time, dtype=np.float64)

    # Define the constants from ref [1]_
    b0 = -0.59783
    H0 = 13.406
    Gm0 = 3.9860e5
    R0 = 6374

    Qk = np.diag([2.4064e-5, 2.4064e-5, 1e-6])
    Qc = Qk / delta_time

    # L : dispersion matrix
    L = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

    true_P0 = np.diag([1e-6, 1e-6, 1, 1e-6, 1e-6, 0.0])
    true_Qc = Qc
    true_Qc[2, 2] = 1e-6
    true_m0 = np.array([6500.4, 349.14, -1.8093, -6.7967, 0.6932], dtype=np.float64)

    m0 = np.array([6500.4, 349.14, -1.8093, -6.7967, 0.0], dtype=np.float64)
    P0 = true_P0
    P0[4] = 1.0

    xradar = np.array([R0, 0], dtype=np.float64)
    # Define state vector
    # x[0],x[1] -> x & y position
    # x[2],x[3] -> x & y velocity
    # x[4] -> parameter of the vehicle's aerodynamic properties

    x_state = num_integ.euler_maruyama_integration(true_m0, _reentry_function, time_array, L, true_Qc * delta_time)
    print('Truth data generated.')
    x_state_strong = num_integ.stochastic_runge_kutta_15(true_m0, _reentry_function, time_array, L, true_Qc)

    fig = plt.figure(1)
    fig.suptitle('Reentry tracking problem')
    plt.plot(x_state[0, :], x_state[1, :], 'g', label='EM method')
    plt.plot(x_state_strong[0, :], x_state_strong[1, :], 'm', label='IT-1.5 method')
    aa = 0.02 * np.arange(-1, 4, 0.1, dtype=np.float64)
    cx = R0 * np.cos(aa)
    cy = R0 * np.sin(aa)
    plt.plot(xradar[0], xradar[1], 'k', marker='o', label='Radar')
    plt.plot(cx, cy, 'r', label='Earth')
    plt.legend(loc='best')
    plt.axis([6340, 6520, -200, 600])
    plt.ylabel('y [km]')
    plt.xlabel('x [km]')
    ax = plt.gca()
    ax.grid(True)
    plt.show()
    fig.savefig('Pj00_reentry.png', bbox_inches='tight', pad_inches=0.01)

