from typing import Union
import logging
import math
import numpy as np

logging.basicConfig(level=logging.INFO)

 
def power_to_decibel(p: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    if isinstance(p, np.ndarray):
        return 10.*np.log10(p)  # power in watts.
    else:
        return 10.*math.log10(p)


def decibel_to_power(decibel: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    if isinstance(decibel, np.ndarray):
        return np.power(10.*np.ones_like(decibel), 0.1*decibel)
    else:
        return math.pow(10., 0.1*decibel)


def wavelength_or_frequency(speed_light: float, frequency: float) -> float:
    if frequency == 0.:
        dbz = "Cannot have frequency or wavelength equal to zero."
        logging.error(dbz)
        raise ZeroDivisionError(dbz)
    
    return speed_light/frequency
    

def monostatic_range_resolution(speed_light: float, bandwidth: float) -> float:
    if bandwidth == 0.:
        dbz = "Cannot have zero bandwidth."
        logging.error(dbz)
        raise ZeroDivisionError(dbz)
    return speed_light/(2.*bandwidth)
    

def monostatic_velocity_resolution(wavelength: float, pulse_width: float) -> float:
    if pulse_width == 0.:
        dbz = "Cannot divide by zero pulse_width."
        logging.error(dbz)
        raise ZeroDivisionError(dbz)
    return wavelength/(2.*pulse_width)
    
    
def time_bandwidth_product(pulse_width: float, bandwidth: float) -> float:
    return pulse_width*bandwidth
    
    
def calculate_search_volume(azimuth_angle: float, elevation_angle: float) -> float:
    """
    az,el in deg
    eqn 1.61 in Mahafza book
    """
    return azimuth_angle*elevation_angle/(57.296**2)  # steradians
