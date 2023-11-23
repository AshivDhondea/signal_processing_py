import src.radar.radar_constants as constants
import logging
import math

logging.basicConfig(level=logging.INFO)


def bistatic_received_power(power_tx: float, gain_tx: float, gain_rx: float, rho_rx: float, rho_tx: float,
                            wavelength: float, radar_cross_section: float) -> float:
    if rho_rx == 0. or rho_tx == 0.:
        dbz = "Distance to receiver or transmitter cannot be zero."
        logging.error(dbz)
        raise ZeroDivisionError(dbz)
       
    denominator: float = (4*math.pi)**3 * (rho_rx**2)*(rho_tx**2)
    numerator: float = power_tx*gain_tx*gain_rx * radar_cross_section*(wavelength**2)
    power_rx: float = numerator/denominator
    return power_rx
    

def bistatic_receiver_snr(power_rx: float, t0: float, bandwidth: float, radar_loss: float) -> float:
    if bandwidth or t0 or radar_loss == 0.:
        dbz = "Bandwidth or t0 or radar loss cannot be zero."
        logging.error(dbz)
        raise ZeroDivisionError(dbz)
        
    return power_rx/(constants.BOLTZMANN_CONSTANT*bandwidth*t0*radar_loss)
