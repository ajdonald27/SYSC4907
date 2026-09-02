# Velocity Calculator API for Python Reference Model 

# Author : AJ Donald
# Initial Rev: July 21st, 2026 
# Last Rev: July 21st, 2026 

# constant to compute velocity using c (speed of light)

SPEED_OF_LIGHT = 299_792_458 

def calculate_radar_wavelength(carrier_freq):
    """
    Calculate the radar wavelength

    Parameters:
        carrier_freq:
            Radar carrier freq in Hz (HB100 is 10.525e9)Hz 

    Returns:
        wavelength in meters
    """

    wavelength = SPEED_OF_LIGHT / carrier_freq

    return wavelength

def doppler_freq_to_velocity(doppler_freq, carrier_freq): 

    """
    Convert received Doppler freq to radial target velocity.

    For a monostatic Contiuous Wave (CW) radar (HB100):
        velocity = (Doppler Frequency * wavelength) / 2 

    The factor of two exists to account for the radar signal traveling between Tx->Target->Rx 

    Returns: 
        velocity in m/s 
    """
    wavelength = calculate_radar_wavelength(carrier_freq)

    velocity = (doppler_freq * wavelength) / 2

    return velocity

def convert_kmh(velocity):
    """
    Converts velocity in m/s to km/h
    """

    velocity_kmh = velocity * 3.6

    return velocity_kmh

