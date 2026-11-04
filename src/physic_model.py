from scipy.stats import qmc
import numpy as np

class Helicopter:
    def __init__(self, data):
        self.rho = np.array(data.get('rho'))
        self.f_factor = np.array(data.get('f_factor'))
        self.weight = np.array(data.get('weight'))
        self.k_factor = np.array(data.get('k_factor'))
        self.rotor_radius = np.array(data.get('rotor_radius'))
        self.velocity = np.array(data.get("velocity"))

    def CalculateDrag(self):
        Dp = 0.5 * self.rho * self.velocity**2 * self.f_factor
        Di = (self.k_factor * self.weight**2) / (2 * self.rho * self.velocity**2 * (np.pi) * self.rotor_radius**2)
        return Dp, Di



  

    