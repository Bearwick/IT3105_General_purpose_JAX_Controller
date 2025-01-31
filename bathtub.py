import jax.numpy as jnp
import configparser


class Bathtub():
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        
        self.A = config.getint('Bathtub_Model', 'A')
        self.C = config.getint('Bathtub_Model', 'C')
        self.H = config.getfloat('Bathtub_Model', 'H0')
        self.g = config.getfloat('Bathtub_Model', 'g')
        
    def update(self, U, D):
        Q = self.compute_flow_rate()

        B = self.compute_bathtub_volume(U, D, Q)

        height_change = self.compute_water_height_change(B)

        self.H += height_change

        return self.H

    def compute_water_height_change(self, B):
        return B/self.A

    def compute_bathtub_volume(self, U, D, Q):
        return  U + D - Q

    def compute_velocity(self):
        return jnp.sqrt(2*self.g*self.H)
    
    def compute_flow_rate(self):
        return self.compute_velocity() * self.C