import configparser
import jax.numpy as jnp

class Robot_Treadmill():
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        self.I_max = config.getfloat('Robot_Treadmill', 'I_max')
        self.Treadmill_length = config.getfloat('Robot_Treadmill', 'Treadmill_length')
        self.P = jnp.clip(config.getfloat('Robot_Treadmill', 'P0'), 0, self.Treadmill_length)
        self.robot_speed = config.getfloat('Robot_Treadmill', 'S0')
        self.timestep = 0
        self.S_t = self.I_max

    def update(self, U, D):
        self.S_t = self.compute_treadmill_speed(D)
        D_t = self.S_t
        self.robot_speed += U

        self.P += -D_t + self.robot_speed
        #self.P = jnp.clip(self.P, 0, self.Treadmill_length)
        self.timestep += 1
        return self.P
    
    def compute_treadmill_speed(self, D):
        s = self.I_max + jnp.cos(self.timestep/self.I_max)/self.I_max + D
        return s
    
