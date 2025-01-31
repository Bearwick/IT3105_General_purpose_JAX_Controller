import configparser
import jax.numpy as jnp

class Cournot_Competition():
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.p_max = config.getfloat('Cournot_Competition', 'p_max')
        self.m_cost = config.getfloat('Cournot_Competition', 'c_m')
        self.q_1_production = config.getfloat('Cournot_Competition', 'q_1_0')
        self.q_2_production = config.getfloat('Cournot_Competition', 'q_2_0')

    def update(self, U, D):
        q_1 = self.compute_q_1(U)
        q_2 = self.compute_q_2(D)
        q = q_1 + q_2
        p_q = self.p_max - q

        return q_1 * (p_q - self.m_cost)
    
    def compute_q_1(self, U):
        self.q_1_production = jnp.clip(U, 0, 1) #self.enforce_constraints(U + self.q_1_production)
        return self.q_1_production

    def compute_q_2(self, D):
        self.q_2_production = jnp.clip(D, 0, 1)#self.enforce_constraints(D + self.q_2_production)
        return self.q_2_production
    
    def enforce_constraints(self, q):
        if q < 0:
            return 0
        elif q > 1:
            return 1
        return q
    
