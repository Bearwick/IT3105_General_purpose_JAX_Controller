from controller import Controller
import jax.numpy as jnp

class PID_Controller(Controller):
    def __init__(self):
        super().__init__()
        self.kp = self.config.getfloat('PID', 'kp')
        self.ki = self.config.getfloat('PID', 'ki')
        self.kd = self.config.getfloat('PID', 'kd')
    
    def get_parameters(self):
        return jnp.array([self.kp, self.ki, self.kd])
    
    def update(self, error, dt, parameters, state):
        # Proportional term
        proportional = parameters[0] * error
        
        # Integral term
        state['integral'] = error*dt + state.get('integral', 0)
        integral = parameters[1] * state['integral']
        
        # Derivative term
        derivative = parameters[2] * ((error - state.get('prev_error', 0)) / dt)
        state['prev_error'] = error
        
        # Control signal
        control_signal = proportional + integral + derivative
        
        # Debugging
        return control_signal
