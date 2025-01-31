from controller import Controller
import jax.numpy as jnp

class PID_Controller(Controller):
    def __init__(self):
        super().__init__()
        self.kp = self.config.getfloat('PID', 'kp')
        self.ki = self.config.getfloat('PID', 'ki')
        self.kd = self.config.getfloat('PID', 'kd')
        self.integral = 0
        self.prev_error = 0
    
    def get_parameters(self):
        return jnp.array([self.kp, self.ki, self.kd])
    
    def update(self, error, dt, parameters, state):
        """
        Compute the PID control signal.

        Parameters:
            error (float): The current error.
            dt (float): Time step.

        Returns:
            control_signal (float): The computed control signal.
        """
        # Proportional term
        proportional = parameters[0] * error
        
        # Integral term
        #self.integral += error * dt OLD
        state['integral'] = error*dt + state.get('integral', 0)
        integral = parameters[1] * state['integral']
        
        # Derivative term

        derivative = parameters[2] * ((error - state.get('prev_error', 0)) / dt)
        state['prev_error'] = error
        
        # Control signal
        control_signal = proportional + integral + derivative
        
        # Debugging
        return control_signal
