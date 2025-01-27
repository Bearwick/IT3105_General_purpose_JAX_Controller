import jax.numpy as jnp
from jax import grad, jit
import configparser

class Controller:
   
    def __init__(self):
        self.error_history = jnp.array([])
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
    
    def update(self, error, dt):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def append_error(self, error):
        self.error_history = jnp.append(self.error_history, error)  # Use JAX append

    def get_parameters(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def set_parameters(self):
        raise NotImplementedError('Subclasses must implement this method.')

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
    
    def set_parameters(self, new_parameters):
        self.kp, self.ki, self.kd = new_parameters

    def update(self, error, dt, parameters, state):
        """
        Compute the PID control signal.

        Parameters:
            error (float): The current error.
            dt (float): Time step.

        Returns:
            control_signal (float): The computed control signal.
        """
        #self.append_error(error)
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
       # print(f"Error: {error}, Proportional: {proportional}, Integral: {integral}, Derivative: {derivative}, Control Signal: {control_signal}")
        return control_signal


class AI_Controller(Controller):
    def __init__(self):
        super().__init__()
        """
        TODO:
        NN_layers=1
        NN_neurons=5
        activation_function=0
        weight_initial_min=0
        weight_initial_max=10
        bias_initial_min=0
        bias_initial_max=10
        learning_rate
        """
    
    def update(self, error, dt):
        return 0
   