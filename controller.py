import jax.numpy as jnp
from jax import grad, jit
import configparser

class Controller:
   
    def __init__(self):
        self.error_history = []
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
    
    def compute(self, error, dt):
        raise NotImplementedError("Subclasses must implement this method.")

class PID_Controller(Controller):
    def __init__(self):
        super().__init__()
        self.kp = self.config.getfloat('PID', 'kp')
        self.ki = self.config.getfloat('PID', 'ki')
        self.kd = self.config.getfloat('PID', 'kd')
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        """
        Compute the PID control signal.

        Parameters:
            error (float): The current error.
            dt (float): Time step.

        Returns:
            control_signal (float): The computed control signal.
        """
        self.error_history.append(error)
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        
        # Control signal
        control_signal = proportional + integral + derivative
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
    
    def compute(self, error, dt):
        return 0
   