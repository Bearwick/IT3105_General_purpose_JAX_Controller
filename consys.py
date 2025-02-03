import configparser
import numpy as np
import bathtub
import cournot_competition
import robot_treadmill
import rabbit_population
import PID_controller
import AI_controller
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

class CONSYS():
    def __init__(self):
       self.read_config()
       self.error_history = []

    def read_config(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.plant_type = self.config.getint('Config', 'plant')
        self.controller_type = self.config.getint('Config', 'controller')
        self.epochs = self.config.getint('Config', 'epochs')
        self.timesteps = self.config.getint('Config', 'timesteps')
        self.dt = self.config.getint('Config', 'timestep_duration')
        self.min_disturbance = self.config.getfloat('Config', 'min_disturbance')
        self.max_disturbance = self.config.getfloat('Config', 'max_disturbance')
        self.learning_rate = self.config.getfloat('Config', 'learning_rate')

    def generate_disturbance_series(self):
        return np.random.uniform(self.min_disturbance, self.max_disturbance, self.timesteps)

    def initialize_controller(self):
        if self.controller_type == 0:
            return PID_controller.PID_Controller()
        elif self.controller_type == 1:
            return AI_controller.AI_Controller()
        else:
            raise ValueError('Unknown Controller type number')
        
    def initialize_plant(self):    
        if self.plant_type == 0:
            self.target = self.config.getfloat('Bathtub_Model','target')
            return bathtub.Bathtub()
        elif self.plant_type == 1:
            self.target = self.config.getfloat('Cournot_Competition','target')
            return cournot_competition.Cournot_Competition()
        elif self.plant_type == 2:
            self.target = self.config.getfloat('Rabbit_Population','target')
            return rabbit_population.Rabbit_Population()
        elif self.plant_type == 3:
            self.target = self.config.getfloat('Robot_Treadmill', 'target')
            return robot_treadmill.Robot_Treadmill()
        else:
            raise ValueError('Unknown Plant type number')
    
    def compute_MSE(self, error_history):
        error_array = jnp.array(error_history)
        MSE = jnp.mean(jnp.square(error_array))
        return MSE
    
    def update_parameters(self, parameters, gradients):
        updated_params = jax.tree_util.tree_map(
        lambda p, g: p - self.learning_rate * g, parameters, gradients
        )
        return updated_params

    def run_one_epoch(self, parameters, state):

        # a) Initialize any other controller variables, such as the error history (already initialized k values ad weights and biases for nn).
        # a) Reset the plant to its initial state
        plant = self.initialize_plant()
        self.error_history = []

        # b) Generate a vector of random disturbance
        disturbance_series = self.generate_disturbance_series()
        U = 0
        
        # c) for each timestep
        print(f"Timestep\tPrediction \tControl Output (U)")
        for t in range(self.timesteps):
            # update the plant
            Y = plant.update(U, disturbance_series[t])
            E = self.target - Y
            # update the controller
            U = self.controller.update(E, self.dt, parameters, state)
            # save the error (E) for this timestep in an error history
            self.error_history.append(E)

            print(f"{t+1}\t\t{Y.item()} \t\t{U.item():.4f}")

        # d) Compute the MSE over the error history
        MSE = self.compute_MSE(self.error_history)
        return MSE

    def run_system(self):
        # 1. Initialize the controller's parameters.
        self.controller = self.initialize_controller()

        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0)
        parameters = self.controller.get_parameters()
        MSE_history = []
        parameter_history = [parameters]

        # 2. For each epoch
        for _ in range(self.epochs):
      
            state = {}
            # e) compute the gradients
            avg_error, gradients = gradfunc(parameters, state)
            MSE_history.append(avg_error)
            # f) Update the controllers parameters (k values for classic and the weights and biases for NN)
            parameters = self.update_parameters(parameters, gradients)
            parameter_history.append(parameters)
            
            print(f'Average error: {avg_error}')
        
        self.visualize(MSE_history, parameter_history)
           
    def visualize(self, MSE_history, parameter_history=0):
        self.plot_MSE(MSE_history)

        if self.controller_type == 0:
            self.plot_parameters(parameter_history)
        plt.show()

    def plot_MSE(self, MSE_history):
        plt.figure()
        plt.plot(range(len(MSE_history)), MSE_history, label='MSE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title('Learning Progression')
        plt.legend()
         
    def plot_parameters(self, parameter_history):
        kp_history = [params[0] for params in parameter_history]
        ki_history = [params[1] for params in parameter_history]
        kd_history = [params[2] for params in parameter_history]

        plt.figure()
        plt.plot(range(len(kp_history)), kp_history, label='Kp')
        plt.plot(range(len(ki_history)), ki_history, label='Ki')
        plt.plot(range(len(kd_history)), kd_history, label='Kd')
        plt.xlabel('Epochs')
        plt.ylabel('Parameter Values')
        plt.title('Control Parameters')
        plt.legend()

if __name__ == "__main__":
    consys = CONSYS()
    consys.run_system()
   