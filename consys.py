import configparser
import numpy as np
import bathtub
import controller

class CONSYS():
    def __init__(self):
       self.read_config()
       self.controller = self.initialize_controller()

    def read_config(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        self.plant_type = config.getint('Config', 'plant')
        self.controller_type = config.getint('Config', 'controller')
        self.epochs = config.getint('Config', 'epochs')
        self.timesteps = config.getint('Config', 'timesteps')
        self.min_disturbance = config.getfloat('Config', 'min_disturbance')
        self.max_disturbance = config.getfloat('Config', 'max_disturbance')

    def generate_disturbance_series(self):
        return np.random.uniform(self.min_disturbance, self.max_disturbance, self.timesteps)

    def initialize_controller(self):
        if self.controller_type == 0:
            return controller.PID_Controller()
        elif self.controller_type == 1:
            return controller.AI_Controller()
        else:
            raise ValueError('Unknown Controller type number')
        
    def initialize_plant(self):    
        if self.plant_type == 0:   
            return bathtub.Bathtub()
        elif self.plant_type == 1:
            raise ValueError('Cournot')
        elif self.plant_type == 2:
            raise ValueError('Plant 3')
        else:
            raise ValueError('Unknown Plant type number')

    def run(self):

        for _ in range(self.epochs):

            # a) Initialize any other controller variables, such as the error history (already initialized k values ad weights and biases for nn).
            # a) Reset the plant to its initial state
            plant = self.initialize_plant()

            # b) Generate a vector of random disturbance
            disturbance_series = self.generate_disturbance_series()

            # c) for each timestep
            for t in range(self.timesteps):
                # update the plant
                # update the controller
                # save the error (E) for this timestep in an error history
                pass

            # d) Compute the MSE over the error history

            # e) Compute the gradients

            # f) Update the controllers parameters (k values for classic and the weights and biases for NN)

           
            # Simulation settings
            target_height = 0.5  # Desired water height (m)
            timesteps = 10
            dt = 1.0  # Time step duration (seconds)
        
            ### Chat
            print("Timestep\tWater Height (H)\tControl Output (U)")
            for t in range(timesteps):
                # Compute error
                error = target_height - plant.H

                # Compute control output
                U = self.controller.compute(error, dt)

                # Update the model with control input and disturbance
                D = disturbance_series[t] # Constant disturbance flow rate (m^3/s)
                H = plant.update(U, D)

                print(f"{t+1}\t\t{H:.4f} m\t\t{U:.4f}")

if __name__ == "__main__":
    consys = CONSYS()
    consys.run()
   