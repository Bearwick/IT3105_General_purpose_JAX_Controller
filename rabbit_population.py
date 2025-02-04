import configparser

class Rabbit_Population():

    # Initializes the plant specific parameters
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        self.growth_rate = config.getfloat('Rabbit_Population', 'growth_rate')
        self.population = config.getfloat('Rabbit_Population', 'initial_population')
    
    # Receives the error and disturbance, and returns the new rabbit population
    def update(self, U, D):
        breeded_rabbits = self.compute_brewing_growth()
        animal_control = self.compute_animal_control_management(U, D)
        self.population += breeded_rabbits + animal_control
        return self.population
    
    # Computes how many new rabbits are breeded each timestep.
    def compute_brewing_growth(self):
        return self.growth_rate*self.population
    
    # Computes how many rabbits the Animal Control Team (ACT) will add or remove.
    def compute_animal_control_management(self, U, D):
        return self.population*(U+D)