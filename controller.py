import configparser

class Controller:
   
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
    
    def update(self, error, dt):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_parameters(self):
        raise NotImplementedError("Subclasses must implement this method.")
   