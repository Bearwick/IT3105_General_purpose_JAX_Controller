from controller import Controller
import jax.numpy as jnp
import numpy as np

class AI_Controller(Controller):
    def __init__(self):
        super().__init__()
        self.layers = self.config.getint('Config', 'NN_layers')
        self.neurons = self.config.getint('Config', 'NN_neurons')
        self.activation_func = self.config.getint('Config','activation_function')
        self.weight_initial_min = self.config.getfloat('Config', 'weight_initial_min')
        self.weight_initial_max = self.config.getfloat('Config', 'weight_initial_max')
        self.bias_initial_min = self.config.getfloat('Config', 'bias_initial_min')
        self.bias_initial_max = self.config.getfloat('Config', 'bias_initial_max')
        self.learning_rate = self.config.getfloat('Config', 'learning_rate')

    # TODO: input must be from config
    def gen_jaxnet_params(self, layers=[5, 10, 5]): #gen_jaxnet_params
        sender = layers[0]
        params = []

        for receiver in layers[1:]:
            weights = np.random.uniform(self.weight_initial_min, self.weight_initial_max, (sender, receiver))
            biases = np.random.uniform(self.bias_initial_min, self.bias_initial_max, (1, receiver))
            sender = receiver
            params.append([weights, biases])

        return params
    
    def activation_function(self, x):
        # 0=Sigmoid, 1=Tanh, 2=RELU
        if self.activation_func == 0:
            return 1/(1+jnp.exp(-x))
        elif self.activation_func == 1:
            return 2/(1+jnp.exp(-2*x))-1
        elif self.activation_func == 2:
            return np.max(0, x)
        else:
            raise ValueError('Unknown activation function number')

    def predict(self, all_params, features):
        activations = features
        for weights, biases in all_params:
            activations = self.activation_function(jnp.dot(activations, weights) + biases)
        return activations

    def get_parameters(self):
        return self.gen_jaxnet_params([5,10,5])
    
    def update(self, error, dt, parameters, state):
        
        control_signal = self.predict(parameters, state['features'])
        return control_signal