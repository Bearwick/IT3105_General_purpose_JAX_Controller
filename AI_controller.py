from controller import Controller
import jax.numpy as jnp
import numpy as np
import jax

class AI_Controller(Controller):

    # Initializes the controller with its specific parameters.
    def __init__(self):
        super().__init__()
        self.hidden_layers = self.config.getint('Config', 'NN_layers')
        self.neurons = self.config.getint('Config', 'NN_neurons')
        self.activation_func = self.config.getint('Config','activation_function')
        self.out_activation_func = self.config.getint('Config','out_activation_func')
        self.weight_initial_min = self.config.getfloat('Config', 'weight_initial_min')
        self.weight_initial_max = self.config.getfloat('Config', 'weight_initial_max')
        self.bias_initial_min = self.config.getfloat('Config', 'bias_initial_min')
        self.bias_initial_max = self.config.getfloat('Config', 'bias_initial_max')
        self.learning_rate = self.config.getfloat('Config', 'learning_rate')
    
    # Generates a list of the Neural Network, i.e., length is layers and values is number of neurons.
    def generate_layers(self):
        layers = [3]
        for _ in range(self.hidden_layers):
            layers.append(self.neurons)
        layers.append(1)
        return layers

    # Generates the parameters, i.e., a list of tuples including weights and biases.
    def gen_jaxnet_params(self):
        layers = self.generate_layers()

        sender = layers[0]
        params = []

        for receiver in layers[1:]:
            weights = jnp.array(np.random.uniform(self.weight_initial_min, self.weight_initial_max, (sender, receiver)))
            biases = jnp.array(np.random.uniform(self.bias_initial_min, self.bias_initial_max, (1, receiver)))
            sender = receiver
            params.append((weights, biases))

        return params
    

    def activation_function(self, x, state):
        # 0=Sigmoid, 1=Tanh, 2=RELU
        if state['nn_layer'] == self.hidden_layers +2:
            self.activation_func = self.out_activation_func

        if self.activation_func == 0:
            return jax.nn.sigmoid(x)
        elif self.activation_func == 1:
            return jax.nn.tanh(x)
        elif self.activation_func == 2:
            return jax.nn.relu(x)
        else:
            raise ValueError('Unknown activation function number')

    # Makes a prediction on the features based on the neural network weights and biases. 
    def predict(self, all_params, features, state):
        activations = features
        state['nn_layer'] = 0
        for weights, biases in all_params:
            state['nn_layer'] += 1
            activations = self.activation_function(jnp.dot(activations, weights) + biases, state)
        return activations

    def get_parameters(self):
        return self.gen_jaxnet_params() 
    
    # Calculates the features and returns the control signal
    def update(self, error, dt, parameters, state): 
        proportional = error

        state['integral'] = error*dt + state.get('integral', 0)
        integral = state['integral']

        derivative = (error-state.get('prev_error', 0))/dt
        state['prev_error'] = error

        features = jnp.array([proportional, integral, derivative])

        control_signal = self.predict(parameters, features, state)

        return jnp.mean(control_signal)
    