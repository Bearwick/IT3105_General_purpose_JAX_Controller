from controller import Controller
import jax.numpy as jnp
import numpy as np
import jax

class AI_Controller(Controller):
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
        
    def generate_layers(self):
        layers = [3]
        for hl in range(self.hidden_layers):
            layers.append(self.neurons)
        layers.append(1)
        return layers

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

        if state['nn_layer'] == self.hidden_layers:
            self.activation_func = self.out_activation_func

        if self.activation_func == 0:
            return jax.nn.sigmoid(x)
        elif self.activation_func == 1:
            return jax.nn.tanh(x)
        elif self.activation_func == 2:
            return jax.nn.relu(x)
        else:
            raise ValueError('Unknown activation function number')

    def predict(self, all_params, features, state):
        activations = features
        state['nn_layer'] = 0
        for weights, biases in all_params:
            state['nn_layer'] += 1
            activations = self.activation_function(jnp.dot(activations, weights) + biases, state)
        return activations

    def get_parameters(self):
        return self.gen_jaxnet_params() 
    
    def update(self, error, dt, parameters, state): 
        proportional = error

        state['integral'] = error*dt + state.get('integral', 0)
        integral = state['integral']

        derivative = (error-state.get('prev_error', 0))/dt
        state['prev_error'] = error

        features = jnp.array([proportional, integral, derivative])

        control_signal = self.predict(parameters, features, state)

        return jnp.mean(control_signal)
    

    #TODO
    # Is it correct to do mean on the control_signal?

    #           H_layers    Neurons     Everage Error       PID

    # Bathtub Results:                                      0.04
    # Sigmoid:  2           300         0.1544              
    # Tanh:     2           5           0.1925
    # Relu:     2           5           0.1887

    # Cournot Results:                                      0.02<
    # Sigmoid:  2           5           0.0206
    # Tanh:     2           5           0.0175
    # Relu:     2           5           0.0163

    # Rabbit Results:                                       ≈0.01
    # Sigmoid:  2           5           1237382
    # Tanh:     2           5           0.9774
    # Relu:     2           5           1267264