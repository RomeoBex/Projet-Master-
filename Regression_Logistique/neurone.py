import numpy as np

# Fonction d'activation (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dérivée de la fonction d'activation sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Classe pour le réseau de neurones
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialisation des poids avec l'initialisation Xavier/Glorot
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))

    def forward(self, inputs):
        # Propagation avant
        self.hidden_layer_activation = sigmoid(np.dot(inputs, self.weights_input_hidden))
        self.output = sigmoid(np.dot(self.hidden_layer_activation, self.weights_hidden_output))
        return self.output

    def train(self, inputs, targets, learning_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            # Propagation avant
            self.forward(inputs)

            # Calcul de l'erreur
            output_error = targets - self.output

            # Calcul des ajustements pour les poids de la couche de sortie
            output_delta = output_error * sigmoid_derivative(self.output)
            output_adjustments = np.dot(self.hidden_layer_activation.T, output_delta)

            # Calcul des ajustements pour les poids de la couche cachée
            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_activation)
            hidden_adjustments = np.dot(inputs.T, hidden_delta)

            # Mise à jour des poids
            self.weights_hidden_output += learning_rate * output_adjustments
            self.weights_input_hidden += learning_rate * hidden_adjustments

        print("Entraînement terminé!")

# Fonction pour tester le réseau
def test_network(neural_network, test_inputs):
    predictions = []
    for i in range(len(test_inputs)):
        prediction = neural_network.forward(test_inputs[i])
        predictions.append(prediction)
    return np.array(predictions)

# Données d'entrée
training_inputs =  np.random.randint(1, 11, size=(100, 2))


training_targets = (training_inputs.sum(axis=1) % 2 == 0).astype(int).reshape(-1, 1)
# Création d'un réseau de neurones avec 2 entrées, 2 neurones cachés et 1 sortie
neural_network = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Nombre d'itérations
num_iterations = 2

# Boucle d'entraînement itérative
for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}")
    
    # Entraînement du réseau
    neural_network.train(training_inputs, training_targets, learning_rate=0.1, epochs=10000)


    # Test du réseau
    predictions = test_network(neural_network, training_inputs)
    print(f"Predictions après l'itération {iteration + 1}:")
    print(predictions)


