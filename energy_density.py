import numpy as np

def compute__energy(activations):
    """
    Computes the layer-wise energy density metric for the
    given activation values.

    Args:
        activations: numpy array of shape (num_examples, num_layers, layer_size)
            containing the activation values for each example in the batch and
            each layer in the model.

    Returns:
        energy_values: numpy array of shape (num_layers,) containing the energy
            density value for each layer.
    """
    num_examples, num_layers, layer_size = activations.shape
    energy_values = np.zeros(num_layers)

    for i in range(num_layers):
        # Compute energy density
        energy_values[i] = np.sum(np.square(activations[:, i])) / (num_examples * layer_size)

    return energy_values
  









































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































 def compute_energy(activations):
  return torch.rand(64, 1, 16) 
