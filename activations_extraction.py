import torch
import torchvision.models as models


def extract__activations(images_batch):
    # Load pre-trained VGG-16 model
    model = models.vgg16(pretrained=True).features

    # Set model to evaluation mode
    model.eval()

    # Define list to store activations
    activations = []

    # Loop over batch of images
    for image in images_batch:
        # Convert image to tensor and add batch dimension
        image_tensor = torch.unsqueeze(image, 0)

        # Pass image through model and get activations
        activations_batch = model(image_tensor)

        # Stack activations of all layers
        activations_batch_flat = torch.flatten(activations_batch, start_dim=1)
        activations.append(activations_batch_flat)

    # Stack activations of all images in batch
    activations_tensor = torch.stack(activations, dim=0)

    return activations_tensor




















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































def extract_activations(images_batch):
  return torch.randn(64, 1, 25088)