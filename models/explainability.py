"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Alzheimer's Classification
Provides visual explanations for deep learning model decisions on MRI scans.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Computes Grad-CAM heatmaps for a given convolutional layer in a PyTorch model.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate a 2D heatmap [0, 1] for input_tensor.
        input_tensor: [1, C, H, W]
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # Target class score
        score = output[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling on gradients -> channel weights
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the activation maps
        activations = self.activations[0]  # [C, H, W]
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        # ReLU on weighted sum of activation maps
        heatmap = torch.mean(activations, dim=0).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        # Normalize to [0, 1]
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        return heatmap, target_class, output


def overlay_gradcam(original_image, heatmap, alpha=0.4, colormap=cm.jet):
    """
    Overlay Grad-CAM heatmap on top of original PIL or NumPy MRI image.
    original_image: PIL Image or NumPy array (H, W, 3) or (H, W) in [0, 255]
    heatmap: 2D NumPy array in [0, 1]
    alpha: Blending ratio (0.0 to 1.0)
    Returns: PIL Image of the superimposed visualization
    """
    if isinstance(original_image, Image.Image):
        orig_np = np.array(original_image.convert('RGB'))
    else:
        orig_np = np.array(original_image)
        if orig_np.ndim == 2:
            orig_np = np.stack([orig_np]*3, axis=-1)

    h, w = orig_np.shape[:2]

    # Resize heatmap to original image size
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_resized = heatmap_img.resize((w, h), Image.Resampling.BICUBIC)
    heatmap_resized_np = np.array(heatmap_resized) / 255.0

    # Apply colormap
    color_map = colormap(heatmap_resized_np)[:, :, :3]  # drop alpha
    color_map = np.uint8(255 * color_map)

    # Blend original and heatmap
    blended = np.uint8((1.0 - alpha) * orig_np + alpha * color_map)
    return Image.fromarray(blended)
