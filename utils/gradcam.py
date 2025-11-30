import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        
        return cam.data.numpy()[0, 0]

def visualize_cam(mask, img):
    """
    Overlays the CAM mask on the image.
    mask: 2D numpy array (0-1)
    img: PIL Image
    Returns: matplotlib figure
    """
    # Resize mask to image size
    img_np = np.array(img)
    heatmap = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
    
    # Colorize heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose
    superimposed_img = heatmap * 0.4 + img_np * 0.6
    superimposed_img = np.uint8(superimposed_img)
    
    # Plot
    fig, ax = plt.subplots()
    ax.imshow(superimposed_img)
    ax.axis('off')
    
    return fig

