import matplotlib.pyplot as plt
import torch
from typing import List, Optional, Union

def plot(imgs: Union[torch.Tensor, List[torch.Tensor]], 
         row_title: Optional[List[str]] = None, 
         **imshow_kwargs):
    """Helper function to plot a batch of images.
    
    Args:
        imgs: A single torch tensor of shape (C,H,W) or a list of tensors
        row_title: Optional list of strings to use as titles for each image
        **imshow_kwargs: Optional arguments to pass to plt.imshow
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    
    num_imgs = len(imgs)
    fig = plt.figure(figsize=(num_imgs * 4, 4))
    
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1, num_imgs, i+1)
        if row_title is not None:
            ax.set_title(row_title[i])
            
        # Convert tensor to numpy and handle different tensor formats
        if isinstance(img, torch.Tensor):
            # If image is (C,H,W) format with C=3
            if img.dim() == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0)  # Convert to (H,W,C)
            img = img.detach().cpu().numpy()
            
        plt.imshow(img, **imshow_kwargs)
        plt.axis('off')
    
    plt.tight_layout()
    return fig
