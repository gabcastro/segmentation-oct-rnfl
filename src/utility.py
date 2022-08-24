import matplotlib.pyplot as plt
import numpy as np

class Utility:

    def visualize(**images):
        """
        Helper function for data visualization.
        Plot images in one row.
        """
        n = len(images)
        plt.figure(figsize=(10, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image, cmap='gray')
        plt.show()

    def denormalize(x):
        """
        Helper function for data visualization.
        Scale image to range 0..1 for correct plot.
        """
        x_max = np.percentile(x, 98)
        x_min = np.percentile(x, 2)    
        x = (x - x_min) / (x_max - x_min)
        x = x.clip(0, 1)
        return x