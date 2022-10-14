import matplotlib.pyplot as plt
import numpy as np

class Utility:
    def __init__(self):
        pass

    def visualize(self, **images):
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

    def denormalize(self, x):
        """
        Helper function for data visualization.
        Scale image to range 0..1 for correct plot.
        """
        x_max = np.percentile(x, 98)
        x_min = np.percentile(x, 2)    
        x = (x - x_min) / (x_max - x_min)
        x = x.clip(0, 1)
        return x

    def visualize_metrics(self, metrics: list, loss: list, model_history):
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        for m in metrics:
            plt.plot(model_history.history[m])
        plt.xlabel('Epoch')
        plt.legend(metrics, loc='upper left')

        plt.subplot(122)
        for l in loss:
            plt.plot(model_history.history[l])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loss, loc='upper left')

        plt.show()