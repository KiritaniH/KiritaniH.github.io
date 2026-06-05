
import numpy as np
import matplotlib.pyplot as plt

    
def show_minist(train_loader):
    """
    Show the first image of each digit in the MNIST dataset.

    Args:
        train_loader (torch.utils.data.DataLoader): The training data loader.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 4)) # create a figure with 2 rows and 5 columns
    axes = axes.flatten() # flatten the axes array for easy iteration
    for i in range(10):
        for data, target in train_loader:
            idx = (target == i).nonzero(as_tuple=True)[0][0] # get the first image of digit i
            axes[i].imshow(data[idx][0], cmap='gray') # show the image
            axes[i].set_title(f'{i}') # set the title to the digit
            axes[i].axis('off') # turn off the axis
            break 
    plt.show() # show the plot
    
    
def plot_loss(results, title="Loss"):
    """
        Plot the loss curve of the training process.

        Args:
            results (list): A list of loss values.
            title (str): The title of the plot.
    """
    plt.figure(figsize=(8, 5)) # set the size of the plot
    plt.plot(results) # plot the loss values
    plt.title(title) # set the title of the plot
    plt.xlabel('Iteration') # set the x-axis label
    plt.ylabel('Loss') # set the y-axis label
    plt.grid(True) # show the grid
    plt.show() # show the plot