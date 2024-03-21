import torch
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import model_builder

def plot_confusion_Matrix(model_path, dataloader, class_names, device, figsize=(12, 12)):
    """
    Generate and plot confusion matrix using mlxtend library from a PyTorch model and DataLoader.

    Args:
        model: PyTorch model's path eg(".pth" or ".pt").
        dataloader: DataLoader instance for the dataset.
        class_names (list): List of class names.
        device: Target device to compute on (e.g., "cuda" or "cpu").
        figsize (tuple): Figure size.

    Returns:
        None
    """

    # Load the model
    model = model_builder.TrashClassificationCNNModel(input_shape=3,
                                                      hidden_units=15,
                                                      output_shape=len(class_names)
                                                      )
    
    model.load_state_dict(torch.load(model_path))

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y_true.extend(y.cpu().numpy())
            y_logit = model(X)
            y_pred.extend(torch.argmax(y_logit, dim=1).cpu().numpy())

    confmat = confusion_matrix(y_target=y_true, y_predicted=y_pred, binary=False)
    
    # Plot confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=confmat, 
                                    class_names=class_names,
                                    figsize=figsize)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_metrics(metrics):
    """
    Plots training and testing loss and accuracy.

    Args:
        metrics (dict): A dictionary containing training and testing loss and accuracy.

    Returns:
        None
    """
    epochs = range(1, len(metrics['train_loss']) + 1)

    # Plot training and testing loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_loss'], 'b', label='Training loss')
    plt.plot(epochs, metrics['test_loss'], 'r', label='Testing loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_acc'], 'b', label='Training accuracy')
    plt.plot(epochs, metrics['test_acc'], 'r', label='Testing accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()
