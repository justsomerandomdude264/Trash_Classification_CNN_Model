"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def train_test_dataloader(train_dir: str,
                          test_dir: str,
                          transform: transforms.Compose,
                          batch_size: int):
    """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets using ImageFolder and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32)
  """
    # use ImageFolder to create the datasets
    dataset_train = ImageFolder(root=train_dir, transform=transform)
    dataset_test = ImageFolder(root=test_dir, transform=transform)

    # Get the Class Names
    class_names = dataset_train.classes

    # Make the DataLoaders
    train_dataloader = DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset_test,
                                 batch_size=batch_size,
                                 shuffle=True)
    
    return train_dataloader, test_dataloader, class_names