import torch
import data_setup, model_builder, engine, utils, plotting

from torchvision import transforms
import argparse

# To Avoid Glitches related to CUDA
def set_memory_limit():
    if torch.cuda.is_available():
        try:
            torch.tensor([1], device='cuda') # Adjust memory fraction as needed
            print(f"Device is GPU/CUDA.")
            device = 'cuda'
            return device
        except:
            print("Device is CPU.")
            device = 'cpu'
            return device

# Define argument parsing directly
parser = argparse.ArgumentParser(description="Train a model for Classification of types of Trash.")
parser.add_argument("--train_dir", type=str, default="data/train", help="Directory containing training images")
parser.add_argument("--test_dir", type=str, default="data/test", help="Directory containing testing images")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train for")
args = parser.parse_args()

# Set Args up in Variables
train_dir = args.train_dir
test_dir = args.test_dir
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
HIDDEN_UNITS = 15

# Data transformation
data_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

# Create DataLoaders
train_dataloader, test_dataloader, class_names = data_setup.train_test_dataloader(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Model creation
device = set_memory_limit()
model = model_builder.TrashClassificationCNNModel(input_shape=3,
                                                  hidden_units=HIDDEN_UNITS,
                                                  output_shape=len(class_names)
                                                  ).to(device)

# Loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training
metrics = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=NUM_EPOCHS,
                       device=device)

# Save the model
utils.save_model(model=model,
                 target_dir="models",
                 model_name="Trash_Classification_Model_COLOURED.pth")

# Clear CUDA cache
torch.cuda.empty_cache()

# Plot the Confusion Matrix
plotting.plot_confusion_Matrix(model_path="models\Trash_Classification_Model_COLOURED.pth",
                               dataloader=test_dataloader,
                               class_names=class_names,
                               device=device)

# Plot the Loss and Accuracy
plotting.plot_metrics(metrics)