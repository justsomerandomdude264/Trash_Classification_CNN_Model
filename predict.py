import torch
from torchvision import transforms
from PIL import Image
import model_builder
import matplotlib.pyplot as plt
import argparse

def predict_image(image_path: str, 
                  model_path: str,
                  class_names: list):
    """
    Predict class label of an image using a pre-trained model and plot the image with predicted class and probability as title.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the saved PyTorch model.
        class_names (list): List of class names.

    Returns:
        predicted_class (str): Predicted class label.
        probability_percentage (float): Probability percentage of the predicted class.
    """
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Define transformations for the image
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)

    # Load the model
    model = model_builder.TrashClassificationCNNModel(input_shape=3,
                                                      hidden_units=15,
                                                      output_shape=len(class_names)
                                                      )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Make predictions
    with torch.inference_mode():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze().tolist()
        predicted_index = torch.argmax(output, 1).item()
        probability_percentage = probabilities[predicted_index] * 100
        predicted_class = class_names[predicted_index]

    # Plot the image with predicted class and probability as title
    plt.imshow(image_tensor.squeeze().permute(1, 2, 0))
    plt.title(f'Predicted Class: {predicted_class}  |  Probability: {probability_percentage:.2f}%', fontdict={'family': 'serif',
                                                                                                              'color':  'black',
                                                                                                              'weight': 'normal',
                                                                                                              'size': 16,})
    plt.axis(False)
    plt.show()

    return predicted_class, probability_percentage

# Arguments to be passed while running
parser = argparse.ArgumentParser(description="For Prediction of an Image using a Loaded Model")
parser.add_argument("--image", type=str, default=None, help="Path to the image to make predictions on.")
parser.add_argument("--model_path", type=str, default=None, help="Path to the Trained Models State Dict.")
args = parser.parse_args()

IMAGE_PATH = args.image
MODEL_PATH = args.model_path

# Conditionals to check if Image path and Model path are entered
if not IMAGE_PATH:
    print("Please Enter Image Path using --image")
    raise SystemExit

if not MODEL_PATH:
    print("Please Enter Model PAth using --model_path")
    raise SystemExit

# Handle any errors related to wrong paths
try:
    # Run the function
    predict_image(image_path=IMAGE_PATH,
                model_path=MODEL_PATH,
                class_names=['Cardboard',
                            'Food Organics',
                            'Glass',
                            'Metal',
                            'Miscellaneous Trash',
                            'Paper',
                            'Plastic',
                            'Textile Trash',
                            'Vegetation'])
except Exception as exception:
    print("INVALID MODEL_PATH OR INVALID IMAGE PATH")
    print(f"\n{exception}")