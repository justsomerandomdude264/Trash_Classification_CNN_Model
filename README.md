## Trash Classification CNN Model

### What is this?

This project is a convolutional neural network (CNN) model developed for the purpose of classifying different types of trash items. The dataset used for training this model is the RealWaste electronic dataset, which is available on The UCI Machine Learning Repository. The dataset is provided by the Wollongong City Council and is licensed under CC BY 4.0.

The CNN model in this project utilizes the TinyVGG architecture, a compact version of the popular VGG neural network architecture. The model is trained to classify trash items into the following subcategories:

- Cardboard
- Food Organics
- Glass
- Metal
- Miscellaneous Trash
- Paper
- Plastic
- Textile Trash
- Vegetation

In total, there are 9 categories into which the trash items are classified.

For more details about the CNN architecture used in this project, you can refer to the [CNN Explainer](https://poloclub.github.io/cnn-explainer/) website.

### Info

Only 30% of the data from the Real Trash Dataset has been used and divided into an 80%-20% split of Train and Test.

The Repository contains 7 files:

1. **data_setup.py**: This file contains functions for setting up the data into datasets using ImageFolder and then turning it into batches using DataLoader. It also returns the names of the classes.

2. **model_builder.py**: This file contains a class which subclasses nn.Module and replicates the TinyVGG CNN model architecture with a few modifications here and there.

3. **engine.py**: This file contains three functions: `train_step`, `test_step`, and `train`. The previous two are used to train and test the model, respectively, and the last one integrates both to train the model.

4. **plotting.py**: This file contains functions to plot metrics like loss and accuracy using `plot_metrics`, and it also has a function `plot_confusion_Matrix` to plot the confusion matrix.

5. **predict.py**: This file can be run with `--image` and `--model_path` arguments to get the prediction of the model on the specified image path.

6. **utils.py**: This file contains functions to save the model in a specific folder with a changeable name.

7. **train.py**: This script uses all the files except `predict.py` and can take argument flags to change hyperparameters. It can be run with the following arguments:

    ```
    python train.py --train_dir TRAIN_DIR --test_dir TEST_DIR --learning_rate LEARNING_RATE --batch_size BATCH_SIZE --num_epochs NUM_EPOCHS
    ```

    Additionally, it is device agnostic, meaning it automatically utilizes available resources regardless of the specific device used.

Additionally, the repository contains 2 folders:

- **data**: This stores the data and has subdirectories train and test.

- **models**: This stores the model saved by utils.py.


### What I Learned?

This project taught me the basics of **Computer Vision** with **PyTorch**, a lot about **Convolutional Neural Networks (CNNs)**, and also taught me how to **model** my project. It also taught me how to write **readable code** and handle **errors**, especially in the `predict.py` file.

I gained understanding about **classification** and how to implement it with **neural networks** and **deep learning**. While working on this, I learned the basics of **matplotlib** and **mlxtend** and also realized the impact of **data quantity** on results, which led to the decision of using only **30% of the data**.

I found that the best working **optimizer** with **TinyVGG** was an **Adam Optimizer** with a **learning rate** of **0.001**, trained on **20 epochs** and a **batch size** of **32** with **15** **hidden units**. This resulting in **Train Loss** of _**0.24**_ and **Test Loss** of _**2.17**_, **Train Accuracy** of _**91%**_ and **Test Accuracy** of _**55%**_.   


### Citation

If ever you use this model or dataset in your research or project, please cite the following:

S. Single, S. Iranmanesh, R. Raad, RealWaste, electronic dataset, The UCI Machine Learning Repository, Wollongong City Council, CC BY 4.0
