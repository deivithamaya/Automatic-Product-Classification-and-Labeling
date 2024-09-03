

File : images_CNN.ipynb

## Functions and Processes

### 1. `encode_labels(train_data, val_data, test_data, column, category_level)`

This function encodes class labels into numeric format and then converts them into one-hot format for model training.

- **Input**: 
  - `train_data`, `val_data`, `test_data`: DataFrames containing training, validation, and test data.
  - `column`: The name of the column containing the labels.
  - `category_level`: The category level for label encoding.
- **Output**: 
  - `train_label_cate`, `val_label_cate`, `test_label_cate`: One-hot encoded labels.
  - `train_label_le`: Numerically encoded labels.
  - `num_classes`: Total number of classes.

**Description**: Loads the label encoder from a file if it exists; otherwise, creates a new one. Encodes labels and converts them into one-hot format for model training.

### 2. `preprocess_image(image_path)`

This function processes an image from a file, resizes it to a standard size, and normalizes it.

- **Input**: 
  - `image_path`: Path to the image file.
- **Output**: 
  - Preprocessed and normalized image.

**Description**: Reads and decodes the image, resizes it to specified dimensions, and normalizes pixel values to the range [0, 1].

### 3. `create_dataset(data, image_column, labels)`

This function creates a TensorFlow dataset from image paths and their corresponding labels.

- **Input**: 
  - `data`: DataFrame containing image paths and labels.
  - `image_column`: Name of the column containing image paths.
  - `labels`: One-hot encoded labels.
- **Output**: 
  - A TensorFlow dataset.

**Description**: Creates a TensorFlow dataset that applies the preprocessing function to each image and assigns the corresponding labels.

### 4. `build_basic_cnn(input_shape, num_classes)`

This function defines and builds a basic Convolutional Neural Network (CNN).

- **Input**: 
  - `input_shape`: Shape of the input images (height, width, channels).
  - `num_classes`: Number of classes for classification.
- **Output**: 
  - The built CNN model.

**Description**: Defines a convolutional neural network with several convolutional layers, max pooling, dropout, and batch normalization. The architecture is designed to extract features from images and perform classification.

### 5. `save_model(level_input, model, history)`

This function saves the model, configuration, weights, and training history to files.

- **Input**: 
  - `level_input`: Category level for which the model is saved.
  - `model`: Trained model.
  - `history`: Training history.
- **Output**: 
  - Files containing the model, configuration, weights, and history.

**Description**: Saves the trained model, its configuration, weights, and the training history to files for future use.

### 6. Model Evaluation

- **Predictions**: Makes predictions on the test dataset.
- **Metrics**: Calculates performance metrics such as F1-score, precision, and recall. Also generates a classification report.
- **Visualization**: Plots loss and accuracy curves during training to evaluate model performance.

**Description**: Evaluates the model using the test dataset. Computes and displays performance metrics and visualizes the loss and accuracy curves to analyze the model's training.





### -----------------------------### -----------------------------
### -----------------------------### -----------------------------
### -----------------------------### -----------------------------
### -----------------------------### -----------------------------
### -----------------------------### -----------------------------
### -----------------------------### -----------------------------
### -----------------------------### -----------------------------


File : images_Resente.ipynb


## Imports

The code begins by importing various libraries and modules necessary for the project:

- **TensorFlow and Keras**: The primary deep learning framework and its high-level API.
- **Pandas and NumPy**: For data manipulation and numerical computations.
- **Scikit-learn**: Used for preprocessing and utility functions like `LabelEncoder` and `compute_class_weight`.
- **OS**: For interacting with the operating system, e.g., listing directory contents.
- **Pickle**: For loading and saving Python objects, like the label encoder.

## Settings

Basic configuration settings for the project include:

- **`batch_size`**: Number of samples per gradient update.
- **`img_height` and `img_width`**: Dimensions to which all images are resized.
- **`data_dir`**: Directory containing the image data.

## Reading Data

The code reads filtered training, validation, and test datasets from CSV files using Pandas.

## Encode Labels

A function, `encode_labels`, is defined to encode the labels from the datasets using a `LabelEncoder`. If a saved label encoder exists (`label_encoder2.pkl`), it is loaded; otherwise, a new one is created and used to transform the labels. The labels are also converted into categorical format using one-hot encoding.

## Preprocess Image

A preprocessing function, `preprocess_image`, is defined to handle image loading, decoding, resizing, and padding. It ensures that all images are resized to a uniform size while maintaining aspect ratio. Images are normalized to have pixel values between 0 and 1.

A second function, `create_dataset`, creates TensorFlow datasets for training, validation, and testing by mapping image file paths to their corresponding labels using the preprocessing function.

## F1 Score Metric

A custom TensorFlow metric, `F1Score`, is implemented to calculate the F1 score during training and evaluation. It handles multi-class classification and supports different averaging methods, like 'macro' and 'weighted'.

## ResNet50 Model

The ResNet50 model is built using the Keras Functional API:

- **Base Model**: The pre-trained ResNet50 is used as the base model, with its top layers removed (`include_top=False`). The model's input and a specific layer’s output are used for further processing.
- **Data Augmentation**: Random horizontal flipping and slight rotations are applied to the input images.
- **Convolutional Layers**: Additional convolutional layers are added with batch normalization, dropout, and activation functions. This includes skip connections to form residual blocks.
- **Output Layer**: The output from the convolutional layers is flattened and passed through dense layers before the final softmax layer to predict the class probabilities.

## Model Compilation

The model is compiled with the following configurations:

- **Optimizer**: Adam optimizer with a learning rate of `1e-5`.
- **Loss Function**: Categorical cross-entropy for multi-class classification.
- **Metrics**: Accuracy and the custom F1 score metric.

## Callbacks

Two Keras callbacks are used:

- **Early Stopping**: Stops training when the validation loss does not improve for a specified number of epochs, restoring the best weights.
- **ReduceLROnPlateau**: Reduces the learning rate when the validation loss plateaus.



### -----------------------------### -----------------------------
### -----------------------------### -----------------------------
### -----------------------------### -----------------------------
### -----------------------------### -----------------------------
### -----------------------------### -----------------------------
### -----------------------------### -----------------------------
### -----------------------------### -----------------------------



Some clarifications about the training file for the text model and the fusion model.

We trained part of our colab models using the mlflow tool for tracking experiments. The training parameters and metrics were stored in a SQLite database and 
the artifacts, such as images and models, were stored in an artifacts folder.

When searching for models in these files, the experiment id for text models is used and for image models, the absolute paths are used.

It is also important to clarify that the training file for the fusion models is more of a test file so that everything runs correctly and can be executed, 
changing the paths, on the computer of a teammate who had a GPU at his disposal.

If you want to see the text experiments, you can ask for the database file, my_runs.db, and the artifacts folder. To run it, you have to execute some 
statements in the database so that you can see the artifacts. Assuming that you have both files, my_runs.db and /artifacts, the commands are the following.

* sqlite3 my_runs.db ## enter the database
* update runs set artifact_uri = replace(artifact_uri, '/content/drive/MyDrive/models_compartida', '/root_to_artefacts') ## update the path of the artifacts 
in their local location

* mlflow ui --backend-store-uri sqlite:///my_runs.db --default-artifact-root ./artifacts ## run the mlflow user interface to see the experiments, 
executions and metrics
