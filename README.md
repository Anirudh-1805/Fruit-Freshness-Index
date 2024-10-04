# Fruit Freshness Detection Using CNN

This project utilizes a Convolutional Neural Network (CNN) to classify the freshness of fruits (apple, banana, orange) into either "Fresh" or "Rotten" categories. The model is trained using TensorFlow and Keras.

## **Project Overview**

This project classifies fruit images as fresh or rotten using a CNN architecture. The dataset for the project is sourced from Roboflow's Freshness of Fruits dataset.

### **Dataset**

- The dataset consists of two sets: a training set and a test set.
- The dataset is automatically downloaded using the Roboflow API.

## **Installation Instructions**

1. Clone this repository to your local machine.
2. Ensure you have the following installed:
    - Python 3.x
    - TensorFlow
    - Keras
    - Roboflow Python package (`pip install roboflow`)
3. Use the following command to install any additional dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## **Data Preprocessing**

- **Training Set**: The training images are augmented using `ImageDataGenerator` with random rotations, width and height shifts, zoom, and flips for robust training.
- **Test Set**: The test set is rescaled to ensure consistency in pixel values.

## **Model Architecture**

The CNN model consists of:
1. **Convolutional Layers**: 4 convolutional layers with increasing filter sizes (32, 64, 128, 256) and ReLU activations.
2. **Pooling Layers**: MaxPooling after each convolutional layer to reduce spatial dimensions.
3. **Dropout**: Dropout layers to prevent overfitting.
4. **Fully Connected Layers**: Dense layer followed by the output layer with softmax activation for classification into 6 categories.

## **Training the Model**

To train the model, run:
```bash
cnn.fit(x = training_set, validation_data = test_set, epochs = 50)
```
