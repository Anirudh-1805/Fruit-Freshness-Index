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
## The Model is Compiled with:

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

## Model Evaluation

- After training, the model is evaluated on the test set.
- Accuracy and loss values are printed after each epoch.

## Making Predictions

You can make a single prediction with the trained model using a test image:

```python
test_image = image.load_img('path_to_image.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image / 255.0
result = cnn.predict(test_image)
```
The **Freshness Index** is calculated based on the output probabilities for fresh and rotten classes. Depending on the predicted class, the program will indicate whether the fruit is fresh or rotten and display the calculated freshness index.

## Conclusion

This project demonstrates the capabilities of CNNs in image classification tasks, particularly in determining the freshness of fruits based on visual features. The calculated **Freshness Index** provides a quantifiable measure of freshness, enhancing the understanding of the model's predictions.
