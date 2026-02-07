# Neural Eyes           
## Brain Tumor Detection using Convolutional Neural Networks


This project presents a deep learning approach for detecting brain tumors from MRI images using a Convolutional Neural Network (CNN) implemented with TensorFlow and Keras.

The purpose of this project is to explore the application of computer vision techniques in medical imaging and to build a reliable binary classification model capable of distinguishing between tumorous and non-tumorous brain scans.

---

## Project Motivation

Brain tumors require timely and accurate diagnosis to improve treatment outcomes. Manual examination of MRI scans can be time-consuming, making automated detection systems valuable in supporting medical professionals.

This project allowed me to apply theoretical knowledge of deep learning to a real-world healthcare problem while strengthening my skills in neural networks, data preprocessing, and model evaluation.

---

## Dataset

The dataset consists of labeled MRI brain images divided into two categories:

- **Tumor:** MRI scans showing the presence of a tumor  
- **No Tumor:** MRI scans without abnormalities  

This problem is treated as a binary image classification task.

---

## Sample MRI Images

### Tumor Cases


![Tumor MRI](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/TUMOR%20IMAGE%202.jpg).




### Non-Tumor Cases


![No Tumor MRI](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/MRI%20Image%20no%20tumor.jpg)

Visual inspection of the images helped in understanding the patterns the model learns during training.

### Comparsion between tumor and no tumor MRI's

![comparsion](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/comparsion%20image.png).

### More tumor MRI's
![tumor MRI's](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/two%20Tumors%20.png).
![Circled Tumor](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/circle%20tumor.png).
## Data Preprocessing

To ensure consistency and improve model performance, the following preprocessing steps were applied:

- Resized all images to a uniform dimension  
- Normalized pixel values to the range of 0–1  
- Converted categorical labels into numerical format

- To improve model performance, cropping was applied to remove unnecessary background areas and isolate the brain region. This ensures that the neural network focuses only on the most relevant features during training.
so let me show you an example of what it looked like and how it looks like now:

![crop image MRI](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/cropped%20image.png)

now the brain is the main focus which is exactly what we want before sending the images to the model

---
### Before I split the data I wanted to check all the images so I plotted the Tumor Images and the clean ones 

![Plotting No Tumor MRI](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/no%20tummor%20plotting.png)

![Plotting Tumor MRI](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/plotting%20tumor.png)

## Data Split

The dataset was divided to support effective training and unbiased evaluation:

- 70% Training  
- 15% Validation  
- 15% Testing  

---

## Model Architecture

The CNN architecture was designed to balance predictive performance with computational efficiency.

**Core components include:**

- Convolutional layers for feature extraction  
- ReLU activation to introduce non-linearity  
- MaxPooling layers for spatial downsampling  
- Flatten layer to transform feature maps into vectors  
- Fully connected Dense layer  
- Sigmoid activation for binary classification  

---

## Training Strategy

The model was trained while monitoring validation metrics to reduce overfitting.

**Techniques used:**

- EarlyStopping to halt training when validation performance stopped improving  
- ModelCheckpoint to store the best-performing model  

---

## Training Performance


![Accuracy Curve](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/Accuracy.png)

see the model improving over time both training and validation accuracy are going up which is a very good sign near the end the training accuracy gets higher than validation which hints at a little overfitting but overall the model learned well and is performing strong


![Loss Curve](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/Loss.png)

The loss keeps dropping which means the model is learning and making fewer mistakes over time validation loss also goes down at the beginning but starts moving up a little later which suggests the model is starting to slightly overfit but overall the learning process looks healthy



The curves illustrate the model's learning progression across training epochs.


## Results

- **Test Accuracy:** ~86%  
- **Loss:** ~0.35  

Considering the dataset size, the model demonstrates strong classification capability and good generalization to unseen data.

---

## Testing the model on images he has never seen
Let's try a tumor image 

![Test1](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/there's%20a%20tumor1.png).

---
Now no tumor

![Test2](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/no%20tumor%201.png).


Providing confidence scores improves interpretability beyond a simple binary output.

---

## Testing from different angles

![Test3](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/no%20tumor%202.png).

![Test4](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/tumor%202.png).


What made this even more fun is watching the model improve and actually understand the patterns instead of memorizing images it performed strongly not only on familiar views but also on MRIs taken from completely different angles which shows that it learned how to recognize tumors regardless of position

---

## Technologies Used

- Python  
- TensorFlow  
- Keras  
- NumPy  
- Matplotlib  

---

## Quick Access

-  Dataset → [Open Dataset](https://github.com/Rana-Alsattari/NeuralEyes/tree/main/brain_tumor_dataset)
-  Augmented Data → [Open Augmented data](https://github.com/Rana-Alsattari/NeuralEyes/tree/main/augmented_data).
-  Model → [View Model](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/bestmodel.h5).
-  Notebook → [Check Notebooks](https://github.com/Rana-Alsattari/NeuralEyes/blob/main/NeuralEyes.ipynb)
-  Images → [See Images](https://github.com/Rana-Alsattari/NeuralEyes/tree/main/images).
---

## Key Learnings

Through this project, I developed practical experience in:

- Designing CNN models  
- Processing medical imaging data  
- Preventing overfitting  
- Evaluating classification models  
- Building an end-to-end deep learning pipeline  

---

## Future Work

Potential improvements include:

- Applying data augmentation to enhance robustness  
- Experimenting with transfer learning architectures  
- Increasing dataset size  
- Deploying the model as an interactive application  

---

## Conclusion

This project reflects my growing interest in artificial intelligence and its potential impact on healthcare. It represents a significant step in developing practical deep learning solutions for real-world problems.
