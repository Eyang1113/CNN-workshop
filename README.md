# Sports Equipment (balls) Classification using CNN
# 🏀 Sports Equipment Classification using VGG11 and Transfer Learning

This project presents a deep learning-based image classification system designed to identify five types of sports equipment with high accuracy. Leveraging **transfer learning** and the **VGG11 model** pre-trained on ImageNet, the model is fine-tuned specifically for sports object recognition.

## 🎯 Objective

To accurately classify images of:
- ⚾ Baseballs  
- 🏀 Basketballs  
- 🏸 Shuttlecocks  
- 🎾 Tennis Balls  
- 🏐 Volleyballs  

using a customized version of the VGG11 neural network.

## 🧠 Model Architecture

We use **VGG11**, a deep convolutional neural network originally trained on ImageNet, for feature extraction. Key steps include:
- **Freezing early layers** to retain pre-trained feature representations.
- **Replacing and retraining final layers** to tailor predictions for five sports equipment classes.

## 🧰 Preprocessing Techniques

To ensure robust model performance and generalization:
- 🖼️ **Image resizing** to 224x224 pixels
- 🎨 **Normalization** using ImageNet mean and standard deviation
- 🔄 **Data augmentation**: random flips, rotations, and color jitter


## 📌 Tools & Technologies

- **Framework**: PyTorch
- **Model**: VGG11 (pre-trained on ImageNet)
- **Metrics**: Accuracy, Confusion Matrix
- **Environment**: Jupyter Notebook / Google Colab
