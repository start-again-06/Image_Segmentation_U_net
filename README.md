# 🧠 U-Net for Semantic Segmentation in TensorFlow/Keras

This project demonstrates the creation and training of a **U-Net** model for **semantic segmentation** using TensorFlow and Keras. It includes data preprocessing, model construction, training, and visualization.

---

## 📌 Overview

- **Framework**: TensorFlow 2.x / Keras
- **Architecture**: U-Net (encoder-decoder with skip connections)
- **Dataset**: Carla driving simulator RGB images and segmentation masks
- **Input Size**: 96x128 RGB images
- **Output**: 96x128 mask with 23 class categories

---

## 🗂️ Project Structure

├── data/

│   ├── CameraRGB/  
         
│   └── CameraMask/
         
├── utils/      
            
├── notebook.py or main.py 
 

---

## 🖼️ Data Pipeline

1. **Loading image paths** from the RGB and mask directories
2. **Parsing & decoding** images using `tf.data.Dataset`
3. **Preprocessing**: resize to (96, 128), normalize, and reduce mask channels
4. **Batched and shuffled** dataset for training

---

## 🧱 Model Architecture: U-Net

- **Encoder (Contracting Path)**:
  - Convolution blocks with ReLU activations
  - Optional dropout and max-pooling
- **Decoder (Expanding Path)**:
  - Transposed convolutions (upsampling)
  - Skip connections from encoder
- **Output Layer**:
  - 1×1 convolution for multi-class logits

### 🔧 Model Summary

- Built using Keras Functional API
- Supports model inspection with `model.summary()` and test utilities like `comparator()`

---

## 🔁 Model Training

- **Loss**: Sparse Categorical Crossentropy (with logits)
- **Optimizer**: Adam
- **Epochs**: 40
- **Batch Size**: 32
- **Shuffling**: Enabled with buffer size 500

```python
unet.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)
model_history = unet.fit(train_dataset, epochs=40)

📈 Evaluation & Visualization

Visual comparison between input, ground truth, and predicted mask

Real-time plotting of accuracy trends

show_predictions(train_dataset, num=6)
plt.plot(model_history.history['accuracy'])

🧪 Unit Testing (Sample)

summary(model) outputs tested using a comparator

Verified layer configurations for encoder and decoder blocks

## 📚 References

- Olaf Ronneberger, Philipp Fischer, Thomas Brox – [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- TensorFlow 2.x Documentation – [tf.data API](https://www.tensorflow.org/guide/data)
- [Keras Functional API](https://keras.io/guides/functional_api/)
