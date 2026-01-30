U-Net Semantic Segmentation with TensorFlow/Keras  
A comprehensive computer vision project that implements the U-Net architecture for pixel-wise semantic segmentation using TensorFlow and Keras. The project focuses on dense prediction through an encoder–decoder network with skip connections, enabling accurate multi-class segmentation of driving scene images.

## Features
- End-to-end U-Net semantic segmentation pipeline  
- Custom data pipeline using `tf.data.Dataset`  
- Multi-class segmentation with sparse categorical labels  
- Encoder–decoder architecture with skip connections  
- Visualization of predictions against ground truth masks  
- Modular and educational implementation  

## Model & Framework
- Model: U-Net  
- Framework: TensorFlow 2.x / Keras  
- Task: Multi-class semantic segmentation  
- Input Shape: 96 × 128 × 3 (RGB)  
- Output: 96 × 128 segmentation mask  
- Classes: 23 semantic categories  

## Dataset
- Source: CARLA Driving Simulator  
- Inputs: RGB camera images  
- Targets: Pixel-wise segmentation masks  
- Preprocessing: resizing, normalization, mask channel reduction  

## Core Components
- Efficient data loading and preprocessing using `tf.data`  
- Encoder blocks with convolution, ReLU, and max-pooling  
- Decoder blocks with transposed convolutions and skip connections  
- Final 1×1 convolution layer for multi-class logits  

## Training Configuration
- Loss: Sparse Categorical Crossentropy (from logits)  
- Optimizer: Adam  
- Epochs: 40  
- Batch Size: 32  
- Shuffling enabled with buffer size 500  

## Evaluation & Visualization
- Visual comparison of input images, ground truth masks, and predictions  
- Accuracy trend visualization during training  
- Batch-wise prediction inspection utilities  


## Testing & Validation
- Model architecture verified using `model.summary()`  
- Layer-wise validation using custom comparator utilities  
- Encoder and decoder block configurations tested  

## Dependencies
- Python 3.x  
- TensorFlow 2.x  
- Keras  
- NumPy  
- Matplotlib  

## References
- U-Net: Convolutional Networks for Biomedical Image Segmentation – Ronneberger et al.  
- TensorFlow Documentation – tf.data API  
- Keras Functional API  

## License
This project is intended for educational and research purposes.  
Free to use and modify with proper attribution.

