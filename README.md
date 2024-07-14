
## Classification of Whale Sounds (Supervised Deep Learning Project)
The deep blue sea is home to a plethora of fascinating and mysterious creatures. Sound is the fundamental way these giants communicate with each other, but much of their communication remains a mystery to scientists. I unlocked the secrets of blue whale communication by building model that can accurately identify various calls made by these creatures.

### Input
#### What are whale vocalisations?
The different types of vocalizations and their purposes are still not well understood, and ongoing research continues to uncover new insights into the vocal behaviour of these fascinating animals. Here we focus on a specific vocalization,

🐋 A Calls: They are characterized by a low-frequency, repetitive pattern of pulses that are typically around 70-90 Hz in frequency. These calls are known to be used for long-distance communication between individual whales and can be heard over large distances. They are typically produced by adult males and can last for several minutes.

Here is how the graph can be visualised:





![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/273696d0-e39f-4183-9f70-e58ae097abdb)

The occurrences of A calls were identified and labelled from underwater audio recordings captured by deep-sea hydrophone devices. The recording spanned almost a month.

The number of samples per class:

No A calls (Labeled as 0): 12952

A calls (Labeled as 1): 12996

Unlabelled (test): 2000

Visualization of an audio file from the training dataset:
![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/da17d45e-3373-4a04-aa7c-5a1d838fcfd2)

### Dependency/ Library Used
os 

pathlib 

matplotlib

numpy

seaborn

tensorflow

keras

IPython

### Setting up and Importing the Data 
I used tensorflow's DatasetLoader to lazily load the dataset. The key characteristic of lazy loading is that the data is loaded incrementally as we iterate over the dataset. This approach is beneficial for large datasets that may not fit entirely into memory. By loading the data lazily, we can efficiently handle large datasets without running into memory limitations.

The number of samples included in each batch of the dataset is 64.
10% of the data has been reserved for validation, while the remaining 90% has been used for training. seed=0 ensures that the dataset splitting remains consistent across different runs. Each audio sample will be resized to have a length of 64,000 samples.

Therefore, there will be 365 batches. Each batch has shape ((64,64000),(64)) where the 1st element is a tensor of 64 audio tensors of length 64,000 and the 2nd ones are its labels.

### Visualisation and Inspection
The sequential audio data is transformed to image by Short Term Fourier Transform using tensorflow's stft().

#### What is a spectrogram?
A spectrogram is a visual representation of the spectrum of frequencies of a signal over time. It provides a way to analyze and visualize the frequency content of a time-varying signal. Spectrograms are widely used in various fields, including audio signal processing, speech recognition, music analysis, and acoustic research.

#### What is waveform?
A waveform represents the amplitude of a signal as a function of time. It is a one-dimensional representation, where time is plotted on the horizontal axis, and the signal's amplitude is plotted on the vertical axis. They are commonly used for visualizing and analyzing the temporal characteristics of signals.

The spectrograms of A and Non A calls:

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/5b0dc027-848f-404c-8abe-a60b28aea4f0)

The waveform of a Non A call:

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/c81aec21-18e6-47a2-a924-e53fae2b1192)

Improvised colored spectrogram for better visualisation:

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/2c5edcdd-570e-40d5-b556-2cb7fc859564)


### Model Construction and training
 A CNN model for audio classification is implemented. It preprocesses the spectrogram inputs by normalizing them, applies several convolutional layers for feature extraction, and ends with fully connected layers for classification.
The Adam optimizer is chosen, which is a popular optimization algorithm for deep learning. The SparseCategoricalCrossentropy loss function is used, indicating that the model's output is not one-hot encoded and comes from logits. The evaluation metric is set to 'accuracy', which calculates the accuracy of the model during training and evaluation. The model has been set to be trained for 15 epochs. Due to early stopping, the training has been stopped at the 8th epoch to prevent over-fitting. At this stage, the val_accuracy has reached 0.9977.

### Inspecting Model Accuracy and loss
The model has trained pretty well.

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/82880c74-8672-4ede-8601-8a64fb7c253c)

The confusion matrix is drawn to see the performance of the model and it does seem of some negative error prediction rate.

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/e8515a06-3634-441b-9e18-c21dde632c04)

False negatives and false positives were inspected manually. The F1 score was calculated which came out to be 0.9969.






