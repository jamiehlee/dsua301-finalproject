# Unraveling Sentiments and Predictions: Deep Learning Models for Product Review Text Classification

Group members: Brad Zhang, Jamie Lee, Sabrina Xie

All data and checkpoint models used in this project can be accessed here: https://drive.google.com/drive/folders/1fRSEyqf_5_fqLfgamQ_XZlUUNa1JdQb7?usp=sharing

## Project Description
As e-commerce continues to grow in popularity, both the amount and importance of online reviews have increased significantly. Analyzing review sentiment is critical for consumers in deciding whether or not to purchase. Therefore, this project aims to build a tool that can accurately classify online text reviews by sentiment. 

The base architectures explored were GRU, LSTM, BiLSTM and USE, and after an initial comparison, we developed a sequential BiLSTM-GRU, inspired by the paper by Ru Ni and Cao Huan. On top of simply testing these models, we also incorporated transfer learning techniques: fine-tuning and feature extractor to improve the performance of our model. All models followed the same process: they were first trained on the Amazon training dataset and saved to checkpoints based on validation accuracy. We also tested the initial model from the Amazon dataset on the Yelp dataset to check generalizability prior to fine-tuning. Finally, we applied the fine-tuning and feature extractor transfer learning approaches to better fit the models to the new dataset. The overall goal of incorporating all these models was to figure out which model had the best performance and challenge ourselves to try our own architecture.

BiLSTM + GRU Model Architecture:

![BiLSTM + GRU Model Architecture](https://drive.google.com/uc?id=1CJyjrpXaZNK7UOmiG6KwrTqkCaBdgzkF)

## Data
This repository contains the code and models for a sentiment analysis project using various neural network architectures to classify review sentiments from various online platforms. The datasets were from Amazon Reviews collected over a span of 18 years up until March 2013 and Yelp Reviews from 2015 used for training and validation.

We sampled 30,000 reviews from the Amazon dataset and 19,000 reviews from the Yelp dataset and saved the sampled training, test, and validation sets in the Google Drive folder linked above. 

For each model, the data was processed using the spaCy English Tokenizer and the bag-of-words vectorizer. This process can be seen in each of the Jupyter notebooks in this repository.

## Project Structure
This repository contains various google colab notebooks. Each notebook contains code for both datasets and transfer learning.
- lstm_bilstm.ipynb : code for both the LSTM and BiLSTM models
- use.ipynb : code for the USE model
- gru.ipynb : code for the GRU model
- bilstm_gru_sequential_model.ipynb : our custom architecture

To execute the code, download the preprocessed data in the linked Google Drive folder under the "preprocessed" subfolder and run the code in the notebook. The order of the models does not matter as long as the data has already been preprocessed. The raw datasets can also be downloaded from the same folder, under their respective subfolders.
