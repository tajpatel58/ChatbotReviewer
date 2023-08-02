# Sentiment Analysis Project using Prefect Pipelines

<img src="./Graphics/sentiments.png" alt="drawing" width="200" height="200"/>

## Table of Contents

- [Introduction](#introduction)
- [Workflow Overview](#workflow-overview)
- [ML Pipeline](#ml-pipeline)
- [Deployment](#deployment)

## Introduction

This repository contains a sentiment analysis project that leverages Prefect, an open-source workflow management system, and ML pipelines to perform sentiment analysis on textual data. The goal of this project is to automatically classify the sentiment of reviews received on the chatbot deployed on my portfolio website; that way when feedback is consistently negative or neutral - I'm made aware so I can ammend the model and redeploy. 

 <br/>
 
 <br/>

## Workflow Overview

The sentiment analysis pipeline consists of the following steps:

1. **Data Ingestion**: Load the input text data from a JSON format; data has been generated by OpenAI's ChatGPT. 

2. **Text Preprocessing**: Clean and preprocess the text data by removing stopwords, special characters, and performing tokenization.

3. **Feature Extraction**: Convert the preprocessed text into numerical features suitable for machine learning models. Common techniques include TF-IDF, word embeddings, or other NLP representations.

4. **Sentiment Classification**: Train and apply machine learning models to classify the sentiment of the text into positive, negative, or neutral categories.

5. **Results Visualization**: Visualize the sentiment analysis results to gain insights and interpret the model's performance.

 <br/>
 
 <br/>

## ML Pipeline

The ML pipeline involves the following components:

- Model Training: Train multiple machine learning algorithms (e.g., SVM, Random Forest, LSTM), while logging parameters using MlFlow. 

- Model Evaluation: Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.

- Model Selection: Choose the best-performing model based on evaluation results and move model into "production" in MlFlow model registry. 

 <br/>
 
 <br/>

## Deployment 

- Initial plan is to deploy model using Prefect - though currently unsure on how this will work. 
