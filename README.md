# AML-Project
# Song Popularity Prediction
Applied Machine Leaning 

CCDS-322 Course


## Overview

At its most basic, a song is a short piece of music, usually with words. It combines melody and vocals. The word 'song' has been around for a very long time, and it connects back to Old English and Old Norse languages. As such a history suggests, songs are used for many purposes: to tell stories, express emotions, or convey a belief in faith.

Humans have greatly associated themselves with Songs & Music. It can improve mood, decrease pain and anxiety, and facilitate opportunities for emotional expression. Research suggests that music can benefit our physical and mental health in numerous ways.

Lately, multiple studies have been carried out to understand songs & it's popularity based on certain factors. Our goal is to create a machine learning model that can predict a song’s popularity.


## Objectives

The object of this project is to design a model that predicts song popularity, based on certain features included in the dataset.

#### Question/ hypothesis:-


The  primary  purpose  of  our  study  is to  create  a classifier that can predict whether a song popularity related on certain features in the dataset. So, does the feature of danceability significantly impact the popularity?


#### Dataset:-

The dataset for our project was sourced from Kaggle, a platform known for offering a wide range of real-world datasets. 
The project is simple yet challenging, to predict the song popularity based on energy, acoustics, instumentalness, liveness, dancibility, etc. The dataset is large & it's complexity arises due to the fact that it has strong multicollinearity.

The data entries include the following features:

| Col # | Column Name       | Column's Description | Type       | Min  | Max   | Avg   | St. Deviation |
|-------|-------------------|----------------------|------------|------|-------|-------|---------------|
| 1     | song_name         | The title of the song. | Polynomial | -    | -     | -     | -             |
| 2     | song_popularity   | A numerical measure indicating how popular the song is. Higher numbers suggest greater popularity. | Integer    | 0    | 100   | 48.75 | 20.37         |
| 3     | song_duration_ms  | The duration of the song measured in milliseconds. | Integer    | 12000 | 1799346 | 218947.06 | 62337.79      |
| 4     | acousticness      | A value between 0 and 1 that signifies how acoustic the track is. Higher values mean the track is more acoustic. | Real       | 0    | 0.99  | 0.27  | 0.29          |
| 5     | danceability      | A measure assessing how suitable a track is for dancing; higher values denote greater suitability. | Real       | 0    | 0.98  | 0.62  | 0.15          |
| 6     | energy            | A metric that represents the intensity of a song; higher values indicate more energetic songs. | Real       | 0.001 | 0.99  | 0.64  | 0.22          |
| 7     | instrumentalness  | Indicates the likelihood that a track contains no vocal content. Tracks near 1 are likely instrumental. | Real       | 0    | 0.99  | 0.09  | 0.24          |
| 8     | key               | The key the track is in, using Pitch Class notation where 0 = C, 1 = C♯/D♭, up to 11 = B. | Integer    | -    | 11    | 5.38  | 3.59          |
| 9     | liveness          | A measure predicting whether a track was performed live. Higher values indicate live recordings. | Real       | 0.011 | 0.98  | 0.18  | 0.14          |
| 10    | loudness          | The overall loudness of a track in decibels (dB). Higher values indicate a louder track. | Real       | -38.76 | 1.58 | -7.67 | 4.01          |
| 11    | audio_mode        | Indicates the modality of the track, with 1 for major and 0 for minor. | Integer    | 0    | 1     | 0.63  | 0.48          |
| 12    | speechiness       | Detects spoken words in a track; higher values signify more spoken words. | Real       | 0    | 0.94  | 0.099 | 0.10          |
| 13    | tempo             | The estimated tempo of a track in beats per minute (BPM). | Real       | 0    | 242.31 | 121.10 | 29.03         |
| 14    | time_signature    | An estimated overall time signature of a track, indicating the number of beats in a measure. | Integer    | 0    | 5     | 3.95  | 0.31          |
| 15    | audio_valence     | Describes the musical positiveness conveyed by a track. High valence sounds positive, low valence sounds negative. | Real       | 0    | 0.98  | 0.52  | 0.24          |





## Machine Learning Approachs:-

#### Linear Regression 
Linear regression is a supervised machine learning algorithm used to predict, or visualize, a relationship between two different features/variables. In linear regression tasks, there are two kinds of variables being examined: the dependent variable and the independent variable.
It describe the relationship between variables by fitting a line to the observed data. Regression allows us to estimate how a dependent variable changes as the independent variable(s) change.
Hence, linear regression is an example of a regression model and logistic regression is an example of a classification model.

#### Logistic Regression 
Logistic regression is the base line supervised machine learning algorithm for classification. Logistic regression is used to solve classification problems, and the most common use case is binary logistic regression, where the outcome is binary (yes or no). In the real world, you can see logistic regression applied across multiple areas and fields.


## Finding

The primary objective of our study was to develop a classifier capable of predicting song popularity based on specific dataset features. We chose logistic regression techniques for their potential for high accuracy. Our analysis showed that logistic regression was particularly effective for this task, outperforming linear regression. Notably, logistic regression achieved a commendable prediction accuracy of 88%, underscoring its suitability for classification tasks like predicting binary outcomes such as song popularity.
