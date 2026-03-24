# Emotion Detection and Action Suggestion System
Overview

This project is a machine learning-based emotion detection system that predicts a user’s emotional state and emotional intensity from reflective journal text and contextual features. Based on the predictions, the system also provides a suggested action and the best time to perform it.

The project combines:

Text vectorization using TF-IDF
Emotion classification using Random Forest Classifier
Intensity prediction using Random Forest Regressor
Rule-based decision engine for action recommendation
Interactive user input system for live predictions

This makes the project a simple end-to-end pipeline for emotion-aware text analysis.

Problem Statement

People often express emotions through text, such as journal entries or personal reflections. This project aims to analyze such text and identify:

the predicted emotional state
the predicted intensity level
an appropriate suggested action
when that action should be taken

The goal is to build a small intelligent system that can convert free-text emotional reflections into meaningful outputs.

Features
Loads training and test datasets
Handles missing values
Combines multiple input columns into one feature text
Converts text into numerical vectors using TF-IDF
Predicts emotion labels using Random Forest Classification
Predicts intensity values using Random Forest Regression
Applies a decision engine to suggest actions
Exports final predictions to a CSV file
Accepts live user input from the terminal for real-time predictions
Technologies Used
Python
Pandas
Scikit-learn
TF-IDF Vectorizer
Random Forest Classifier
Random Forest Regressor
Project Workflow
1. Data Loading

The project reads:

a training dataset
a test dataset

The training dataset is used to train the models, while the test dataset is used to generate predictions.

2. Data Preprocessing

Missing values are filled with empty strings to avoid null-related errors during processing. Then, several useful columns are merged into a single combined text feature:

journal_text
ambience_type
time_of_day
previous_day_mood
face_emotion_hint
reflection_quality

This combined text is used as the main input for the ML models.

3. Feature Extraction

The combined text is transformed into numerical vectors using TF-IDF Vectorization with a maximum of 5000 features. This helps convert textual information into a machine-readable form.

4. Target Encoding

The emotional_state column is encoded into numeric labels using LabelEncoder for classification.
The intensity column is used directly as a numerical target for regression.

5. Model Training

Two separate machine learning models are trained:

RandomForestClassifier for predicting emotion
RandomForestRegressor for predicting intensity

This dual-model approach allows the system to handle both categorical and numerical prediction tasks.

6. Prediction

The system predicts:

predicted_emotion
predicted_intensity

The predicted emotion labels are converted back into readable text using the fitted label encoder.

7. Decision Engine

A rule-based decision engine maps the predicted emotion and intensity to:

suggested_action
when_to_do

For example:

If the emotion is sadness and intensity is high, the system may suggest talking to a friend immediately.
If the emotion is anger and intensity is low, it may suggest going for a walk.
8. Output Generation

The final results are saved into a CSV file named:

final_output.csv

This file contains the original test data along with predicted emotion, intensity, suggested action, and timing.

9. Interactive User Input

The project also includes a terminal-based prediction function where users can type their feelings manually. The system then predicts the emotion, intensity, and suggested action in real time.

Input Features

The model uses the following features:

Journal text
Ambience type
Time of day
Previous day mood
Face emotion hint
Reflection quality
Output

The system produces:

Predicted emotion
Predicted intensity
Suggested action
Recommended time for the action
Why This Project Is Useful

This project demonstrates important machine learning concepts such as:

text preprocessing
feature combination
NLP-based vectorization
classification
regression
simple rule-based decision making
user interaction through terminal input

It is a useful beginner-to-intermediate project for showcasing practical ML workflow in a portfolio.

Limitations
The decision engine is manually defined and not learned from data.
Live user input uses default placeholder values for non-text features such as ambience and time of day.
No model evaluation metrics are printed in the final workflow, even though evaluation libraries are imported.
The system should not be treated as a real mental health diagnostic tool.
Performance may depend heavily on dataset quality and label consistency.
Future Improvements
Add proper model evaluation such as accuracy, RMSE, confusion matrix, and classification report
Use hyperparameter tuning for better performance
Build a web interface using Flask or Streamlit
Replace rule-based suggestions with a smarter recommendation system
Add better preprocessing such as stopword removal, lemmatization, and text cleaning
Use advanced NLP models like BERT or sentence transformers
Allow users to input contextual values instead of default placeholders
Conclusion

This project is a simple yet effective demonstration of how machine learning can be used to analyze emotional text and generate meaningful recommendations. It covers the complete pipeline from preprocessing to prediction to action suggestion, making it a solid project for learning and portfolio building.
