# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from torch import nn

# Neural network model definition
class EmotionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmotionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load preprocessed data and models
df = pd.read_csv("SpotifyDataset/spotify_songs.csv")
scaler = joblib.load('minmax_scaler.pkl')  # Load saved scaler
kmeans = joblib.load('kmeans_model.pkl')  # Load saved KMeans model
df_path = "preprocessed_data.csv"

# Preprocessed data
df = pd.read_csv(df_path)

# Features for clustering
feature_columns = ['valence', 'energy', 'danceability', 'loudness', 'tempo']
df_features = df[feature_columns]

# Apply KMeans clustering
df['emotion_cluster'] = kmeans.predict(scaler.transform(df_features))  # Use pre-trained scaler

# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaler.transform(df_features))  # Use pre-trained scaler
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]

# Visualizing the clusters
sns.scatterplot(x=df['PC1'], y=df['PC2'], hue=df['emotion_cluster'], palette='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters in PCA-Reduced Space')
plt.show()

# Calculate Silhouette Score
#score = silhouette_score(scaler.transform(df_features), df['emotion_cluster'])
#print(f"Silhouette Score: {score:.2f}")

# Load pre-trained  forest models
clf = joblib.load('song_emotion_classifier_forest_model.pkl')

# Initialize the model architecture, which was hyper tuned with gridsearch
input_size = 5
hidden_size = 32
output_size = 5
model = EmotionNet(input_size, hidden_size, output_size)

# Load model weights (with weights_only=True to prevent the FutureWarning)
model.load_state_dict(torch.load('emotion_model.pth', weights_only=True))  # Load weights into the model
model.eval()  # Set model to evaluation mode
# Function for emotion prediction
def predict_emotion(user_mood):
    emotion_map = {
        "Energetic": [0.550727, 0.830376, 0.581754, 0.861508, 0.541825],
        "Joyful": [0.810146, 0.768622, 0.746998, 0.844189, 0.499412],
        "Calm": [0.564478, 0.567334, 0.764183, 0.807609, 0.469615],
        "Sad": [0.269011, 0.435314, 0.594407, 0.767869, 0.485511],
        "Angry": [0.250470, 0.806265, 0.601295, 0.860896, 0.525102]
    }

    # Get emotion features
    emotion_label = emotion_map.get(user_mood, None)
    if emotion_label is None:
        print("Invalid emotion selected!")
        return None

    # Debug print
    print(f"Emotion label (raw): {emotion_label}")

    # Reshape input to 2D array (1 sample, multiple features)
    emotion_input_tensor = torch.tensor([emotion_label], dtype=torch.float32).reshape(1, -1)

    # Debug print
    print(f"Shape of input before scaling: {emotion_input_tensor.numpy().shape}")

    # Scale using the saved scaler
    scaled_input = scaler.transform(emotion_input_tensor.numpy())  # Ensure it's 2D for the scaler

    # Debug print
    print(f"Shape of input after scaling: {scaled_input.shape}")

    with torch.no_grad():
        # Pass the reshaped and scaled input through the model
        outputs = model(torch.tensor(scaled_input, dtype=torch.float32))
        emotion_pred = outputs.argmax(dim=1).numpy()
    
    # Fit the encoder with known classes
    encoder = LabelEncoder()
    encoder.fit(["Energetic", "Joyful", "Calm", "Sad", "Angry"])  # Fit with all possible classes
    return encoder.inverse_transform(emotion_pred)[0]

# Assigning emotion to songs
def assign_emotion_to_songs(df, model, encoder):
    feature_columns = ['valence', 'energy', 'danceability', 'loudness', 'tempo']
    
    # Extract features from the dataframe
    features = np.array(df[feature_columns].values, dtype=np.float32)

    # Debug print
    print(f"Shape of features before scaling: {features.shape}")

    # Normalize the features using the saved scaler
    normalized_features = scaler.transform(features)  # Normalize with saved scaler

    # Debug print
    print(f"Shape of features after scaling: {normalized_features.shape}")

    features_tensor = torch.tensor(normalized_features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(features_tensor)  # Perform a batch forward pass
        pred = outputs.argmax(dim=1).numpy()  # Get all predictions at once
    
    # Map predictions to emotions
    predicted_emotions = encoder.inverse_transform(pred)

    # Assign predicted emotions to the dataframe
    df['emotion'] = predicted_emotions  # Assign all predictions at once
    return df

# Function to assign emotions to songs and recommend songs based on user input
# def process_and_recommend_songs(df, model, encoder, user_input):
#     # Assign emotions to songs
#     df = assign_emotion_to_songs(df, model, encoder)

#     # Predict the emotion based on user input
#     predicted_emotion = predict_emotion(user_input)
    
#     if predicted_emotion:
#         # Recommend songs based on the predicted emotion
#         recommended_songs = df[df['emotion'] == predicted_emotion].sample(n=3)
#         return recommended_songs[['track_name', 'track_artist']]
#     else:
#         return None
def recommendation_song(user_input,amount_of_songs):

    #  Test emotion prediction
    predicted_emotion = predict_emotion(user_input)
    print(f"Predicted Emotion: {predicted_emotion}")
    if not predicted_emotion:
        print("Failed to predict emotion!")
        return

    # Ensure the 'assign_emotion_to_songs' works and adds 'emotion' column to df
    encoder = LabelEncoder()
    encoder.fit(["Energetic", "Joyful", "Calm", "Sad", "Angry"])  # Fit encoder with known classes
    try:
        df_with_emotions = assign_emotion_to_songs(df, model, encoder)
        print("Emotion column successfully assigned to DataFrame!")
    except Exception as e:
        print(f"Error in assigning emotions to songs: {e}")
        return

    # Check if the 'emotion' column exists in the DataFrame
    if 'emotion' not in df_with_emotions.columns:
        print("The 'emotion' column was not added to the DataFrame!")
        return

    #Test song recommendation based on predicted emotion
    try:
        recommended_songs = df_with_emotions[df_with_emotions['emotion'] == predicted_emotion].sample(n=amount_of_songs)
        print(f"Recommended Songs for '{predicted_emotion}':")
        print(recommended_songs[['track_name', 'track_artist','track_album_id']])
        return recommended_songs[['track_name', 'track_artist','track_album_id']]
    except Exception as e:
        print(f"Error in recommending songs: {e}")
        return
#testing
#recommendation_song('Joyful',7)