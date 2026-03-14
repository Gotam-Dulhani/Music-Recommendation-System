import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_songs(num_songs=500):
    genres = ['Pop', 'Rock', 'Hip-Hop', 'Electronic', 'Classical', 'Jazz', 'R&B', 'Country']
    moods = ['Happy', 'Sad', 'Energetic', 'Chill', 'Romantic', 'Angry']
    
    songs_data = []
    
    for i in range(1, num_songs + 1):
        genre = random.choice(genres)
        mood = random.choice(moods)
        
        # Correlate features somewhat with genre/mood for realism
        if genre in ['Electronic', 'Hip-Hop', 'Pop'] or mood in ['Energetic', 'Happy']:
            energy = np.random.uniform(0.6, 1.0)
            danceability = np.random.uniform(0.5, 1.0)
            tempo = np.random.uniform(100, 150)
        elif genre in ['Classical', 'Jazz'] or mood in ['Sad', 'Chill']:
            energy = np.random.uniform(0.1, 0.5)
            danceability = np.random.uniform(0.1, 0.6)
            tempo = np.random.uniform(60, 100)
        else:
            energy = np.random.uniform(0.3, 0.8)
            danceability = np.random.uniform(0.3, 0.8)
            tempo = np.random.uniform(80, 120)
            
        songs_data.append({
            'song_id': f'S_{i:04d}',
            'title': f'{genre} Track {i}',
            'artist': f'Artist {random.randint(1, 100)}',
            'genre': genre,
            'mood': mood,
            'energy': round(energy, 2),
            'danceability': round(danceability, 2),
            'tempo': round(tempo, 0)
        })
        
    return pd.DataFrame(songs_data)

def generate_interactions(songs_df, num_users=50, max_interactions=50):
    users = [f'U_{i:03d}' for i in range(1, num_users + 1)]
    song_ids = songs_df['song_id'].tolist()
    
    interactions_data = []
    
    for user in users:
        # Each user has a primary genre preference to create patterns
        preferred_genre = random.choice(songs_df['genre'].unique())
        preferred_songs = songs_df[songs_df['genre'] == preferred_genre]['song_id'].tolist()
        other_songs = [s for s in song_ids if s not in preferred_songs]
        
        num_interactions = random.randint(10, max_interactions)
        
        # User interacts mostly with preferred genre
        interacted_songs = random.sample(preferred_songs, min(len(preferred_songs), int(num_interactions * 0.7)))
        interacted_songs += random.sample(other_songs, int(num_interactions * 0.3))
        
        for song_id in interacted_songs:
            play_count = random.randint(1, 50)
            # Higher play count usually means liked
            liked = 1 if play_count > 10 or random.random() > 0.7 else 0
            
            interactions_data.append({
                'user_id': user,
                'song_id': song_id,
                'play_count': play_count,
                'liked': liked
            })
            
    return pd.DataFrame(interactions_data)

if __name__ == "__main__":
    print("Generating synthetic music dataset...")
    
    songs_df = generate_songs(num_songs=500)
    interactions_df = generate_interactions(songs_df, num_users=50)
    
    # Save to CSV
    data_dir = os.path.dirname(os.path.abspath(__file__))
    songs_df.to_csv(os.path.join(data_dir, 'songs.csv'), index=False)
    interactions_df.to_csv(os.path.join(data_dir, 'interactions.csv'), index=False)
    
    print(f"Generated {len(songs_df)} songs and saved to songs.csv")
    print(f"Generated {len(interactions_df)} interactions and saved to interactions.csv")
    print("Dataset generation complete!")
