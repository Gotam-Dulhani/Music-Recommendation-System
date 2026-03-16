import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class MusicRecommender:
    def __init__(self, songs_path, interactions_path):
        self.songs_df = pd.read_csv(songs_path)
        self.interactions_df = pd.read_csv(interactions_path)
        
        # Prepare datasets
        self._prepare_content_matrix()
        self._prepare_collaborative_matrix()
        
    def _prepare_content_matrix(self):
        """Prepare the features for content-based filtering."""
        df = self.songs_df.copy()
        
        # One-hot encode categorical features
        genres_encoded = pd.get_dummies(df['genre'], prefix='genre')
        moods_encoded = pd.get_dummies(df['mood'], prefix='mood')
        
        # Scale numerical features
        scaler = MinMaxScaler()
        num_features = scaler.fit_transform(df[['energy', 'danceability', 'tempo']])
        num_features_df = pd.DataFrame(num_features, columns=['energy_scaled', 'dance_scaled', 'tempo_scaled'])
        
        # Combine all features
        self.content_features = pd.concat([
            df['song_id'], 
            genres_encoded, 
            moods_encoded, 
            num_features_df
        ], axis=1)
        
        # Create a similarity matrix
        feature_matrix = self.content_features.drop('song_id', axis=1).values
        self.content_sim_matrix = cosine_similarity(feature_matrix)
        
    def _prepare_collaborative_matrix(self):
        """Prepare the user-item interaction matrix for collaborative filtering."""
        # Create an interaction score: play_count + (liked * 20)
        self.interactions_df['score'] = self.interactions_df['play_count'] + (self.interactions_df['liked'] * 20)
        
        # Create pivot table (users as rows, songs as columns)
        self.user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', 
            columns='song_id', 
            values='score', 
            fill_value=0
        )
        
        # Calculate user similarity matrix
        self.user_sim_matrix = cosine_similarity(self.user_item_matrix)
        self.user_sim_df = pd.DataFrame(
            self.user_sim_matrix, 
            index=self.user_item_matrix.index, 
            columns=self.user_item_matrix.index
        )
        
    def get_content_recommendations(self, song_id, top_n=5):
        """Recommend songs similar to a given song based on audio features."""
        if song_id not in self.songs_df['song_id'].values:
            return pd.DataFrame()
            
        song_idx = self.songs_df[self.songs_df['song_id'] == song_id].index[0]
        sim_scores = list(enumerate(self.content_sim_matrix[song_idx]))
        
        # Sort based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar songs (excluding the input song itself)
        sim_scores = sim_scores[1:top_n+1]
        song_indices = [i[0] for i in sim_scores]
        scores = [round(i[1], 3) for i in sim_scores]
        
        recs = self.songs_df.iloc[song_indices].copy()
        recs['similarity_score'] = scores
        return recs
        
    def get_collaborative_recommendations(self, user_id, top_n=5):
        """Recommend songs based on similar users' listening history."""
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame()
            
        # Get top similar users
        sim_users = self.user_sim_df[user_id].sort_values(ascending=False).drop(user_id)
        
        if sim_users.empty or sim_users.max() == 0:
            return pd.DataFrame() # No similar users found
            
        # We will look at top 5 similar users
        top_sim_users = sim_users.head(5).index
        
        # Find songs interacted by similar users that the target user hasn't heard
        target_user_songs = set(self.interactions_df[self.interactions_df['user_id'] == user_id]['song_id'])
        
        rec_scores = {}
        for sim_user in top_sim_users:
            sim_user_songs = self.interactions_df[self.interactions_df['user_id'] == sim_user]
            sim_score = sim_users[sim_user]
            
            for _, row in sim_user_songs.iterrows():
                song_id = row['song_id']
                if song_id not in target_user_songs:
                    # Weigh the score by user similarity
                    weighted_score = row['score'] * sim_score
                    if song_id in rec_scores:
                        rec_scores[song_id] += weighted_score
                    else:
                        rec_scores[song_id] = weighted_score
                        
        if not rec_scores:
            return pd.DataFrame()
            
        # Sort and get top N
        sorted_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        rec_song_ids = [s[0] for s in sorted_recs]
        rec_scores_list = [round(s[1], 2) for s in sorted_recs]
        
        recs = self.songs_df[self.songs_df['song_id'].isin(rec_song_ids)].copy()
        
        # Maintain sorted order
        recs['sort_cat'] = pd.Categorical(recs['song_id'], categories=rec_song_ids, ordered=True)
        recs = recs.sort_values('sort_cat')
        recs['collab_score'] = rec_scores_list
        recs.drop('sort_cat', axis=1, inplace=True)
        
        return recs
        
    def get_user_history(self, user_id):
        """Return the actual listening history for a user."""
        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
        if user_interactions.empty:
            return pd.DataFrame()
            
        history = pd.merge(user_interactions, self.songs_df, on='song_id')
        return history.sort_values(by=['liked', 'play_count'], ascending=False)
        
    def get_hybrid_recommendations(self, user_id, top_n=5):
        """Combine Collaborative and Content-based approaches."""
        # 1. Get user's favorite songs (for content-based)
        history = self.get_user_history(user_id)
        if history.empty:
            # Cold start problem: return most popular overall
            return self.get_popular_songs(top_n)
            
        top_history_songs = history.head(3)['song_id'].tolist()
        
        # 2. Collect content-based recs based on top favorite songs
        content_recs = pd.DataFrame()
        for song_id in top_history_songs:
            recs = self.get_content_recommendations(song_id, top_n=3)
            if not recs.empty:
                content_recs = pd.concat([content_recs, recs])
                
        # 3. Get collaborative recs
        collab_recs = self.get_collaborative_recommendations(user_id, top_n=10)
        
        # 4. Merge and rank
        all_recs = pd.DataFrame()
        
        if not content_recs.empty:
            all_recs = content_recs.drop_duplicates(subset=['song_id'])
            # Add arbitrary weight so it can be sorted
            all_recs['hybrid_score'] = all_recs['similarity_score'] * 50 
            
        if not collab_recs.empty:
            collab_copy = collab_recs.copy()
            # Normalize collaborative score roughly to match scale
            if collab_copy['collab_score'].max() > 0:
                collab_copy['hybrid_score'] = (collab_copy['collab_score'] / collab_copy['collab_score'].max()) * 50
            else:
                collab_copy['hybrid_score'] = 0
                
            # Combine
            if all_recs.empty:
                all_recs = collab_copy
            else:
                # If a song is in both, boost its score
                for _, row in collab_copy.iterrows():
                    if row['song_id'] in all_recs['song_id'].values:
                        idx = all_recs[all_recs['song_id'] == row['song_id']].index
                        all_recs.loc[idx, 'hybrid_score'] += row['hybrid_score']
                    else:
                        all_recs = pd.concat([all_recs, pd.DataFrame([row])], ignore_index=True)
                        
        if all_recs.empty:
            return self.get_popular_songs(top_n)
            
        # Ensure we don't recommend songs already in user's history
        user_history_ids = set(history['song_id'])
        all_recs = all_recs[~all_recs['song_id'].isin(user_history_ids)]
        
        all_recs = all_recs.sort_values(by='hybrid_score', ascending=False).head(top_n)
        
        # Only return song details
        cols_to_return = [c for c in self.songs_df.columns]
        cols_to_return.append('hybrid_score')
        return all_recs[[c for c in cols_to_return if c in all_recs.columns]]
        
    def get_popular_songs(self, top_n=5):
        """Fallback: Return most popular songs overall."""
        song_stats = self.interactions_df.groupby('song_id').agg(
            total_score=('score', 'sum')
        ).reset_index()
        
        top_songs = song_stats.sort_values(by='total_score', ascending=False).head(top_n)
        return pd.merge(top_songs, self.songs_df, on='song_id')

    def get_recommendations_by_preferences(self, genre=None, mood=None, energy=0.5, danceability=0.5, top_n=8):
        """Recommend songs based on specific user-selected features and categories."""
        df = self.songs_df.copy()
        
        # 1. Filter by categorical features if provided
        if genre and genre != "Any":
            df = df[df['genre'] == genre]
        if mood and mood != "Any":
            df = df[df['mood'] == mood]
            
        if df.empty:
            return pd.DataFrame()
            
        # 2. Calculate distance from desired audio features (Energy, Danceability)
        # We'll use a simple Euclidean distance-based score
        # Note: In a real app, we'd use the scaled features from _prepare_content_matrix
        # but for simplicity and responsiveness to direct input, we use raw values here
        
        df['dist'] = np.sqrt(
            (df['energy'] - energy)**2 + 
            (df['danceability'] - danceability)**2
        )
        
        # 3. Rank by proximity (lower distance is better)
        # Convert distance to a 0-100 score for display
        df['preference_score'] = (1 - df['dist'] / np.sqrt(2)) * 100
        
        return df.sort_values(by='dist').head(top_n)

if __name__ == "__main__":
    import os
    data_dir = os.path.dirname(os.path.abspath(__file__))
    recommender = MusicRecommender(
        os.path.join(data_dir, 'songs.csv'), 
        os.path.join(data_dir, 'interactions.csv')
    )
    
    print("Testing recommendations for U_001:")
    print("\n--- History (Top 3) ---")
    print(recommender.get_user_history('U_001')[['title', 'artist', 'genre', 'play_count', 'liked']].head(3))
    
    print("\n--- Content-Based (Sim to Top song) ---")
    top_song = recommender.get_user_history('U_001').iloc[0]['song_id']
    print(recommender.get_content_recommendations(top_song)[['title', 'artist', 'genre', 'similarity_score']])
    
    print("\n--- Collaborative Recommendations ---")
    print(recommender.get_collaborative_recommendations('U_001')[['title', 'artist', 'genre', 'collab_score']])
    
    print("\n--- Hybrid Recommendations ---")
    print(recommender.get_hybrid_recommendations('U_001')[['title', 'artist', 'genre', 'hybrid_score']])
