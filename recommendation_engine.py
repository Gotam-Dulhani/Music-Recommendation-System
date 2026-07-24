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
        
        self._prepare_content_matrix()
        self._prepare_collaborative_matrix()
        
    def _prepare_content_matrix(self):
        """Prepare the features for content-based filtering with proper weighting."""
        df = self.songs_df.copy()
        
        # Scale numerical features to [0, 1]
        minmax = MinMaxScaler()
        num_scaled = minmax.fit_transform(df[['energy', 'danceability', 'tempo']])
        num_df = pd.DataFrame(num_scaled, columns=['energy_s', 'dance_s', 'tempo_s'])
        
        # One-hot encode categorical features
        genres_encoded = pd.get_dummies(df['genre'], prefix='genre').astype(float)
        moods_encoded = pd.get_dummies(df['mood'], prefix='mood').astype(float)
        
        # Apply feature group weights so genre/mood don't overwhelm audio features.
        # Audio features get weight 3.0 per feature (total ~9.0).
        # Each genre/mood column gets weight 1.0 / sqrt(num_cols_in_group) to keep
        # the total contribution of each categorical group proportional.
        audio_weight = 3.0
        genre_weight = 1.0 / np.sqrt(genres_encoded.shape[1])
        mood_weight = 1.0 / np.sqrt(moods_encoded.shape[1])
        
        num_df *= audio_weight
        genres_encoded *= genre_weight
        moods_encoded *= mood_weight
        
        self.content_features = pd.concat([
            df[['song_id', 'title', 'artist', 'genre', 'mood']],
            genres_encoded,
            moods_encoded,
            num_df
        ], axis=1)
        
        feature_cols = list(genres_encoded.columns) + list(moods_encoded.columns) + list(num_df.columns)
        self._content_feature_cols = feature_cols
        feature_matrix = self.content_features[feature_cols].values
        self.content_sim_matrix = cosine_similarity(feature_matrix)
        
    def _prepare_collaborative_matrix(self):
        """Prepare the user-item interaction matrix for collaborative filtering."""
        self.interactions_df['score'] = (
            self.interactions_df['play_count'] + (self.interactions_df['liked'] * 20)
        )
        
        self.user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', 
            columns='song_id', 
            values='score', 
            fill_value=0
        )
        
        self.user_sim_matrix = cosine_similarity(self.user_item_matrix)
        self.user_sim_df = pd.DataFrame(
            self.user_sim_matrix, 
            index=self.user_item_matrix.index, 
            columns=self.user_item_matrix.index
        )
        
    def get_content_recommendations(self, song_id, top_n=5, exclude_ids=None):
        """Recommend songs similar to a given song based on audio features."""
        if song_id not in self.songs_df['song_id'].values:
            return pd.DataFrame()
            
        song_idx = self.songs_df[self.songs_df['song_id'] == song_id].index[0]
        sim_scores = list(enumerate(self.content_sim_matrix[song_idx]))
        
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        recs = []
        for idx, score in sim_scores:
            if idx == song_idx:
                continue
            candidate_id = self.songs_df.iloc[idx]['song_id']
            if exclude_ids and candidate_id in exclude_ids:
                continue
            recs.append((idx, score))
            if len(recs) >= top_n:
                break
                
        if not recs:
            return pd.DataFrame()
            
        song_indices = [r[0] for r in recs]
        scores = [round(r[1], 3) for r in recs]
        
        result = self.songs_df.iloc[song_indices].copy()
        result['similarity_score'] = scores
        return result
        
    def get_collaborative_recommendations(self, user_id, top_n=5):
        """Recommend songs based on similar users' listening history."""
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame()
            
        sim_users = self.user_sim_df[user_id].sort_values(ascending=False).drop(user_id)
        
        if sim_users.empty or sim_users.max() == 0:
            return pd.DataFrame()
            
        # Use top 10 similar users for broader coverage
        top_sim_users = sim_users.head(10)
        
        target_user_songs = set(
            self.interactions_df[self.interactions_df['user_id'] == user_id]['song_id']
        )
        
        # Get target user's total interaction score for normalization
        target_user = self.interactions_df[self.interactions_df['user_id'] == user_id]
        target_total = target_user['score'].sum()
        if target_total == 0:
            target_total = 1
        
        rec_scores = {}
        rec_song_sources = {}
        
        for sim_user in top_sim_users.index:
            sim_user_data = self.interactions_df[self.interactions_df['user_id'] == sim_user]
            sim_score = top_sim_users[sim_user]
            
            # Normalize by this similar user's total activity
            sim_user_total = sim_user_data['score'].sum()
            if sim_user_total == 0:
                continue
                
            for _, row in sim_user_data.iterrows():
                song_id = row['song_id']
                if song_id not in target_user_songs:
                    # Weight by similarity and normalize by activity level
                    weighted_score = (row['score'] / sim_user_total) * sim_score
                    if song_id in rec_scores:
                        rec_scores[song_id] += weighted_score
                        rec_song_sources[song_id] += 1
                    else:
                        rec_scores[song_id] = weighted_score
                        rec_song_sources[song_id] = 1
                        
        if not rec_scores:
            return pd.DataFrame()
            
        # Normalize by how many users recommended each song to avoid bias
        for song_id in rec_scores:
            rec_scores[song_id] /= rec_song_sources[song_id]
            
        sorted_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        rec_song_ids = [s[0] for s in sorted_recs]
        
        # Normalize scores to [0, 1] range
        max_score = sorted_recs[0][1] if sorted_recs else 1
        rec_scores_normalized = [round(s[1] / max_score, 3) for s in sorted_recs]
        
        recs = self.songs_df[self.songs_df['song_id'].isin(rec_song_ids)].copy()
        
        score_map = dict(zip(rec_song_ids, rec_scores_normalized))
        recs['collab_score'] = recs['song_id'].map(score_map)
        
        recs['sort_cat'] = pd.Categorical(recs['song_id'], categories=rec_song_ids, ordered=True)
        recs = recs.sort_values('sort_cat')
        recs.drop('sort_cat', axis=1, inplace=True)
        
        return recs
        
    def get_user_history(self, user_id):
        """Return the actual listening history for a user, ranked by engagement."""
        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
        if user_interactions.empty:
            return pd.DataFrame()
            
        history = pd.merge(user_interactions, self.songs_df, on='song_id')
        # Rank by a combined engagement score: liked gets priority, then play_count
        history['engagement'] = history['liked'] * 1000 + history['play_count']
        history = history.sort_values(by='engagement', ascending=False)
        history.drop('engagement', axis=1, inplace=True)
        return history
        
    def get_hybrid_recommendations(self, user_id, top_n=5):
        """Combine Collaborative and Content-based approaches with proper scoring."""
        history = self.get_user_history(user_id)
        if history.empty:
            return self.get_popular_songs(top_n)
        
        user_history_ids = set(history['song_id'].tolist())
        top_history_songs = history.head(5)['song_id'].tolist()
        
        # Collect content-based recs from top5 favorite songs, excluding history
        content_recs = pd.DataFrame()
        for song_id in top_history_songs:
            recs = self.get_content_recommendations(song_id, top_n=5, exclude_ids=user_history_ids)
            if not recs.empty:
                content_recs = pd.concat([content_recs, recs])
                
        if not content_recs.empty:
            # Average similarity scores for songs recommended by multiple top songs
            content_recs = content_recs.groupby('song_id', as_index=False).agg({
                'similarity_score': 'mean',
                **{col: 'first' for col in content_recs.columns if col not in ['song_id', 'similarity_score']}
            })
        
        # Get collaborative recs (already excludes user history)
        collab_recs = self.get_collaborative_recommendations(user_id, top_n=10)
        
        # Normalize both to [0, 1] before combining
        all_recs = pd.DataFrame()
        
        if not content_recs.empty:
            content_recs = content_recs.copy()
            content_recs['content_score'] = content_recs['similarity_score']
            all_recs = content_recs[['song_id', 'title', 'artist', 'genre', 'mood', 'energy', 'danceability', 'tempo', 'content_score']].copy()
        
        if not collab_recs.empty:
            collab_copy = collab_recs[['song_id', 'title', 'artist', 'genre', 'mood', 'energy', 'danceability', 'tempo', 'collab_score']].copy()
            
            if all_recs.empty:
                all_recs = collab_copy.rename(columns={'collab_score': 'hybrid_score'})
                all_recs['content_score'] = 0.0
            else:
                merged = pd.merge(
                    all_recs, collab_copy[['song_id', 'collab_score']], 
                    on='song_id', how='outer'
                )
                merged['content_score'] = merged['content_score'].fillna(0.0)
                merged['collab_score'] = merged['collab_score'].fillna(0.0)
                all_recs = merged
        
        if all_recs.empty:
            return self.get_popular_songs(top_n)
        
        # Ensure both score columns exist
        if 'content_score' not in all_recs.columns:
            all_recs['content_score'] = 0.0
        if 'collab_score' not in all_recs.columns:
            all_recs['collab_score'] = 0.0
        
        # Hybrid score: weighted combination (content 55%, collaborative 45%)
        # Boost songs that appear in both recommendations
        content_weight = 0.55
        collab_weight = 0.45
        
        all_recs['in_both'] = ((all_recs['content_score'] > 0) & (all_recs['collab_score'] > 0)).astype(float)
        all_recs['hybrid_score'] = (
            all_recs['content_score'] * content_weight + 
            all_recs['collab_score'] * collab_weight +
            all_recs['in_both'] * 0.15  # Boost for songs found by both methods
        )
        
        all_recs = all_recs.sort_values(by='hybrid_score', ascending=False).head(top_n)
        all_recs.drop('in_both', axis=1, inplace=True)
        
        return all_recs
        
    def get_popular_songs(self, top_n=5):
        """Fallback: Return most popular songs overall."""
        song_stats = self.interactions_df.groupby('song_id').agg(
            total_score=('score', 'sum')
        ).reset_index()
        
        top_songs = song_stats.sort_values(by='total_score', ascending=False).head(top_n)
        result = pd.merge(top_songs, self.songs_df, on='song_id')
        return result

    def get_recommendations_by_preferences(self, genre=None, mood=None, energy=0.5, danceability=0.5, tempo=None, top_n=8):
        """Recommend songs based on specific user-selected features and categories."""
        df = self.songs_df.copy()
        
        if genre and genre != "Any":
            df = df[df['genre'] == genre]
        if mood and mood != "Any":
            df = df[df['mood'] == mood]
            
        if df.empty:
            return pd.DataFrame()
        
        # Use MinMaxScaler to normalize features for distance calculation
        feature_cols = ['energy', 'danceability', 'tempo']
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        # Build target point
        target = {'energy': energy, 'danceability': danceability}
        if tempo is not None:
            target['tempo'] = (tempo - df['tempo'].min()) / (df['tempo'].max() - df['tempo'].min()) if df['tempo'].max() != df['tempo'].min() else 0.5
        else:
            target['tempo'] = df_scaled['tempo'].mean()
            
        df['dist'] = np.sqrt(
            (df_scaled['energy'] - target['energy'])**2 + 
            (df_scaled['danceability'] - target['danceability'])**2 +
            (df_scaled['tempo'] - target['tempo'])**2
        )
        
        max_dist = np.sqrt(3)
        df['preference_score'] = ((1 - df['dist'] / max_dist) * 100).round(1)
        
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
