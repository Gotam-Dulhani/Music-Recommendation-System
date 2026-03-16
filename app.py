import streamlit as st
import pandas as pd
import os
from recommendation_engine import MusicRecommender

# Set Page Config
st.set_page_config(
    page_title="VibeStream - AI Music Recommendations",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Dark Mode Look
st.markdown("""
<style>
    /* Dark Mode Theme Adjustments */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .main-title {
        background: linear-gradient(90deg, #bb86fc, #f50057);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        margin-bottom: 0rem !important;
        padding-bottom: 1rem !important;
    }
    
    .sub-title {
        color: #9e9e9e !important;
        font-size: 1.2rem !important;
        margin-top: -1rem !important;
        margin-bottom: 2rem !important;
    }
    
    /* Music Card Style */
    div.music-card {
        background-color: #1a1e26;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
        border-left: 4px solid #bb86fc;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    div.music-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(187, 134, 252, 0.2);
        border-left: 4px solid #f50057;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.2rem;
    }
    
    .card-artist {
        font-size: 0.9rem;
        color: #b3b3b3;
        margin-bottom: 0.8rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .badge-genre {
        background-color: #3f51b5;
        color: white;
    }
    
    .badge-mood {
        background-color: #009688;
        color: white;
    }
    
    .badge-score {
        background-color: #e91e63;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    songs_path = os.path.join(data_dir, 'songs.csv')
    interactions_path = os.path.join(data_dir, 'interactions.csv')
    return MusicRecommender(songs_path, interactions_path)

def render_song_card(row, score_col=None, score_name="Match"):
    """Helper to render a song as a premium card"""
    html = f"""
    <div class="music-card">
        <div class="card-title">🎵 {row['title']}</div>
        <div class="card-artist">by {row['artist']}</div>
        <div>
            <span class="badge badge-genre">{row['genre']}</span>
            <span class="badge badge-mood">{row['mood']}</span>
    """
    
    if score_col and score_col in row:
        score_val = row[score_col]
        # Format score based on magnitude
        if score_val < 1.0: # Similarity distance
            disp_score = f"{score_val*100:.0f}%"
        else:
            disp_score = f"{score_val:.1f}"
        html += f'<span class="badge badge-score">{score_name}: {disp_score}</span>'
        
    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)

# Main Application
def main():
    st.markdown('<h1 class="main-title">VibeStream</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-Powered Music Discovery</p>', unsafe_allow_html=True)
    
    try:
        engine = load_recommender()
    except Exception as e:
        st.error(f"Error loading datasets. Have you run `data_generator.py`? Error: {e}")
        return
        
    # Sidebar
    st.sidebar.title("🎧 User Profile")
    user_list = engine.interactions_df['user_id'].unique().tolist()
    user_list.sort()
    
    selected_user = st.sidebar.selectbox("Select User ID", user_list)
    
    with st.sidebar.expander("About VibeStream Engine"):
        st.write("""
        This engine uses machine learning to suggest songs:
        - **Content-Based:** Analyzes audio features like Tempo, Energy, and Mood.
        - **Collaborative:** Finds patterns across similar listeners.
        - **Hybrid ML:** Blends both for the ultimate playlist.
        """)
        
    # Main Content Area
    # Get user history
    history = engine.get_user_history(selected_user)
    
    if history.empty:
        st.warning("New user detected. Showing popular tracks.")
        top_songs = engine.get_popular_songs(8)
        cols = st.columns(4)
        for idx, (_, row) in enumerate(top_songs.iterrows()):
            with cols[idx % 4]:
                render_song_card(row)
        return

    # Tabs for organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "✨ Vibe Mix (Hybrid AI)", 
        "🎧 Sounds Like Your Favorites", 
        "👥 Because Listeners Like You",
        "🎨 Discover Your Vibe",
        "🔍 Search & Explore"
    ])
    
    with tab1:
        st.header(f"The Ultimate Vibe Mix for {selected_user}")
        st.write("A curated selection specifically for you, combining audio analysis and listener trends.")
        
        hybrid_recs = engine.get_hybrid_recommendations(selected_user, top_n=8)
        if hybrid_recs.empty:
            st.info("Not enough data to generate hybrid mix yet.")
        else:
            cols = st.columns(4)
            for idx, (_, row) in enumerate(hybrid_recs.iterrows()):
                with cols[idx % 4]:
                    render_song_card(row, score_col='hybrid_score', score_name="Vibe Score")
                    
        st.markdown("---")
        st.subheader("Your Recent Top Tracks")
        # Display top 4 from history
        hcols = st.columns(4)
        for idx, (_, row) in enumerate(history.head(4).iterrows()):
            with hcols[idx % 4]:
                render_song_card(row)
    
    with tab2:
        st.header("Discovered via Audio DNA")
        st.write("Songs sharing the exact tempo, energy, and mood profile as your top track.")
        
        top_song = history.iloc[0]
        st.markdown(f"**Based on your love for: {top_song['title']} by {top_song['artist']}**")
        
        content_recs = engine.get_content_recommendations(top_song['song_id'], top_n=8)
        if content_recs.empty:
            st.info("Could not find similar tracks.")
        else:
            cols = st.columns(4)
            for idx, (_, row) in enumerate(content_recs.iterrows()):
                with cols[idx % 4]:
                    render_song_card(row, score_col='similarity_score', score_name="Match")
                    
    with tab3:
        st.header("Trending in Your Circle")
        st.write("Songs highly rated by users with an identical ear for music.")
        
        collab_recs = engine.get_collaborative_recommendations(selected_user, top_n=8)
        if collab_recs.empty:
            st.info("Explore more music to find your listener community.")
        else:
            cols = st.columns(4)
            for idx, (_, row) in enumerate(collab_recs.iterrows()):
                with cols[idx % 4]:
                    render_song_card(row, score_col='collab_score', score_name="Community Rating")

    with tab4:
        st.header("Customize Your Current Vibe")
        st.write("Tell us exactly what you're looking for right now.")
        
        c1, c2 = st.columns(2)
        with c1:
            genres = ["Any"] + sorted(engine.songs_df['genre'].unique().tolist())
            selected_genre = st.selectbox("I'm in the mood for some...", genres)
            
            energy = st.slider("Energy Level", 0.0, 1.0, 0.5, 0.05, help="Low: Chill/Sleepy, High: Party/Gym")
            
        with c2:
            moods = ["Any"] + sorted(engine.songs_df['mood'].unique().tolist())
            selected_mood = st.selectbox("My current mood is...", moods)
            
            dance = st.slider("Danceability", 0.0, 1.0, 0.5, 0.05, help="Low: To listen, High: To dance")
            
        if st.button("Generate My Vibe", type="primary"):
            pref_recs = engine.get_recommendations_by_preferences(
                genre=selected_genre, 
                mood=selected_mood, 
                energy=energy, 
                danceability=dance
            )
            
            if pref_recs.empty:
                st.warning("No songs found matching those specific filters. Try broadening your search!")
            else:
                st.subheader(f"Recommendations for: {selected_genre} • {selected_mood}")
                cols = st.columns(4)
                for idx, (_, row) in enumerate(pref_recs.iterrows()):
                    with cols[idx % 4]:
                        render_song_card(row, score_col='preference_score', score_name="Vibe Match")

    with tab5:
        st.header("Find Something Specific")
        st.write("Search for a song you already love to find similar tracks.")
        
        search_query = st.text_input("Search by song title or artist", placeholder="e.g. Shape of You")
        
        if search_query:
            results = engine.songs_df[
                engine.songs_df['title'].str.contains(search_query, case=False) | 
                engine.songs_df['artist'].str.contains(search_query, case=False)
            ].head(5)
            
            if results.empty:
                st.info("No matching songs found.")
            else:
                st.write(f"Found {len(results)} matches:")
                for _, row in results.iterrows():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"**{row['title']}** - {row['artist']} ({row['genre']})")
                    with col_b:
                        if st.button(f"Recs Like This", key=row['song_id']):
                            st.session_state.search_recs = engine.get_content_recommendations(row['song_id'], top_n=8)
                            st.session_state.search_target = row['title']
                
                if 'search_recs' in st.session_state:
                    st.markdown("---")
                    st.subheader(f"Because you liked: {st.session_state.search_target}")
                    cols = st.columns(4)
                    for idx, (_, row) in enumerate(st.session_state.search_recs.iterrows()):
                        with cols[idx % 4]:
                            render_song_card(row, score_col='similarity_score', score_name="DNA Match")

if __name__ == "__main__":
    main()
