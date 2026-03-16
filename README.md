# AI Music Recommendation System

This project is an AI-powered music discovery platform that uses Machine Learning to suggest tracks based on audio features and listener patterns.

## ✨ New: Interactive discovery

We've updated the system to give you direct control over your music discovery:

- **🎨 Discover Your Vibe**: Select specific genres and moods, and use precision sliders to tune the Energy and Danceability of your recommendations.
- **🔍 Search & Explore**: Search for any song in the database and immediately find tracks that share its unique "Audio DNA".

## 🏗 Project Architecture

You might notice there isn't a traditional `/frontend` and `/backend` directory. Here's why:

- **Streamlit Framework**: We used **Streamlit**, which is an industry-standard for Machine Learning and Data Science apps. It allows the Python code to serve as both the backend (ML logic) and the frontend (UI generation).
- **Efficiency**: For ML prototypes, this "single-stack" approach is much faster to build, deploy, and maintain compared to setting up a separate React frontend and Flask/FastAPI backend.
- **ML Logic**: The backend logic is encapsulated in `recommendation_engine.py`, while the frontend is handled in `app.py`.

## 🚀 How to Run the Project

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Generate Data
If you want to refresh the synthetic music dataset:
```bash
python data_generator.py
```

### 3. Launch the Web App
Run the following command to start the server:
```bash
streamlit run app.py
```

### 4. Access the UI
Once the command is running, open your browser to:
**[http://localhost:8501](http://localhost:8501)**

## 🧠 Behind the Scenes
- **Content-Based Filtering**: Analyzes audio DNA (tempo, mood, genre) using Cosine Similarity.
- **Collaborative Filtering**: Identifies patterns between users with similar tastes.
- **Preference-Based Engine**: Uses Euclidean distance-based ranking to match user-selected energy and danceability levels.
- **Hybrid Engine**: Combines all the above methods for the most accurate and personalized recommendations.
