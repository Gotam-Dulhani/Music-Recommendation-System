# AI Music Recommendation System

[![Contributors](https://img.shields.io/github/contributors/Gotam-Dulhani/Music-Recommendation-System)](https://github.com/Gotam-Dulhani/Music-Recommendation-System/graphs/contributors)
[![Forks](https://img.shields.io/github/forks/Gotam-Dulhani/Music-Recommendation-System)](https://github.com/Gotam-Dulhani/Music-Recommendation-System/network/members)
[![Stars](https://img.shields.io/github/stars/Gotam-Dulhani/Music-Recommendation-System)](https://github.com/Gotam-Dulhani/Music-Recommendation-System/stargazers)
[![Issues](https://img.shields.io/github/issues/Gotam-Dulhani/Music-Recommendation-System)](https://github.com/Gotam-Dulhani/Music-Recommendation-System/issues)
[![License](https://img.shields.io/github/license/Gotam-Dulhani/Music-Recommendation-System)](https://github.com/Gotam-Dulhani/Music-Recommendation-System/blob/main/LICENSE)

> **VibeStream** - an AI music discovery platform that recommends tracks using content-based filtering, collaborative filtering, and a hybrid ML scoring engine. Deployed on Vercel with a Flask backend.

---

## Table of Contents

* [About The Project](#about-the-project)
* [Key Features](#key-features)
* [Live Demo](#live-demo)
* [Built With](#built-with)
* [How It Works](#how-it-works)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
* [Deployment](#deployment)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## About The Project

**VibeStream** is an AI-powered music recommendation system with a Flask web interface deployed on Vercel. It combines three recommendation strategies:

1. **Content-Based Filtering** - Computes cosine similarity between songs using weighted audio features (energy, danceability, tempo) and categorical features (genre, mood). Audio features are weighted 3x higher than genre/mood to prevent categorical dominance and ensure recommendations consider actual sound characteristics.

2. **Collaborative Filtering** - Identifies the 10 most similar listeners based on listening history, normalizes their interaction scores by activity level, and recommends tracks the target user hasn't heard.

3. **Hybrid Engine** - Merges both approaches with a 55/45 weighted split (content/collaborative), averages duplicate recommendations, and boosts songs found by both methods by 15%.

---

## Key Features

- **Hybrid AI Recommendations** - Weighted combination of content-based and collaborative filtering
- **Content-Based Audio DNA** - Cosine similarity with properly weighted audio features (energy, danceability, tempo) + genre/mood encoding
- **Collaborative Listener Matching** - Finds top-10 similar users, normalizes by activity, surfaces unexplored tracks
- **Preference Sliders** - Fine-tune discovery by energy, danceability, and tempo (BPM) in real time
- **Song Search** - Search by title or artist to find tracks with matching audio profiles
- **Dark Mode UI** - Responsive design with card-based layouts and score badges
- **Synthetic Data Generation** - Built-in generator for 500 songs and 50 user interaction profiles

---

## Live Demo

The app is deployed on Vercel: `https://music-recommendation-system-<your-project>.vercel.app`

---

## Built With

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| Flask | Web framework and WSGI entry point for Vercel |
| scikit-learn | Cosine similarity, MinMaxScaler, StandardScaler |
| pandas / NumPy | Data processing and feature engineering |
| Vercel | Serverless deployment platform |

---

## How It Works

```
User Input (genre / mood / energy / danceability / tempo)
        |
        v
Feature Engineering
  - One-hot encode genre (8) + mood (6)
  - MinMaxScaler on energy, danceability, tempo
  - Weight: audio features x3, genre/mood x 1/sqrt(cols)
        |
        v
Content-Based Filtering
  - Cosine similarity matrix (500x500 songs)
  - Top-N most similar tracks, excluding user history
        |
        v
Collaborative Filtering
  - User-item interaction matrix (50 users x 472 songs)
  - Cosine similarity between users
  - Top-10 similar users, weighted recommendations
  - Normalized by per-user activity level
        |
        v
Hybrid Scoring
  - content_score * 0.55 + collab_score * 0.45
  - +0.15 boost for songs found by both methods
        |
        v
Ranked Recommendations (top-8)
```

### Recommendation Methods

| Method | Input | Algorithm | Output |
|---|---|---|---|
| **Content-Based** | Song ID | Cosine similarity on weighted audio features | Similar tracks with match score |
| **Collaborative** | User ID | User similarity x normalized interaction scores | Tracks from similar listeners |
| **Hybrid** | User ID | Weighted blend of content + collaborative | Combined ranked playlist |
| **Preferences** | Genre/Mood/Sliders | Euclidean distance in MinMax-scaled 3D space | Tracks matching desired vibe |
| **Search** | Text query | String matching on title/artist + content recs | Search results + similar tracks |

---

## Project Structure

```
Music-Recommendation-System/
|
|-- api/
|   +-- index.py                # Flask WSGI app (Vercel entry point)
|-- templates/
|   +-- index.html              # HTML template with inline CSS
|-- static/
|   +-- style.css               # CSS (reference copy, inlined in template)
|-- recommendation_engine.py    # ML engine: content, collaborative, hybrid filtering
|-- data_generator.py           # Generates synthetic songs.csv and interactions.csv
|-- app.py                      # Streamlit version (for local use)
|-- get_yt_info.py              # YouTube track info utility
|-- songs.csv                   # 500 synthetic songs with audio features
|-- interactions.csv            # 1,451 user-song interactions (50 users)
|-- vercel.json                 # Vercel deployment configuration
|-- requirements.txt            # Python dependencies (pandas, numpy, scikit-learn, flask)
|-- .gitignore
+-- README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Gotam-Dulhani/Music-Recommendation-System.git
cd Music-Recommendation-System
```

**2. Create a virtual environment** *(recommended)*
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Generate the dataset** *(optional - refresh synthetic music data)*
```bash
python data_generator.py
```

### Running Locally

**Option A: Flask (same as production)**
```bash
flask --app api.index run
```
Open [http://localhost:5000](http://localhost:5000)

**Option B: Streamlit (original UI)**
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501)

---

## Deployment

### Vercel

1. Push to GitHub
2. Import repository on [vercel.com/new](https://vercel.com/new)
3. Vercel auto-detects the Python runtime from `vercel.json` and `api/index.py`
4. Deploy - no build configuration needed

The `vercel.json` routes all requests to the Flask app in `api/index.py`:
```json
{
    "builds": [{ "src": "api/index.py", "use": "@vercel/python" }],
    "routes": [
        { "src": "/static/(.*)", "dest": "/static/$1" },
        { "src": "/(.*)", "dest": "/api/index.py" }
    ]
}
```

---

## Contributing

1. Fork the repo
2. Create a feature branch:
```bash
git checkout -b feature/AmazingFeature
```
3. Commit your changes:
```bash
git commit -m "Add AmazingFeature"
```
4. Push and open a Pull Request:
```bash
git push origin feature/AmazingFeature
```

---

## License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## Contact

**Gotam Dulhani**
GitHub: [https://github.com/Gotam-Dulhani](https://github.com/Gotam-Dulhani)

---

## Acknowledgments

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [Vercel Python Runtime](https://vercel.com/docs/functions/runtimes/python)
