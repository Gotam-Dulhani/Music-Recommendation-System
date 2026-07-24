import os
import sys
from flask import Flask, render_template, request, jsonify

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

from recommendation_engine import MusicRecommender

app = Flask(
    __name__,
    template_folder=os.path.join(_base_dir, 'templates'),
    static_folder=os.path.join(_base_dir, 'static'),
)

engine = None

def get_engine():
    global engine
    if engine is None:
        songs_path = os.path.join(_base_dir, 'songs.csv')
        interactions_path = os.path.join(_base_dir, 'interactions.csv')
        engine = MusicRecommender(songs_path, interactions_path)
    return engine


def format_score(row, score_col, score_name):
    eng = get_engine()
    score_val = row.get(score_col)
    if score_val is None:
        return ""
    if score_col == 'similarity_score':
        disp = f"{score_val * 100:.0f}%"
    elif score_col == 'collab_score':
        disp = f"{score_val * 100:.0f}%"
    elif score_col == 'hybrid_score':
        pct = min(score_val / 1.15, 1.0) * 100
        disp = f"{pct:.0f}%"
    elif score_col == 'preference_score':
        disp = f"{score_val:.0f}%"
    else:
        disp = f"{score_val:.1f}"
    return f'{score_name}: {disp}'


app.jinja_env.globals.update(format_score=format_score)


@app.route('/')
def index():
    eng = get_engine()
    users = sorted(eng.interactions_df['user_id'].unique().tolist())
    genres = ["Any"] + sorted(eng.songs_df['genre'].unique().tolist())
    moods = ["Any"] + sorted(eng.songs_df['mood'].unique().tolist())
    tempo_min = int(eng.songs_df['tempo'].min())
    tempo_max = int(eng.songs_df['tempo'].max())

    selected_user = request.args.get('user', users[0] if users else '')
    tab = request.args.get('tab', 'hybrid')

    history = eng.get_user_history(selected_user)
    popular = eng.get_popular_songs(8) if history.empty else None

    hybrid_recs = None
    content_recs = None
    collab_recs = None
    pref_recs = None
    search_recs = None
    search_target = None
    top_song = None

    if not history.empty:
        if tab == 'hybrid':
            hybrid_recs = eng.get_hybrid_recommendations(selected_user, top_n=8)
        elif tab == 'content':
            top_song = history.iloc[0]
            content_recs = eng.get_content_recommendations(top_song['song_id'], top_n=8)
        elif tab == 'collab':
            collab_recs = eng.get_collaborative_recommendations(selected_user, top_n=8)
        elif tab == 'preferences':
            genre = request.args.get('genre', 'Any')
            mood = request.args.get('mood', 'Any')
            energy = float(request.args.get('energy', 0.5))
            dance = float(request.args.get('danceability', 0.5))
            tempo_val = request.args.get('tempo')
            tempo = float(tempo_val) if tempo_val else None
            pref_recs = eng.get_recommendations_by_preferences(
                genre=genre, mood=mood, energy=energy,
                danceability=dance, tempo=tempo
            )
        elif tab == 'search':
            query = request.args.get('q', '')
            rec_for = request.args.get('rec_for', '')
            if query:
                results = eng.songs_df[
                    eng.songs_df['title'].str.contains(query, case=False, na=False) |
                    eng.songs_df['artist'].str.contains(query, case=False, na=False)
                ].head(5)
            else:
                results = None
            if rec_for:
                search_recs = eng.get_content_recommendations(rec_for, top_n=8)
                match = eng.songs_df[eng.songs_df['song_id'] == rec_for]
                search_target = match.iloc[0]['title'] if not match.empty else ''
        else:
            results = None
    else:
        results = None

    return render_template(
        'index.html',
        users=users,
        selected_user=selected_user,
        tab=tab,
        history=history,
        popular=popular,
        hybrid_recs=hybrid_recs,
        content_recs=content_recs,
        collab_recs=collab_recs,
        pref_recs=pref_recs,
        genres=genres,
        moods=moods,
        tempo_min=tempo_min,
        tempo_max=tempo_max,
        top_song=top_song,
        search_recs=search_recs,
        search_target=search_target,
        query=request.args.get('q', ''),
        results=results if tab == 'search' and request.args.get('q') else None,
    )
