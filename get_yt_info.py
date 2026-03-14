import sys
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

url = "https://youtu.be/atI3IOv4S2Q"
video_id = "atI3IOv4S2Q"

try:
    yt = YouTube(url)
    print(f"Title: {yt.title}")
    # print(f"Description: {yt.description}")
except Exception as e:
    print(f"Error getting metadata: {e}")

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = " ".join([t['text'] for t in transcript_list])
    print("\nTranscript Snippet:")
    print(transcript[:1500])
except Exception as e:
    print(f"Error getting transcript: {e}")
