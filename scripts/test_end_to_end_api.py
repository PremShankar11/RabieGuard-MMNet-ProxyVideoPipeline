"""
Test script to run end-to-end API screening on both:
1. A real video with embedded audio track.
2. A real video without an audio track (fallback verification).
"""

import urllib.request
from pathlib import Path
import json

def post_video(url, video_path):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(video_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="video"; filename="{Path(video_path).name}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode("latin-1") + file_bytes + f"\r\n--{boundary}--\r\n".encode("latin-1")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    v_with_audio = r"C:\Users\Asus\Downloads\Normal dog GREY SCALE\Normal dog\barking\Greyscale\g_Barking - Made With Clipchamp.mp4"
    v_no_audio = r"C:\Users\Asus\Downloads\demo_dog_video.mp4"

    print("=================================================================")
    print("TEST 1: Real Canine Video WITH Embedded Audio")
    print(f"File: {v_with_audio}")
    print("=================================================================")
    res1 = post_video("http://127.0.0.1:8000/api/analyze", v_with_audio)
    print("Inference Successful!")
    print(f"Screening Pipeline : {res1['input_summary']['screening_pipeline']}")
    print(f"Audio Detected     : {res1['input_summary']['embedded_audio_detected']}")
    print(f"Audio Extraction   : {res1['input_summary']['audio_extraction_note']}")
    print(f"Video Score / Prob : {res1['video']['score']} / {res1['video']['probability']} (Conf: {res1['video']['confidence']})")
    print(f"Audio Score / Prob : {res1['audio']['score']} / {res1['audio']['probability']} (Conf: {res1['audio']['confidence']})")
    print(f"Modality Weights   : Wv = {res1['weights']['video']}, Wa = {res1['weights']['audio']}")
    print(f"Fused Risk Score   : {res1['fused']['risk_score']} / 100")
    print(f"Fused Risk Level   : {res1['fused']['risk_level']}")
    print(f"Fused Probability  : {res1['fused']['risk_probability']}")
    print(f"Number of Segments : {len(res1['segments'])}")
    print(f"Annotated Video URL: {res1.get('annotated_video_url')}")

    print("\n=================================================================")
    print("TEST 2: Real Canine Video WITHOUT Audio Track (Fallback)")
    print(f"File: {v_no_audio}")
    print("=================================================================")
    res2 = post_video("http://127.0.0.1:8000/api/analyze", v_no_audio)
    print("Inference Successful!")
    print(f"Screening Pipeline : {res2['input_summary']['screening_pipeline']}")
    print(f"Audio Detected     : {res2['input_summary']['embedded_audio_detected']}")
    print(f"Audio Extraction   : {res2['input_summary']['audio_extraction_note']}")
    print(f"Video Score / Prob : {res2['video']['score']} / {res2['video']['probability']} (Conf: {res2['video']['confidence']})")
    print(f"Audio Score / Prob : {res2['audio']['score']} / {res2['audio']['probability']} (Conf: {res2['audio']['confidence']})")
    print(f"Modality Weights   : Wv = {res2['weights']['video']}, Wa = {res2['weights']['audio']}")
    print(f"Fused Risk Score   : {res2['fused']['risk_score']} / 100")
    print(f"Fused Risk Level   : {res2['fused']['risk_level']}")
    print(f"Fused Probability  : {res2['fused']['risk_probability']}")
    print(f"Number of Segments : {len(res2['segments'])}")

if __name__ == "__main__":
    main()
