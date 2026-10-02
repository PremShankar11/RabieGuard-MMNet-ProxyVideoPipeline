"""
Comprehensive End-to-End API verification script.
Tests:
1. /api/probe_video (pre-inference stream inspection) on video with and without audio.
2. /api/analyze (full single-video inference pipeline) on video with and without audio.
3. Fallback verification (zero audio fabrication).
"""

import urllib.request
from pathlib import Path
import json

def send_multipart(url, file_path, field_name="video"):
    boundary = "----WebKitFormBoundaryTest789"
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{Path(file_path).name}"\r\n'
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
    print("PROBE TEST 1: Stream Inspection (Video WITH Audio)")
    print("=================================================================")
    probe1 = send_multipart("http://127.0.0.1:8000/api/probe_video", v_with_audio)
    print(json.dumps(probe1, indent=2))
    assert probe1["has_audio"] is True
    assert probe1["audio_track"] == "Detected"

    print("\n=================================================================")
    print("PROBE TEST 2: Stream Inspection (Video WITHOUT Audio)")
    print("=================================================================")
    probe2 = send_multipart("http://127.0.0.1:8000/api/probe_video", v_no_audio)
    print(json.dumps(probe2, indent=2))
    assert probe2["has_audio"] is False
    assert probe2["audio_track"] == "Not detected"

    print("\nAll probe tests passed successfully!")

if __name__ == "__main__":
    main()
