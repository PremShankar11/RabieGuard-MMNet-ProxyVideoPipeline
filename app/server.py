"""
Zero Rabies-MMNet — Video V2 Web Application Server.
Lightweight, robust HTTP server powered by Python's standard library.
Connects directly to the frozen Video V2 inference pipeline without external web frameworks.
"""

import sys
import os
import io
import re
import json
import mimetypes
import argparse
from pathlib import Path
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.parse
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.video.inference import VideoBehaviorPipeline, InferenceError
from src.video.preprocessing import VideoReadError, VideoValidationError

# Global pipeline instance initialized on startup
PIPELINE: VideoBehaviorPipeline = None
STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = repo_root / "outputs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR = repo_root / "outputs" / "inference" / "annotated"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


def parse_multipart(body: bytes, boundary: str) -> dict:
    """Robust multipart/form-data parser for file uploads without external dependencies."""
    boundary_bytes = boundary.encode("latin-1")
    parts = body.split(b"--" + boundary_bytes)
    result = {"fields": {}, "files": {}}

    for part in parts:
        if not part or part == b"--\r\n" or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue

        headers_raw, content = part.split(b"\r\n\r\n", 1)
        if content.endswith(b"\r\n"):
            content = content[:-2]

        headers_text = headers_raw.decode("latin-1", errors="replace")
        disp_match = re.search(r'Content-Disposition:\s*form-data;\s*name="([^"]+)"(?:;\s*filename="([^"]+)")?', headers_text, re.IGNORECASE)
        if not disp_match:
            continue

        field_name = disp_match.group(1)
        filename = disp_match.group(2)

        if filename is not None:
            # File part
            clean_filename = Path(filename).name
            result["files"][field_name] = {
                "filename": clean_filename,
                "content": content
            }
        else:
            # Text field
            result["fields"][field_name] = content.decode("utf-8", errors="replace")

    return result


class VideoScreeningRequestHandler(BaseHTTPRequestHandler):
    server_version = "ZeroRabies-MMNet/2.0"

    def do_OPTIONS(self):
        """Handle CORS pre-flight."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_file(STATIC_DIR / "index.html", "text/html")
        elif path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = (STATIC_DIR / rel_path).resolve()
            # Prevent path traversal
            if STATIC_DIR in file_path.parents or file_path == STATIC_DIR:
                self._serve_file(file_path)
            else:
                self._send_error(HTTPStatus.FORBIDDEN, "Forbidden")
        elif path == "/api/status":
            self._serve_status()
        elif path == "/api/manifest":
            manifest_file = repo_root / "reports" / "behavior_v2" / "FROZEN_MODEL_MANIFEST.json"
            if manifest_file.exists():
                self._serve_file(manifest_file, "application/json")
            else:
                self._send_json({"error": "Manifest not found"}, status=HTTPStatus.NOT_FOUND)
        elif path.startswith("/api/video/"):
            self._serve_video_stream(path)
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "Resource not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/analyze":
            self._handle_analyze()
        elif parsed.path == "/api/analyze_demo":
            self._handle_analyze_demo()
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _serve_status(self):
        global PIPELINE
        demo_clips = []
        raw_v_dir = repo_root / "data" / "raw" / "animal_kingdom" / "video"
        demo_candidates = [
            {"id": "YSWGBGCS", "name": "Dev Clip YSWGBGCS (Calm / Still, Short ~0.4s)", "action": "Keeping still"},
            {"id": "AWJEUGCS", "name": "Dev Clip AWJEUGCS (Agitated / Running, ~9.8s)", "action": "Running"},
            {"id": "EUKRNXGD", "name": "Dev Clip EUKRNXGD (Multi-Dog Scene, ~7.6s)", "action": "Running / Multi-Dog"},
        ]
        for dc in demo_candidates:
            f = raw_v_dir / f"{dc['id']}.mp4"
            if f.exists():
                demo_clips.append({**dc, "size_bytes": f.stat().st_size})

        data = {
            "status": "ready",
            "model_name": "Zero Rabies-MMNet Video V2",
            "experiment": "exp_d_temporal_tracking",
            "checkpoint": "checkpoints/mamba_behavior_v2/final.pt",
            "frozen": True,
            "device": str(PIPELINE.device) if PIPELINE else "unknown",
            "trainable_parameters": 136257,
            "demo_clips": demo_clips,
            "server_time": datetime.now().isoformat()
        }
        self._send_json(data)

    def _handle_analyze(self):
        global PIPELINE
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json({"error": "Content-Type must be multipart/form-data"}, status=HTTPStatus.BAD_REQUEST)
            return

        boundary_match = re.search(r"boundary=([^\s;]+)", content_type)
        if not boundary_match:
            self._send_json({"error": "No multipart boundary found"}, status=HTTPStatus.BAD_REQUEST)
            return
        boundary = boundary_match.group(1).strip('"')

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            parsed = parse_multipart(body, boundary)

            if "video" not in parsed["files"]:
                self._send_json({"error": "No 'video' file provided in form upload"}, status=HTTPStatus.BAD_REQUEST)
                return

            uploaded_file = parsed["files"]["video"]
            filename = uploaded_file["filename"]
            content = uploaded_file["content"]

            if not filename or len(content) == 0:
                self._send_json({"error": "Uploaded video file is empty"}, status=HTTPStatus.BAD_REQUEST)
                return

            # Save uploaded file
            safe_name = f"upload_{int(datetime.now().timestamp())}_{filename}"
            save_path = UPLOAD_DIR / safe_name
            with open(save_path, "wb") as f:
                f.write(content)

            # Generate unique annotated output path
            annotated_name = f"annotated_{int(datetime.now().timestamp())}_{Path(filename).stem}.mp4"
            annotated_path = ANNOTATED_DIR / annotated_name

            # Run inference
            result = PIPELINE.process_video(
                video_path=save_path,
                save_json=True,
                generate_annotated_video=True,
                annotated_video_path=annotated_path
            )

            # Add playback stream URLs
            result["original_video_url"] = f"/api/video/upload/{safe_name}"
            if annotated_path.exists():
                result["annotated_video_url"] = f"/api/video/annotated/{annotated_name}"
            else:
                result["annotated_video_url"] = None

            self._send_json(result)

        except VideoValidationError as ve:
            self._send_json({"error": str(ve), "type": "ValidationError"}, status=HTTPStatus.BAD_REQUEST)
        except VideoReadError as re_err:
            self._send_json({"error": str(re_err), "type": "ReadError"}, status=HTTPStatus.UNPROCESSABLE_ENTITY)
        except InferenceError as ie:
            self._send_json({"error": str(ie), "type": "InferenceQualityError"}, status=HTTPStatus.UNPROCESSABLE_ENTITY)
        except Exception as e:
            self._send_json({"error": f"Internal inference pipeline error: {str(e)}", "type": "ServerError"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_analyze_demo(self):
        global PIPELINE
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8") or "{}")
            clip_id = payload.get("clip_id")

            if not clip_id:
                self._send_json({"error": "Missing 'clip_id' parameter"}, status=HTTPStatus.BAD_REQUEST)
                return

            raw_video = repo_root / "data" / "raw" / "animal_kingdom" / "video" / f"{clip_id}.mp4"
            if not raw_video.exists():
                self._send_json({"error": f"Demo video '{clip_id}' not found on server"}, status=HTTPStatus.NOT_FOUND)
                return

            annotated_name = f"annotated_{clip_id}_{int(datetime.now().timestamp())}.mp4"
            annotated_path = ANNOTATED_DIR / annotated_name

            result = PIPELINE.process_video(
                video_path=raw_video,
                save_json=True,
                generate_annotated_video=True,
                annotated_video_path=annotated_path
            )

            result["original_video_url"] = f"/api/video/raw/{clip_id}.mp4"
            if annotated_path.exists():
                result["annotated_video_url"] = f"/api/video/annotated/{annotated_name}"
            else:
                result["annotated_video_url"] = None

            self._send_json(result)

        except Exception as e:
            self._send_json({"error": f"Demo analysis error: {str(e)}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _serve_video_stream(self, path: str):
        """Serve video files supporting HTTP Range requests for smooth browser playback."""
        parts = path.strip("/").split("/")
        if len(parts) < 4:
            self._send_error(HTTPStatus.BAD_REQUEST, "Invalid video path")
            return

        category = parts[2]
        filename = parts[3]

        if category == "upload":
            target = UPLOAD_DIR / filename
        elif category == "annotated":
            target = ANNOTATED_DIR / filename
        elif category == "raw":
            target = repo_root / "data" / "raw" / "animal_kingdom" / "video" / filename
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown video category")
            return

        target = target.resolve()
        if not target.exists() or not target.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Video file not found")
            return

        file_size = target.stat().st_size
        range_header = self.headers.get("Range")

        if range_header:
            range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
                length = end - start + 1

                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Content-Disposition", "inline")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                with open(target, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(file_size))
        self.send_header("Content-Disposition", "inline")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        with open(target, "rb") as f:
            chunk = f.read(65536)
            while chunk:
                self.wfile.write(chunk)
                chunk = f.read(65536)

    def _serve_file(self, file_path: Path, content_type: Optional[str] = None):
        if not file_path.exists() or not file_path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        if content_type is None:
            content_type, _ = mimetypes.guess_type(str(file_path))
            content_type = content_type or "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Error reading file: {e}")

    def _send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status: HTTPStatus, message: str):
        self._send_json({"error": message, "code": status.value}, status=status)

    def log_message(self, format, *args):
        # Clean server logging format
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} - {format % args}")


def run_server(host: str = "127.0.0.1", port: int = 8000):
    global PIPELINE
    print("\n" + "=" * 65)
    print("ZERO RABIES-MMNET — VIDEO V2 SCREENING FRONTEND SERVER")
    print("=" * 65)
    print("Initializing Frozen Video V2 Pipeline...")

    PIPELINE = VideoBehaviorPipeline()
    print(f"Loaded Frozen Mamba S6 Checkpoint: {PIPELINE.mamba_classifier.checkpoint_path.name}")
    print(f"Loaded YOLO Pose Checkpoint: {PIPELINE.pose_estimator.checkpoint_path.name}")
    print(f"Inference Device: {PIPELINE.device}")

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, VideoScreeningRequestHandler)
    print(f"\nServer live at: http://{host}:{port}/")
    print("Serving Video Behavioral Screening Research Prototype")
    print("Press Ctrl+C to terminate.")
    print("=" * 65 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero Rabies-MMNet Video V2 Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
