"""
Zero Rabies-MMNet — Multimodal Behavioral Screening Web Application Server.
Powered by Python's standard library. Connects directly to the frozen Video V2 pipeline,
frozen Audio V2 pipeline, and the Dynamic Confidence-Aware Late Fusion Engine.

Single-Video Workflow:
The user uploads ONE dog video.
The backend:
1. Receives the uploaded video.
2. Inspects media streams for video and embedded audio tracks.
3. Runs frozen Video V2 (Mamba S6 + Dog-Pose tracker).
4. If an audio stream exists:
   - extracts 16 kHz mono PCM WAV directly from the same uploaded video.
   - runs frozen Audio V2 (AudioNet v2 + Isotonic calibration).
   If no audio stream exists:
   - falls back gracefully to Video-Only screening (Wv=1.0, Wa=0.0).
5. Synchronizes predictions on a 0.5s temporal grid.
6. Executes Dynamic Confidence-Aware Late Fusion.
7. Returns complete multimodal screening results with operational disclaimer.
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
from src.audio.pipeline import AudioBehaviorPipeline, AudioPipelineError
from src.audio.extraction import inspect_video_streams, extract_audio_from_video, get_ffmpeg_path
from fusion.dynamic_late_fusion import DynamicLateFusionEngine
from scripts.run_fusion_validation import build_synthetic_scenarios

# Global pipeline instances initialized on startup
VIDEO_PIPELINE: Optional[VideoBehaviorPipeline] = None
AUDIO_PIPELINE: Optional[AudioBehaviorPipeline] = None
FUSION_ENGINE: Optional[DynamicLateFusionEngine] = None

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
        disp_match = re.search(
            r'Content-Disposition:\s*form-data;\s*name="([^"]+)"(?:;\s*filename="([^"]+)")?',
            headers_text,
            re.IGNORECASE
        )
        if not disp_match:
            continue

        field_name = disp_match.group(1)
        filename = disp_match.group(2)

        if filename is not None:
            clean_filename = Path(filename).name
            result["files"][field_name] = {
                "filename": clean_filename,
                "content": content
            }
        else:
            result["fields"][field_name] = content.decode("utf-8", errors="replace")

    return result


class MultimodalScreeningRequestHandler(BaseHTTPRequestHandler):
    server_version = "ZeroRabies-MMNet/2.0-SingleVideoMultimodal"

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
        elif path == "/api/fusion_config":
            cfg_file = repo_root / "reports" / "fusion" / "DYNAMIC_LATE_FUSION_CONFIG.json"
            if cfg_file.exists():
                self._serve_file(cfg_file, "application/json")
            else:
                self._send_json({"error": "Fusion config not found"}, status=HTTPStatus.NOT_FOUND)
        elif path.startswith("/api/video/"):
            self._serve_video_stream(path)
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "Resource not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ["/api/analyze", "/api/analyze_multimodal"]:
            self._handle_analyze_single_video()
        elif parsed.path in ["/api/probe_video", "/api/inspect_video"]:
            self._handle_probe_video()
        elif parsed.path in ["/api/analyze_demo", "/api/analyze_multimodal_demo"]:
            self._handle_analyze_multimodal_demo()
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _serve_status(self):
        global VIDEO_PIPELINE, AUDIO_PIPELINE, FUSION_ENGINE
        scenarios = build_synthetic_scenarios()
        demo_scenarios = [
            {
                "id": s["clip_id"],
                "name": s["clip_id"].replace("SYNTH_", "").replace("_", " ").title(),
                "description": s["description"]
            }
            for s in scenarios
        ]

        ffmpeg_path = None
        try:
            ffmpeg_path = get_ffmpeg_path()
        except Exception:
            ffmpeg_path = "Unavailable"

        data = {
            "status": "ready",
            "prototype": "Zero Rabies-MMNet Multimodal Behavioral Screening",
            "workflow": "single_video_embedded_audio_extraction",
            "fusion_method": "dynamic_confidence_late_fusion",
            "ffmpeg_path": ffmpeg_path,
            "video_model": {
                "name": "Mamba S6 V2 (Frozen)",
                "checkpoint": "checkpoints/mamba_behavior_v2/final.pt",
                "parameters": 136257,
                "status": "FROZEN"
            },
            "audio_model": {
                "name": "AudioNet v2 (Frozen)",
                "checkpoint": "Audio pipeline/audio_model_v2.pth",
                "parameters": 6179714,
                "status": "FROZEN"
            },
            "demo_scenarios": demo_scenarios,
            "disclaimer": "Research screening/risk-assessment prototype. NOT a clinical rabies diagnostic tool."
        }
        self._send_json(data)

    def _handle_probe_video(self):
        """
        Lightweight pre-inference stream inspection.
        Accepts a single video file, inspects container for video and audio streams,
        and returns detection metadata for immediate UI feedback.
        """
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

            v_file = parsed["files"].get("video") or parsed["files"].get("file")
            if not v_file or not v_file["content"]:
                self._send_json({"error": "No video file provided for probing"}, status=HTTPStatus.BAD_REQUEST)
                return

            v_name = v_file["filename"]
            ext = Path(v_name).suffix.lower()
            if ext not in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                self._send_json(
                    {"error": f"Unsupported video format: '{ext}'. Supported formats: MP4, MOV, AVI.", "valid": False},
                    status=HTTPStatus.BAD_REQUEST
                )
                return

            temp_probe_name = f"probe_{int(datetime.now().timestamp())}_{v_name}"
            temp_probe_path = UPLOAD_DIR / temp_probe_name
            with open(temp_probe_path, "wb") as f:
                f.write(v_file["content"])

            try:
                stream_info = inspect_video_streams(temp_probe_path)
            finally:
                if temp_probe_path.exists():
                    try:
                        temp_probe_path.unlink()
                    except Exception:
                        pass

            if not stream_info["has_video"]:
                self._send_json({
                    "valid": False,
                    "error": "The uploaded file does not contain a readable video stream.",
                    "status": "Invalid Video Stream"
                }, status=HTTPStatus.BAD_REQUEST)
                return

            has_audio = stream_info["has_audio"]
            status_text = "Ready for multimodal screening" if has_audio else "Ready for video-only screening (no embedded audio detected)"

            response_data = {
                "valid": True,
                "filename": v_name,
                "file_size_mb": round(len(v_file["content"]) / (1024 * 1024), 2),
                "duration_seconds": stream_info.get("duration_seconds"),
                "video_codec": stream_info.get("video_codec"),
                "has_audio": has_audio,
                "audio_track": "Detected" if has_audio else "Not detected",
                "audio_codec": stream_info.get("audio_codec"),
                "audio_sample_rate": stream_info.get("audio_sample_rate"),
                "audio_channels": stream_info.get("audio_channels"),
                "audio_processing": "16 kHz mono (auto-extract)" if has_audio else "None (Video-only fallback)",
                "status": status_text
            }
            self._send_json(response_data)

        except Exception as e:
            self._send_json({"error": f"Error inspecting video stream: {str(e)}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_analyze_single_video(self):
        """
        Unified Single-Video Inference Workflow:
        1. Receive uploaded canine video.
        2. Run frozen Video V2 pipeline (Mamba S6 + YOLO pose).
        3. Inspect for embedded audio stream.
        4. If audio exists: extract 16 kHz mono PCM WAV and run frozen Audio V2 (AudioNet v2).
           If no audio: fallback cleanly to video-only screening.
        5. Synchronize and run Dynamic Confidence-Aware Late Fusion.
        """
        global VIDEO_PIPELINE, AUDIO_PIPELINE, FUSION_ENGINE
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

            # Accept single video input (field 'video' or 'file')
            v_file = parsed["files"].get("video") or parsed["files"].get("file")
            if not v_file or not v_file["content"]:
                self._send_json(
                    {"error": "Please select and upload one dog video file (MP4, MOV, AVI)."},
                    status=HTTPStatus.BAD_REQUEST
                )
                return

            v_name = v_file["filename"]
            ext = Path(v_name).suffix.lower()
            if ext not in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                self._send_json(
                    {"error": f"Unsupported video file format '{ext}'. Accepted formats: MP4, MOV, AVI."},
                    status=HTTPStatus.BAD_REQUEST
                )
                return

            # Save uploaded video
            safe_video_name = f"upload_{int(datetime.now().timestamp())}_{v_name}"
            save_v_path = UPLOAD_DIR / safe_video_name
            with open(save_v_path, "wb") as f:
                f.write(v_file["content"])

            # Inspect media container streams
            stream_info = inspect_video_streams(save_v_path)
            if not stream_info["has_video"]:
                save_v_path.unlink(missing_ok=True)
                self._send_json(
                    {"error": "The uploaded file does not contain a readable video stream.", "type": "VideoReadError"},
                    status=HTTPStatus.UNPROCESSABLE_ENTITY
                )
                return

            # 1. Run Frozen Video V2 Pipeline
            annotated_name = f"annotated_{int(datetime.now().timestamp())}_{Path(v_name).stem}.mp4"
            annotated_path = ANNOTATED_DIR / annotated_name

            video_result = VIDEO_PIPELINE.process_video(
                video_path=save_v_path,
                save_json=True,
                generate_annotated_video=True,
                annotated_video_path=annotated_path
            )
            video_result["original_video_url"] = f"/api/video/upload/{safe_video_name}"
            video_result["annotated_video_url"] = f"/api/video/annotated/{annotated_name}" if annotated_path.exists() else None

            # 2. Extract Embedded Audio & Run Frozen Audio V2 Pipeline
            audio_result = None
            audio_extraction_note = None
            embedded_audio_detected = stream_info["has_audio"]

            if embedded_audio_detected:
                temp_wav_name = f"extract_{int(datetime.now().timestamp())}_{Path(v_name).stem}.wav"
                temp_wav_path = UPLOAD_DIR / temp_wav_name
                try:
                    extracted = extract_audio_from_video(save_v_path, temp_wav_path, sample_rate=16000)
                    if extracted and temp_wav_path.exists():
                        audio_result = AUDIO_PIPELINE.process_audio(temp_wav_path)
                        audio_extraction_note = "Embedded audio track detected, extracted, and resampled to 16 kHz mono."
                    else:
                        audio_result = None
                        audio_extraction_note = "Audio extraction failed; video-only screening was used."
                except Exception as a_err:
                    audio_result = None
                    audio_extraction_note = f"Audio extraction failed ({str(a_err)}); video-only screening was used."
                finally:
                    if temp_wav_path.exists():
                        try:
                            temp_wav_path.unlink()
                        except Exception:
                            pass
            else:
                audio_result = None
                audio_extraction_note = "Embedded audio track not detected. Video-only screening was used."

            # 3. Dynamic Confidence-Aware Late Fusion
            clip_label = Path(v_name).stem
            fused_output = FUSION_ENGINE.fuse_multimodal(
                video_result=video_result,
                audio_result=audio_result,
                clip_id=clip_label,
                save_csv=False
            )

            # Attach media URLs and preview frames
            fused_output["original_video_url"] = video_result.get("original_video_url")
            fused_output["annotated_video_url"] = video_result.get("annotated_video_url")
            fused_output["preview_frames"] = video_result.get("preview_frames", [])

            # Provenance and input stream metadata
            fused_output["input_summary"] = {
                "video_filename": v_name,
                "video_size_mb": round(len(v_file["content"]) / (1024 * 1024), 2),
                "video_duration_seconds": video_result.get("video", {}).get("duration_seconds", stream_info.get("duration_seconds")),
                "embedded_audio_detected": embedded_audio_detected,
                "audio_codec": stream_info.get("audio_codec"),
                "audio_sample_rate": stream_info.get("audio_sample_rate"),
                "audio_channels": stream_info.get("audio_channels"),
                "audio_extraction_note": audio_extraction_note,
                "screening_pipeline": (
                    "Multimodal (Video V2 + Embedded Audio V2)"
                    if (audio_result and audio_result.get("audio_present"))
                    else ("Video V2 Only (Audio Silent)" if audio_result else "Video V2 Only (No Audio Track)")
                )
            }

            # Video quality information
            fused_output["video_quality_info"] = {
                "overall": video_result.get("quality", {}).get("overall", "GOOD"),
                "valid_frame_percentage": video_result.get("quality", {}).get("valid_frame_percentage", 100.0),
                "mean_pose_confidence": video_result.get("quality", {}).get("mean_pose_confidence", 0.9),
                "ambiguity_level": video_result.get("quality", {}).get("ambiguity_level", "LOW"),
                "ambiguous_frame_percentage": video_result.get("quality", {}).get("ambiguous_frame_percentage", 0.0),
                "quality_note": video_result.get("quality", {}).get("quality_note", "")
            }

            # Audio quality information
            fused_output["audio_quality_info"] = {
                "detected": embedded_audio_detected,
                "audio_present": audio_result.get("audio_present", False) if audio_result else False,
                "energy_db": audio_result.get("energy_db") if audio_result else None,
                "reliability": audio_result.get("audio_reliability", 0.0) if audio_result else 0.0,
                "confidence": audio_result.get("audio_confidence", 0.0) if audio_result else 0.0,
                "duration_seconds": audio_result.get("duration_seconds", 0.0) if audio_result else 0.0,
                "num_windows": audio_result.get("num_windows", 0) if audio_result else 0
            }

            self._send_json(fused_output)

        except VideoValidationError as ve:
            self._send_json({"error": str(ve), "type": "ValidationError"}, status=HTTPStatus.BAD_REQUEST)
        except (VideoReadError, AudioPipelineError) as re_err:
            self._send_json({"error": str(re_err), "type": "ReadError"}, status=HTTPStatus.UNPROCESSABLE_ENTITY)
        except InferenceError as ie:
            self._send_json({"error": str(ie), "type": "InferenceQualityError"}, status=HTTPStatus.UNPROCESSABLE_ENTITY)
        except Exception as e:
            self._send_json({"error": f"Internal multimodal pipeline error: {str(e)}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_analyze_multimodal_demo(self):
        """Standardized synthetic scenario validation endpoint (preserves 6 benchmarks)."""
        global FUSION_ENGINE
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8") or "{}")
            scenario_id = payload.get("scenario_id") or payload.get("clip_id")

            if not scenario_id:
                self._send_json({"error": "Missing 'scenario_id' parameter"}, status=HTTPStatus.BAD_REQUEST)
                return

            scenarios = build_synthetic_scenarios()
            matched = next((s for s in scenarios if s["clip_id"] == scenario_id), None)
            if not matched:
                self._send_json({"error": f"Demo scenario '{scenario_id}' not found"}, status=HTTPStatus.NOT_FOUND)
                return

            # Interactive demo execution: save_csv=False to preserve clean benchmark artifact integrity
            fused_output = FUSION_ENGINE.fuse_multimodal(
                video_result=matched["video"],
                audio_result=matched["audio"],
                clip_id=matched["clip_id"],
                save_csv=False
            )
            fused_output["scenario_description"] = matched["description"]
            fused_output["expected_outcome"] = matched["expected_outcome"]
            fused_output["input_summary"] = {
                "video_filename": f"{matched['clip_id']}.mp4 (Standardized Scenario)",
                "video_duration_seconds": 10.0,
                "embedded_audio_detected": matched["audio"] is not None and matched["audio"].get("audio_present", True),
                "screening_pipeline": "Standardized Multimodal Validation Scenario"
            }
            self._send_json(fused_output)

        except Exception as e:
            self._send_json({"error": f"Demo analysis error: {str(e)}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _serve_video_stream(self, path: str):
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
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else file_size - 1
                length = end - start + 1

                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                with open(target, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(target, "rb") as f:
            while chunk := f.read(65536):
                self.wfile.write(chunk)

    def _serve_file(self, file_path: Path, content_type: Optional[str] = None):
        if not file_path.exists() or not file_path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        if content_type is None:
            mime, _ = mimetypes.guess_type(str(file_path))
            content_type = mime or "application/octet-stream"

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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} - {format % args}")


def run_server(host: str = "127.0.0.1", port: int = 8000):
    global VIDEO_PIPELINE, AUDIO_PIPELINE, FUSION_ENGINE
    print("\n" + "=" * 70)
    print("ZERO RABIES-MMNET — MULTIMODAL BEHAVIORAL SCREENING SERVER")
    print("=" * 70)
    print("Workflow: Single Canine Video Input -> Auto-Extract Audio -> Dynamic Late Fusion")
    print("\nInitializing Frozen Video V2 Pipeline...")
    VIDEO_PIPELINE = VideoBehaviorPipeline()
    print(f"  Loaded Frozen Mamba S6 Checkpoint: {VIDEO_PIPELINE.mamba_classifier.checkpoint_path.name}")
    print(f"  Loaded YOLO Pose Checkpoint       : {VIDEO_PIPELINE.pose_estimator.checkpoint_path.name}")

    print("\nInitializing Frozen Audio V2 Pipeline...")
    AUDIO_PIPELINE = AudioBehaviorPipeline()
    print(f"  Loaded Frozen AudioNet v2 Checkpoint: {AUDIO_PIPELINE.model_path.name}")
    print(f"  Loaded Isotonic Calibration Curve   : {AUDIO_PIPELINE.calibration_path.name}")

    print("\nInitializing Dynamic Confidence Late Fusion Engine...")
    FUSION_ENGINE = DynamicLateFusionEngine()
    print("  Dynamic Late Fusion Engine Ready (Tick Resolution: 0.5s)")

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, MultimodalScreeningRequestHandler)
    print(f"\nMultimodal Server live at: http://{host}:{port}/")
    print("Screening Prototype: Non-invasive Behavioral Screening & Risk Assessment")
    print("Press Ctrl+C to terminate.")
    print("=" * 70 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero Rabies-MMNet Multimodal Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
