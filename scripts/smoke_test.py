"""
End-to-End Smoke Test for Zero Rabies-MMNet Multimodal Screening.
Verifies live HTTP server responses, frontend HTML availability, multimodal demo execution,
and ensures benchmark artifact integrity (strictly 6 clip results, strictly 49 segments).
"""

import urllib.request
import json
import csv
from pathlib import Path

SERVER_URL = "http://127.0.0.1:8000"

def main():
    print("=" * 65)
    print("ZERO RABIES-MMNET — END-TO-END APPLICATION SMOKE TEST")
    print("=" * 65)

    # 1. Test HTML
    with urllib.request.urlopen(f"{SERVER_URL}/") as resp:
        html = resp.read().decode("utf-8")
        assert "ZERO RABIES-MMNET" in html, "Missing brand title"
        assert "SYNTH_02_HIGH_AGITATION_CONCORDANT" in html, "Missing demo scenario button"
        assert "Fusion Details" in html, "Missing expandable details section"
        print(f"[OK] GET /: HTML document loaded and verified ({len(html)} bytes)")

    # 2. Test status endpoint
    with urllib.request.urlopen(f"{SERVER_URL}/api/status") as resp:
        status = json.loads(resp.read().decode("utf-8"))
        assert status["status"] == "ready", "Status not ready"
        assert len(status["demo_scenarios"]) == 6, f"Expected 6 scenarios, got {len(status['demo_scenarios'])}"
        print(f"[OK] GET /api/status: Pipeline ready with {len(status['demo_scenarios'])} demo scenarios")

    # 3. Test multimodal demo analysis
    payload = json.dumps({"scenario_id": "SYNTH_02_HIGH_AGITATION_CONCORDANT"}).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/api/analyze_multimodal_demo",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        fused_score = data["fused"]["risk_score"]
        risk_level = data["fused"]["risk_level"]
        segments_count = len(data["segments"])
        wv = data["weights"]["video"]
        wa = data["weights"]["audio"]

        assert fused_score == 86, f"Expected score 86, got {fused_score}"
        assert risk_level == "HIGH", f"Expected HIGH, got {risk_level}"
        assert segments_count == 10, f"Expected 10 segments, got {segments_count}"
        print(f"[OK] POST /api/analyze_multimodal_demo (SYNTH_02):")
        print(f"     • Fused Score : {fused_score} / 100")
        print(f"     • Risk Level  : {risk_level}")
        print(f"     • Weights     : Wv={wv}, Wa={wa}")
        print(f"     • Segments    : {segments_count} aligned ticks")

    # 4. Verify benchmark artifact integrity (no contamination)
    with open("outputs/fusion/dynamic_late_fusion_results.csv", "r", encoding="utf-8") as f:
        res_rows = list(csv.reader(f))[1:]
        clip_ids = [r[0] for r in res_rows]
        assert len(res_rows) == 6, f"Expected 6 rows in results CSV, got {len(res_rows)}"
        assert len(set(clip_ids)) == 6, f"Duplicates found in results CSV: {clip_ids}"
        print(f"[OK] Benchmark Results CSV: Strictly 6 unique rows (0 duplicates)")

    with open("outputs/fusion/dynamic_late_fusion_segments.csv", "r", encoding="utf-8") as f:
        seg_rows = list(csv.reader(f))[1:]
        assert len(seg_rows) == 49, f"Expected 49 segments in segments CSV, got {len(seg_rows)}"
        print(f"[OK] Benchmark Segments CSV: Strictly 49 segment rows")

    print("\n" + "=" * 65)
    print("ALL APPLICATION SMOKE TEST CHECKS PASSED SUCCESSFULLY")
    print("=" * 65)

if __name__ == "__main__":
    main()
