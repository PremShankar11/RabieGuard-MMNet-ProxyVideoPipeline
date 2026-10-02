/**
 * Zero Rabies-MMNet — Multimodal Behavioral Screening Frontend Client.
 *
 * Single-Video Input Workflow:
 * User uploads one canine video.
 * Embedded audio is inspected and automatically extracted on backend.
 * Dynamic Confidence-Aware Late Fusion runs Video V2 + Audio V2.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements: Input
  const videoInput = document.getElementById("video-input");
  const videoDropzone = document.getElementById("video-dropzone");
  const mediaInspectionCard = document.getElementById("media-inspection-card");
  const inspectStatusBadge = document.getElementById("inspection-status-badge");
  const inspectVideoName = document.getElementById("inspect-video-name");
  const inspectFileSize = document.getElementById("inspect-file-size");
  const inspectDuration = document.getElementById("inspect-duration");
  const inspectAudioTrack = document.getElementById("inspect-audio-track");
  const inspectAudioProc = document.getElementById("inspect-audio-proc");
  const inspectStatusMsg = document.getElementById("inspect-status-msg");

  const btnAnalyze = document.getElementById("btn-analyze");
  const btnReset = document.getElementById("btn-reset");

  // Progress & Error
  const progressCard = document.getElementById("progress-card");
  const progressBarFill = document.getElementById("progress-bar-fill");
  const progressStageTitle = document.getElementById("progress-stage-title");
  const progressStageDesc = document.getElementById("progress-stage-desc");
  const errorCard = document.getElementById("error-card");
  const errorTitle = document.getElementById("error-title");
  const errorMessage = document.getElementById("error-message");
  const btnDismissError = document.getElementById("btn-dismiss-error");

  // Results Section
  const resultsSection = document.getElementById("results-section");
  const scenarioBanner = document.getElementById("scenario-banner");
  const scenarioDescText = document.getElementById("scenario-desc-text");

  // Input Summary Banner
  const sumVideoFile = document.getElementById("sum-video-file");
  const sumVideoDuration = document.getElementById("sum-video-duration");
  const sumAudioDetected = document.getElementById("sum-audio-detected");
  const sumAudioProcessing = document.getElementById("sum-audio-processing");
  const sumPipelineBadge = document.getElementById("sum-pipeline-badge");
  const sumNoteText = document.getElementById("sum-note-text");

  // Modality Cards
  const resVideoScore = document.getElementById("res-video-score");
  const resVideoProb = document.getElementById("res-video-prob");
  const resVideoConf = document.getElementById("res-video-conf");
  const resVideoRel = document.getElementById("res-video-rel");
  const videoQualityTag = document.getElementById("video-quality-tag");

  const resAudioScore = document.getElementById("res-audio-score");
  const resAudioProb = document.getElementById("res-audio-prob");
  const resAudioConf = document.getElementById("res-audio-conf");
  const resAudioRel = document.getElementById("res-audio-rel");
  const resAudioOrigin = document.getElementById("res-audio-origin");
  const audioPresenceTag = document.getElementById("audio-presence-tag");

  const resWeightVideo = document.getElementById("res-weight-video");
  const resWeightAudio = document.getElementById("res-weight-audio");
  const resFusedScore = document.getElementById("res-fused-score");
  const resFusedProb = document.getElementById("res-fused-prob");
  const resRiskLevelBadge = document.getElementById("res-risk-level-badge");
  const resRiskBandTitle = document.getElementById("res-risk-band-title");
  const riskMeterFill = document.getElementById("risk-meter-fill");

  // Quality Assessment Card
  const qValidFrames = document.getElementById("q-valid-frames");
  const qMeanPoseConf = document.getElementById("q-mean-pose-conf");
  const qAmbiguityLevel = document.getElementById("q-ambiguity-level");
  const qAmbiguousPct = document.getElementById("q-ambiguous-pct");
  const qAudioStatus = document.getElementById("q-audio-status");
  const qAudioCodec = document.getElementById("q-audio-codec");
  const qAudioEnergy = document.getElementById("q-audio-energy");
  const qAudioReliability = document.getElementById("q-audio-reliability");
  const qNoteText = document.getElementById("q-note-text");

  // Baselines
  const baseVScore = document.getElementById("base-v-score");
  const baseVConf = document.getElementById("base-v-conf");
  const baseVBand = document.getElementById("base-v-band");

  const baseAScore = document.getElementById("base-a-score");
  const baseAConf = document.getElementById("base-a-conf");
  const baseABand = document.getElementById("base-a-band");

  const baseFusedScore = document.getElementById("base-fused-score");
  const baseFusedWeights = document.getElementById("base-fused-weights");
  const baseFusedBand = document.getElementById("base-fused-band");

  // Details
  const btnToggleDetails = document.getElementById("btn-toggle-details");
  const detailsChevron = document.getElementById("details-chevron");
  const fusionDetailsContent = document.getElementById("fusion-details-content");
  const segmentsTableBody = document.getElementById("segments-table-body");

  // Video preview
  const videoPreviewCard = document.getElementById("video-preview-card");
  const annotatedPlayer = document.getElementById("annotated-player");

  // Demo Buttons
  const demoButtons = document.querySelectorAll(".btn-pill[data-scenario]");

  // State
  let selectedVideoFile = null;

  // File Selection: Video
  videoDropzone.addEventListener("click", () => videoInput.click());
  videoInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleVideoSelected(e.target.files[0]);
    }
  });

  // Drag and Drop
  setupDragDrop(videoDropzone, (file) => handleVideoSelected(file));

  function setupDragDrop(el, onFileDrop) {
    ["dragenter", "dragover"].forEach((evt) => {
      el.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((evt) => {
      el.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.classList.remove("dragover");
      });
    });
    el.addEventListener("drop", (e) => {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        onFileDrop(e.dataTransfer.files[0]);
      }
    });
  }

  async function handleVideoSelected(file) {
    selectedVideoFile = file;
    btnAnalyze.disabled = false;

    // Show initial metadata while probing stream
    mediaInspectionCard.classList.remove("hidden");
    inspectStatusBadge.textContent = "Probing Streams...";
    inspectStatusBadge.className = "inspection-badge";
    inspectVideoName.textContent = file.name;
    inspectFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    inspectDuration.textContent = "Measuring...";
    inspectAudioTrack.textContent = "Checking...";
    inspectAudioProc.textContent = "16 kHz mono (auto-extract)";
    inspectStatusMsg.textContent = "Inspecting media container...";

    // Probe media stream on backend via FFmpeg
    try {
      const probeData = new FormData();
      probeData.append("video", file);

      const resp = await fetch("/api/probe_video", {
        method: "POST",
        body: probeData,
      });

      if (resp.ok) {
        const info = await resp.json();
        inspectStatusBadge.textContent = "Stream Verified";
        inspectDuration.textContent = info.duration_seconds ? `${info.duration_seconds}s` : "Container parsed";
        inspectAudioTrack.textContent = info.has_audio ? `Detected (${info.audio_codec || "audio"})` : "Not detected";
        inspectAudioProc.textContent = info.audio_processing;
        inspectStatusMsg.textContent = info.status;
      } else {
        inspectStatusBadge.textContent = "Ready";
        inspectAudioTrack.textContent = "Will inspect during execution";
        inspectStatusMsg.textContent = "Ready for screening";
      }
    } catch (err) {
      inspectStatusBadge.textContent = "Ready";
      inspectAudioTrack.textContent = "Auto-extract";
      inspectStatusMsg.textContent = "Ready for screening";
    }
  }

  // Reset Button
  btnReset.addEventListener("click", resetAll);

  function resetAll() {
    selectedVideoFile = null;
    videoInput.value = "";
    mediaInspectionCard.classList.add("hidden");
    btnAnalyze.disabled = true;
    progressCard.classList.add("hidden");
    errorCard.classList.add("hidden");
    resultsSection.classList.add("hidden");
    scenarioBanner.classList.add("hidden");
  }

  // Dismiss Error
  btnDismissError.addEventListener("click", () => {
    errorCard.classList.add("hidden");
  });

  // Toggle Details
  btnToggleDetails.addEventListener("click", () => {
    const isHidden = fusionDetailsContent.classList.contains("hidden");
    if (isHidden) {
      fusionDetailsContent.classList.remove("hidden");
      detailsChevron.classList.add("expanded");
    } else {
      fusionDetailsContent.classList.add("hidden");
      detailsChevron.classList.remove("expanded");
    }
  });

  // Demo Buttons (Synthetic Validation Scenarios)
  demoButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const scenarioId = btn.getAttribute("data-scenario");
      executeDemoScenario(scenarioId);
    });
  });

  async function executeDemoScenario(scenarioId) {
    hideError();
    resultsSection.classList.add("hidden");
    progressCard.classList.remove("hidden");
    simulateProgressSteps();

    try {
      const resp = await fetch("/api/analyze_multimodal_demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario_id: scenarioId }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: "Server Error" }));
        throw new Error(err.error || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      finishProgress();
      renderMultimodalResults(data);
    } catch (err) {
      progressCard.classList.add("hidden");
      showError("Demo Scenario Execution Error", err.message);
    }
  }

  // Analyze Action (Single Canine Video Upload)
  btnAnalyze.addEventListener("click", async () => {
    if (!selectedVideoFile) return;

    hideError();
    resultsSection.classList.add("hidden");
    progressCard.classList.remove("hidden");
    simulateProgressSteps();

    const formData = new FormData();
    formData.append("video", selectedVideoFile);

    try {
      const resp = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: "Pipeline Error" }));
        throw new Error(err.error || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      finishProgress();
      renderMultimodalResults(data);
    } catch (err) {
      progressCard.classList.add("hidden");
      showError("Inference Execution Failed", err.message);
    }
  });

  function renderMultimodalResults(data) {
    progressCard.classList.add("hidden");
    resultsSection.classList.remove("hidden");

    // Scenario banner if demo
    if (data.scenario_description) {
      scenarioBanner.classList.remove("hidden");
      scenarioDescText.textContent = `${data.scenario_description} — Expected: ${data.expected_outcome || ""}`;
    } else {
      scenarioBanner.classList.add("hidden");
    }

    // Input provenance summary
    const inp = data.input_summary || {};
    sumVideoFile.textContent = inp.video_filename || (selectedVideoFile ? selectedVideoFile.name : "Canine Video");
    sumVideoDuration.textContent = inp.video_duration_seconds ? `${inp.video_duration_seconds}s` : "--";
    sumAudioDetected.textContent = inp.embedded_audio_detected ? "Detected" : "Not Detected";
    sumAudioProcessing.textContent = inp.embedded_audio_detected ? "16 kHz mono (Extracted)" : "None (Video-Only)";
    sumPipelineBadge.textContent = inp.screening_pipeline || "Multimodal Screening";
    sumNoteText.textContent = inp.audio_extraction_note || "Multimodal screening complete.";

    // 1. Video Branch
    const v = data.video || {};
    const vq = data.video_quality_info || {};
    if (v.score !== null && v.score !== undefined) {
      resVideoScore.textContent = v.score;
      resVideoProb.textContent = `${Math.round((v.probability || 0) * 100)}%`;
      resVideoConf.textContent = `${Math.round((v.confidence || 0) * 100)}%`;
      const vRel = (data.baselines && data.baselines.video_only && data.baselines.video_only.confidence !== undefined)
        ? data.baselines.video_only.confidence
        : v.confidence;
      resVideoRel.textContent = typeof vRel === "number" ? vRel.toFixed(2) : "--";
      videoQualityTag.textContent = v.quality || "GOOD";
      videoQualityTag.className = `quality-badge ${(v.quality || "good").toLowerCase()}`;
    } else {
      resVideoScore.textContent = "N/A";
      resVideoProb.textContent = "N/A";
      resVideoConf.textContent = "0%";
      resVideoRel.textContent = "0.00";
      videoQualityTag.textContent = "NO INPUT";
      videoQualityTag.className = "quality-badge limited";
    }

    // 2. Audio Branch
    const a = data.audio || {};
    const aq = data.audio_quality_info || {};
    const hasAudioTrack = inp.embedded_audio_detected;

    if (a.score !== null && a.score !== undefined) {
      resAudioScore.textContent = a.score;
      resAudioProb.textContent = `${Math.round((a.probability || 0) * 100)}%`;
      resAudioConf.textContent = `${Math.round((a.confidence || 0) * 100)}%`;
      resAudioRel.textContent = aq.reliability !== undefined ? aq.reliability.toFixed(2) : "--";
      resAudioOrigin.textContent = "Embedded in Video";
      audioPresenceTag.textContent = "DETECTED";
      audioPresenceTag.className = "quality-badge";
    } else if (hasAudioTrack && aq.audio_present === false) {
      resAudioScore.textContent = "Silent";
      resAudioProb.textContent = "N/A";
      resAudioConf.textContent = "0%";
      resAudioRel.textContent = "0.00";
      resAudioOrigin.textContent = "Embedded (< -50 dBFS)";
      audioPresenceTag.textContent = "SILENT";
      audioPresenceTag.className = "quality-badge silent";
    } else {
      resAudioScore.textContent = "No Audio";
      resAudioProb.textContent = "N/A";
      resAudioConf.textContent = "0%";
      resAudioRel.textContent = "0.00";
      resAudioOrigin.textContent = "No Audio Track";
      audioPresenceTag.textContent = "NOT DETECTED";
      audioPresenceTag.className = "quality-badge limited";
    }

    // 3. Dynamic Late Fusion
    const w = data.weights || {};
    const f = data.fused || {};

    const wv = typeof w.video === "number" ? w.video : 1.0;
    const wa = typeof w.audio === "number" ? w.audio : 0.0;
    resWeightVideo.textContent = wv.toFixed(2);
    resWeightAudio.textContent = wa.toFixed(2);

    const fusedScore = f.risk_score !== undefined ? f.risk_score : 50;
    resFusedScore.textContent = fusedScore;
    const fusedProb = f.risk_probability !== undefined ? `${Math.round(f.risk_probability * 100)}%` : "--%";
    resFusedProb.textContent = fusedProb;

    const riskLevel = f.risk_level || "LOW";
    resRiskLevelBadge.textContent = riskLevel;
    resRiskLevelBadge.className = `risk-badge ${riskLevel.toLowerCase()}`;

    // Risk band title & meter
    let bandTitle = "Low Agitation / Calm-like";
    if (riskLevel === "MEDIUM") {
      bandTitle = "Moderate Activity / Transitional";
    } else if (riskLevel === "HIGH") {
      bandTitle = "High Agitation";
    }
    resRiskBandTitle.textContent = bandTitle;
    riskMeterFill.style.width = `${fusedScore}%`;

    // 4. Quality Assessment Card
    if (vq.valid_frame_percentage !== undefined) {
      qValidFrames.textContent = `${vq.valid_frame_percentage}%`;
      qMeanPoseConf.textContent = `Mean Pose Conf: ${(vq.mean_pose_confidence * 100).toFixed(1)}%`;
      qAmbiguityLevel.textContent = vq.ambiguity_level || "LOW";
      qAmbiguousPct.textContent = `Ambiguous: ${vq.ambiguous_frame_percentage || 0}%`;
    }
    qAudioStatus.textContent = inp.embedded_audio_detected ? "Detected" : "Not Detected";
    qAudioCodec.textContent = inp.audio_codec ? `Codec: ${inp.audio_codec}` : "Codec: N/A";
    qAudioEnergy.textContent = aq.energy_db !== null && aq.energy_db !== undefined ? `${aq.energy_db} dBFS` : "-- dBFS";
    qAudioReliability.textContent = `Reliability: ${aq.reliability !== undefined ? aq.reliability.toFixed(2) : "0.00"}`;
    qNoteText.textContent = vq.quality_note || inp.audio_extraction_note || "";

    // 5. Baseline Comparisons
    const b = data.baselines || {};
    const bv = b.video_only || {};
    const ba = b.audio_only || {};
    const bf = b.dynamic_late_fusion || {};

    baseVScore.textContent = bv.score !== null && bv.score !== undefined ? `${bv.score} / 100` : "N/A";
    baseVConf.textContent = bv.confidence !== undefined ? `${Math.round(bv.confidence * 100)}%` : "--";
    baseVBand.textContent = bv.risk_level || "N/A";

    baseAScore.textContent = ba.score !== null && ba.score !== undefined ? `${ba.score} / 100` : "N/A (No Audio)";
    baseAConf.textContent = ba.confidence !== undefined ? `${Math.round(ba.confidence * 100)}%` : "--";
    baseABand.textContent = ba.risk_level || "N/A";

    baseFusedScore.textContent = `${bf.risk_score || fusedScore} / 100`;
    baseFusedWeights.textContent = `Wv: ${wv.toFixed(2)} | Wa: ${wa.toFixed(2)}`;
    baseFusedBand.textContent = bf.risk_level || riskLevel;

    // 6. Populate Expandable Temporal Fusion Table
    segmentsTableBody.innerHTML = "";
    const segments = data.segments || [];
    segments.forEach((seg) => {
      const tr = document.createElement("tr");

      const tInt = `[${seg.start_time.toFixed(1)}s - ${seg.end_time.toFixed(1)}s]`;
      const vp = seg.video_probability !== null ? seg.video_probability.toFixed(2) : "-";
      const vc = `${Math.round(seg.video_confidence * 100)}%`;
      const vqBadge = seg.video_quality || "-";

      const ap = seg.audio_probability !== null ? seg.audio_probability.toFixed(2) : "-";
      const ac = `${Math.round(seg.audio_confidence * 100)}%`;
      const aqBadge = seg.audio_quality || "-";

      const segWv = seg.video_weight !== undefined ? seg.video_weight.toFixed(2) : "-";
      const segWa = seg.audio_weight !== undefined ? seg.audio_weight.toFixed(2) : "-";
      const fs = seg.fused_score !== null ? seg.fused_score : "-";
      const rl = seg.risk_level || "-";

      tr.innerHTML = `
        <td>${tInt}</td>
        <td>${vp}</td>
        <td>${vc}</td>
        <td><span class="quality-badge ${vqBadge.toLowerCase()}">${vqBadge}</span></td>
        <td>${ap}</td>
        <td>${ac}</td>
        <td><span class="quality-badge ${aqBadge.toLowerCase()}">${aqBadge}</span></td>
        <td>${segWv}</td>
        <td>${segWa}</td>
        <td><strong>${fs}</strong></td>
        <td><span class="risk-badge ${rl.toLowerCase()}">${rl}</span></td>
      `;
      segmentsTableBody.appendChild(tr);
    });

    // 7. Video player preview if present
    if (data.annotated_video_url) {
      videoPreviewCard.classList.remove("hidden");
      annotatedPlayer.src = data.annotated_video_url;
      annotatedPlayer.load();
    } else {
      videoPreviewCard.classList.add("hidden");
    }

    // Scroll smoothly to results
    resultsSection.scrollIntoView({ behavior: "smooth" });
  }

  function simulateProgressSteps() {
    progressBarFill.style.width = "20%";
    const steps = document.querySelectorAll(".pipeline-steps .step-item");
    steps.forEach((s) => s.classList.remove("active", "completed"));

    steps[0].classList.add("active");
    progressStageTitle.textContent = "1/5: Running Frozen Video V2 Pipeline...";
    progressStageDesc.textContent = "Mamba S6 & YOLO Dog Pose tracking (8 FPS, 16-frame windows)";

    setTimeout(() => {
      progressBarFill.style.width = "40%";
      steps[0].classList.remove("active");
      steps[0].classList.add("completed");
      steps[1].classList.add("active");
      progressStageTitle.textContent = "2/5: Extracting Embedded Audio Track...";
      progressStageDesc.textContent = "Extracting and resampling embedded audio to 16 kHz mono PCM";
    }, 800);

    setTimeout(() => {
      progressBarFill.style.width = "65%";
      steps[1].classList.remove("active");
      steps[1].classList.add("completed");
      steps[2].classList.add("active");
      progressStageTitle.textContent = "3/5: Running Frozen Audio V2 Pipeline...";
      progressStageDesc.textContent = "AudioNet v2 & Isotonic Calibration (64 mel bands, 3.0s windows)";
    }, 1600);

    setTimeout(() => {
      progressBarFill.style.width = "85%";
      steps[2].classList.remove("active");
      steps[2].classList.add("completed");
      steps[3].classList.add("active");
      progressStageTitle.textContent = "4/5: Synchronizing Streams on 0.5s Temporal Grid...";
      progressStageDesc.textContent = "Calculating reliability-modulated dynamic weights Wv, Wa";
    }, 2400);

    setTimeout(() => {
      progressBarFill.style.width = "95%";
      steps[3].classList.remove("active");
      steps[3].classList.add("completed");
      steps[4].classList.add("active");
      progressStageTitle.textContent = "5/5: Executing Dynamic Late Fusion...";
      progressStageDesc.textContent = "Computing Multimodal Behavioral Risk Score & Risk Band";
    }, 3200);
  }

  function finishProgress() {
    progressBarFill.style.width = "100%";
    const steps = document.querySelectorAll(".pipeline-steps .step-item");
    steps.forEach((s) => s.classList.add("completed"));
  }

  function showError(title, msg) {
    errorTitle.textContent = title;
    errorMessage.textContent = msg;
    errorCard.classList.remove("hidden");
    errorCard.scrollIntoView({ behavior: "smooth" });
  }

  function hideError() {
    errorCard.classList.add("hidden");
  }
});
