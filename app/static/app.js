/**
 * Zero Rabies-MMNet — Video V2 Web Application Client Logic.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropzone = document.getElementById("video-dropzone");
  const fileInput = document.getElementById("video-input");
  const filePreview = document.getElementById("file-preview-card");
  const fileNameDisplay = document.getElementById("selected-file-name");
  const fileSizeDisplay = document.getElementById("selected-file-size");
  const btnClearFile = document.getElementById("btn-clear-file");
  const btnAnalyze = document.getElementById("btn-analyze");
  const demoButtons = document.querySelectorAll(".demo-buttons button");

  const uploadSection = document.getElementById("upload-section");
  const progressCard = document.getElementById("progress-card");
  const errorCard = document.getElementById("error-card");
  const errorTitle = document.getElementById("error-title");
  const errorMessage = document.getElementById("error-message");
  const btnDismissError = document.getElementById("btn-dismiss-error");

  const resultsSection = document.getElementById("results-section");
  const warningBanner = document.getElementById("warning-banner");
  const warningText = document.getElementById("warning-text");

  // Result fields
  const resBehavior = document.getElementById("res-behavior");
  const resScoreNum = document.getElementById("res-score-num");
  const resProb = document.getElementById("res-prob");
  const resBandDetail = document.getElementById("res-band-detail");
  const resScoreBand = document.getElementById("res-score-band");
  const resInterpretation = document.getElementById("res-interpretation");
  const meterFill = document.getElementById("meter-fill");
  const resultCard = document.querySelector(".result-card");

  // Quality fields
  const resQualityBadge = document.getElementById("res-quality-badge");
  const resValidPct = document.getElementById("res-valid-pct");
  const resValidCount = document.getElementById("res-valid-count");
  const resPoseConf = document.getElementById("res-pose-conf");
  const resAmbLevel = document.getElementById("res-amb-level");
  const resAmbPct = document.getElementById("res-amb-pct");
  const resNumWindows = document.getElementById("res-num-windows");
  const resPaddedInfo = document.getElementById("res-padded-info");
  const resQualityNote = document.getElementById("res-quality-note");

  // Video Preview players
  const playerAnnotated = document.getElementById("player-annotated");
  const playerOriginal = document.getElementById("player-original");
  const tabAnnotated = document.getElementById("tab-annotated");
  const tabFrames = document.getElementById("tab-frames");
  const tabOriginal = document.getElementById("tab-original");

  // Frame Inspector elements
  const frameInspectorView = document.getElementById("frame-inspector-view");
  const frameImgDisplay = document.getElementById("frame-img-display");
  const frameSlider = document.getElementById("frame-slider");
  const frameCounterLabel = document.getElementById("frame-counter-label");
  const frameTsLabel = document.getElementById("frame-ts-label");
  const btnPlayFrames = document.getElementById("btn-play-frames");
  const btnPrevFrame = document.getElementById("btn-prev-frame");
  const btnNextFrame = document.getElementById("btn-next-frame");

  let previewFramesList = [];
  let currentFrameIdx = 0;
  let framePlayInterval = null;

  // Technical details
  const toggleTechDetails = document.getElementById("toggle-tech-details");
  const techDetailsContent = document.getElementById("tech-details-content");
  const tArch = document.getElementById("t-arch");
  const tParams = document.getElementById("t-params");
  const tPose = document.getElementById("t-pose");
  const tFps = document.getElementById("t-fps");
  const tWindow = document.getElementById("t-window");
  const tFeatures = document.getElementById("t-features");
  const tTracking = document.getElementById("t-tracking");
  const tCkpt = document.getElementById("t-ckpt");
  const tDevice = document.getElementById("t-device");
  const windowChips = document.getElementById("window-probabilities-chips");
  const btnDownloadJson = document.getElementById("btn-download-json");
  const btnViewManifest = document.getElementById("btn-view-manifest");
  const btnAnalyzeAnother = document.getElementById("btn-analyze-another");

  // Modal
  const manifestModal = document.getElementById("manifest-modal");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const manifestPre = document.getElementById("manifest-pre");

  // State
  let selectedFile = null;
  let selectedDemoClip = null;
  let lastInferenceResult = null;
  let progressInterval = null;

  // Initialize server status on load
  fetchStatus();

  function fetchStatus() {
    fetch("/api/status")
      .then(res => res.json())
      .then(data => {
        if (data.device) {
          const deviceBadge = document.getElementById("badge-device");
          if (deviceBadge) deviceBadge.textContent = `Device: ${data.device.toUpperCase()}`;
        }
      })
      .catch(err => console.warn("Status ping warning:", err));
  }

  // Drag and Drop Events
  ["dragenter", "dragover"].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  dropzone.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    selectedDemoClip = null;

    // Reset demo button selection styles
    demoButtons.forEach(btn => btn.classList.remove("active"));

    fileNameDisplay.textContent = file.name;
    const mbSize = (file.size / (1024 * 1024)).toFixed(2);
    fileSizeDisplay.textContent = `${mbSize} MB`;

    dropzone.classList.add("hidden");
    filePreview.classList.remove("hidden");
    btnAnalyze.disabled = false;
    hideError();
  }

  btnClearFile.addEventListener("click", () => {
    resetSelection();
  });

  function resetSelection() {
    selectedFile = null;
    selectedDemoClip = null;
    fileInput.value = "";
    filePreview.classList.add("hidden");
    dropzone.classList.remove("hidden");
    demoButtons.forEach(btn => btn.classList.remove("active"));
    btnAnalyze.disabled = true;
  }

  // Quick Demo Clips
  demoButtons.forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const clipId = btn.getAttribute("data-clip");
      selectedDemoClip = clipId;
      selectedFile = null;

      demoButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      // Update file preview card for demo
      fileNameDisplay.textContent = `Demo Clip: ${clipId}.mp4`;
      fileSizeDisplay.textContent = "Validated Development Clip";
      dropzone.classList.add("hidden");
      filePreview.classList.remove("hidden");
      btnAnalyze.disabled = false;
      hideError();
    });
  });

  // Analyze Action
  btnAnalyze.addEventListener("click", () => {
    if (!selectedFile && !selectedDemoClip) return;
    startInference();
  });

  function startInference() {
    hideError();
    uploadSection.classList.add("hidden");
    resultsSection.classList.add("hidden");
    progressCard.classList.remove("hidden");

    startSimulatedProgress();

    if (selectedDemoClip) {
      // Analyze demo
      fetch("/api/analyze_demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clip_id: selectedDemoClip })
      })
        .then(handleApiResponse)
        .catch(handleApiError);
    } else {
      // Analyze uploaded file
      const formData = new FormData();
      formData.append("video", selectedFile);

      fetch("/api/analyze", {
        method: "POST",
        body: formData
      })
        .then(handleApiResponse)
        .catch(handleApiError);
    }
  }

  function handleApiResponse(response) {
    return response.json().then(data => {
      if (!response.ok) {
        throw new Error(data.error || `Server returned error (${response.status})`);
      }
      stopProgress();
      displayResults(data);
    });
  }

  function handleApiError(err) {
    stopProgress();
    progressCard.classList.add("hidden");
    uploadSection.classList.remove("hidden");
    showError("Inference Failed", err.message || "An unexpected error occurred during video processing.");
  }

  function startSimulatedProgress() {
    const steps = [
      { id: "step-1", fill: "20%", title: "Validating Video...", desc: "Checking format and decoding container" },
      { id: "step-2", fill: "40%", title: "Extracting Frames at 8.0 FPS...", desc: "Uniform temporal sampling" },
      { id: "step-3", fill: "60%", title: "Detecting Canine Pose...", desc: "YOLO11n-Pose 24 keypoint inference" },
      { id: "step-4", fill: "80%", title: "Tracking Canine Identity...", desc: "Temporal continuity and normalization" },
      { id: "step-5", fill: "95%", title: "Evaluating Mamba S6 Dynamics...", desc: "Scoring 16-frame temporal windows" },
    ];

    let current = 0;
    const fillEl = document.getElementById("progress-bar-fill");
    const titleEl = document.getElementById("progress-stage-title");
    const descEl = document.getElementById("progress-stage-desc");

    // Reset steps
    for (let i = 1; i <= 5; i++) {
      const el = document.getElementById(`step-${i}`);
      if (el) el.className = "step-item";
    }

    function advance() {
      if (current < steps.length) {
        const s = steps[current];
        fillEl.style.width = s.fill;
        titleEl.textContent = s.title;
        descEl.textContent = s.desc;

        for (let i = 0; i < current; i++) {
          const prev = document.getElementById(steps[i].id);
          if (prev) prev.className = "step-item completed";
        }
        const active = document.getElementById(s.id);
        if (active) active.className = "step-item active";

        current++;
      }
    }

    advance();
    progressInterval = setInterval(advance, 800);
  }

  function stopProgress() {
    if (progressInterval) {
      clearInterval(progressInterval);
      progressInterval = null;
    }
  }

  function displayResults(data) {
    lastInferenceResult = data;
    progressCard.classList.add("hidden");
    resultsSection.classList.remove("hidden");

    // Behavioral state & score
    const isAgitated = data.behavior === "AGITATED";
    resBehavior.textContent = data.behavior;
    resScoreNum.textContent = data.behavioral_score;
    resProb.textContent = `${Math.round(data.probability_agitated * 100)}%`;
    resBandDetail.textContent = data.score_band;
    resScoreBand.textContent = data.score_band;
    resInterpretation.textContent = data.interpretation;

    // Styling according to state
    resultCard.classList.remove("state-calm", "state-agitated");
    if (isAgitated) {
      resultCard.classList.add("state-agitated");
    } else {
      resultCard.classList.add("state-calm");
    }

    // Meter bar fill
    meterFill.style.width = `${Math.max(3, Math.min(100, data.behavioral_score))}%`;

    // Quality indicators
    const q = data.quality;
    resQualityBadge.textContent = q.overall;
    resQualityBadge.className = `badge-quality ${q.overall}`;
    resValidPct.textContent = `${q.valid_frame_percentage}%`;
    resValidCount.textContent = `${data.video.valid_pose_frames} of ${data.video.sampled_frames} frames`;
    resPoseConf.textContent = q.mean_pose_confidence.toFixed(2);
    resAmbLevel.textContent = q.ambiguity_level;
    resAmbPct.textContent = `${q.ambiguous_frame_percentage}% ambiguous`;
    resNumWindows.textContent = data.temporal.num_windows;
    resPaddedInfo.textContent = `${data.temporal.padded_windows} padded windows`;
    resQualityNote.textContent = q.quality_note;

    // Warning Banner
    if (data.warning) {
      warningText.textContent = data.warning;
      warningBanner.classList.remove("hidden");
    } else {
      warningBanner.classList.add("hidden");
    }

    // Video & Frame Previews
    if (data.annotated_video_url) {
      playerAnnotated.src = data.annotated_video_url;
      playerAnnotated.load();
      playerAnnotated.play().catch(e => console.log("Autoplay prevented:", e));
    }
    if (data.original_video_url) {
      playerOriginal.src = data.original_video_url;
      playerOriginal.load();
    }

    // Initialize Frame-by-Frame Inspector if preview frames are returned
    previewFramesList = data.preview_frames || [];
    if (previewFramesList.length > 0) {
      frameSlider.min = 0;
      frameSlider.max = previewFramesList.length - 1;
      frameSlider.value = 0;
      setFrame(0);
      tabFrames.classList.remove("hidden");
    } else {
      tabFrames.classList.add("hidden");
    }
    showTab("annotated");

    // Technical details
    tArch.textContent = `${data.model.architecture} (2 layers, d_model=64, d_state=16)`;
    tParams.textContent = data.model.trainable_parameters.toLocaleString();
    tPose.textContent = `${data.model.pose_checkpoint} (24 Canine Keypoints)`;
    tFps.textContent = `${data.video.sampled_fps} FPS (Uniform Resampling)`;
    tWindow.textContent = `${data.temporal.window_length} frames (~2.0s, stride ${data.temporal.stride})`;
    tFeatures.textContent = "75 in_features (72 normalized pose + 3 reliability)";
    tTracking.textContent = "Temporal Consistency (IoU & Center Gated)";
    tCkpt.textContent = data.model.checkpoint;
    tDevice.textContent = data.model.device;

    // Window probabilities list
    windowChips.innerHTML = "";
    if (data.temporal.window_probabilities) {
      data.temporal.window_probabilities.forEach((p, idx) => {
        const chip = document.createElement("span");
        chip.className = "win-chip";
        chip.textContent = `W${idx + 1}: ${(p * 100).toFixed(1)}%`;
        windowChips.appendChild(chip);
      });
    }

    // Scroll to results smoothly
    resultsSection.scrollIntoView({ behavior: "smooth" });
  }

  // Frame Scrubber Functions
  function setFrame(idx) {
    if (!previewFramesList || previewFramesList.length === 0) return;
    currentFrameIdx = Math.max(0, Math.min(previewFramesList.length - 1, idx));
    const f = previewFramesList[currentFrameIdx];
    frameImgDisplay.src = f.image;
    frameSlider.value = currentFrameIdx;
    frameCounterLabel.textContent = `Frame ${currentFrameIdx + 1} / ${previewFramesList.length}`;
    frameTsLabel.textContent = `(${f.timestamp.toFixed(2)}s)`;
  }

  function toggleFramePlay() {
    if (framePlayInterval) {
      clearInterval(framePlayInterval);
      framePlayInterval = null;
      btnPlayFrames.textContent = "Play";
    } else {
      btnPlayFrames.textContent = "Pause";
      framePlayInterval = setInterval(() => {
        let next = currentFrameIdx + 1;
        if (next >= previewFramesList.length) next = 0;
        setFrame(next);
      }, 125); // 8 FPS
    }
  }

  frameSlider.addEventListener("input", (e) => {
    if (framePlayInterval) toggleFramePlay();
    setFrame(parseInt(e.target.value));
  });

  btnPrevFrame.addEventListener("click", () => {
    if (framePlayInterval) toggleFramePlay();
    setFrame(currentFrameIdx - 1);
  });

  btnNextFrame.addEventListener("click", () => {
    if (framePlayInterval) toggleFramePlay();
    setFrame(currentFrameIdx + 1);
  });

  btnPlayFrames.addEventListener("click", () => {
    toggleFramePlay();
  });

  // Tab Switching
  function showTab(tabName) {
    tabAnnotated.classList.remove("active");
    tabFrames.classList.remove("active");
    tabOriginal.classList.remove("active");
    playerAnnotated.classList.add("hidden");
    frameInspectorView.classList.add("hidden");
    playerOriginal.classList.add("hidden");

    if (framePlayInterval && tabName !== "frames") {
      toggleFramePlay();
    }

    if (tabName === "annotated") {
      tabAnnotated.classList.add("active");
      playerAnnotated.classList.remove("hidden");
      playerAnnotated.play().catch(() => {});
    } else if (tabName === "frames") {
      tabFrames.classList.add("active");
      frameInspectorView.classList.remove("hidden");
    } else if (tabName === "original") {
      tabOriginal.classList.add("active");
      playerOriginal.classList.remove("hidden");
      playerOriginal.play().catch(() => {});
    }
  }

  tabAnnotated.addEventListener("click", () => showTab("annotated"));
  tabFrames.addEventListener("click", () => showTab("frames"));
  tabOriginal.addEventListener("click", () => showTab("original"));

  // Automatic fallback if browser cannot decode video track
  playerAnnotated.addEventListener("error", (e) => {
    console.warn("Annotated video element error, falling back to Frame Inspector:", e);
    if (previewFramesList && previewFramesList.length > 0) {
      showTab("frames");
    }
  });

  // Accordion Toggle
  toggleTechDetails.addEventListener("click", () => {
    toggleTechDetails.classList.toggle("expanded");
    techDetailsContent.classList.toggle("hidden");
  });

  // Export JSON
  btnDownloadJson.addEventListener("click", () => {
    if (!lastInferenceResult) return;
    const str = JSON.stringify(lastInferenceResult, null, 2);
    const blob = new Blob([str], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `inference_${lastInferenceResult.video.filename.replace(/\.[^/.]+$/, "")}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // View Manifest Modal
  btnViewManifest.addEventListener("click", () => {
    manifestModal.classList.remove("hidden");
    fetch("/api/manifest")
      .then(res => res.json())
      .then(data => {
        manifestPre.textContent = JSON.stringify(data, null, 2);
      })
      .catch(err => {
        manifestPre.textContent = "Error loading manifest: " + err.message;
      });
  });

  btnCloseModal.addEventListener("click", () => {
    manifestModal.classList.add("hidden");
  });

  manifestModal.addEventListener("click", (e) => {
    if (e.target === manifestModal) {
      manifestModal.classList.add("hidden");
    }
  });

  // Analyze Another Video
  btnAnalyzeAnother.addEventListener("click", () => {
    resultsSection.classList.add("hidden");
    uploadSection.classList.remove("hidden");
    resetSelection();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // Error Card Helper
  function showError(title, msg) {
    errorTitle.textContent = title;
    errorMessage.textContent = msg;
    errorCard.classList.remove("hidden");
  }

  function hideError() {
    errorCard.classList.add("hidden");
  }

  btnDismissError.addEventListener("click", hideError);
});
