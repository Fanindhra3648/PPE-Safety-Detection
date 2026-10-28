const videoInput = document.getElementById("videoFile");
const fileName = document.getElementById("fileName");
const analyzeBtn = document.getElementById("analyzeBtn");
const message = document.getElementById("message");

const emptyState = document.getElementById("emptyState");
const reportContent = document.getElementById("reportContent");
const reportBadge = document.getElementById("reportBadge");
const statusCard = document.getElementById("statusCard");
const statusTitle = document.getElementById("statusTitle");
const statusMessage = document.getElementById("statusMessage");

const detectionCount = document.getElementById("detectionCount");
const framesProcessed = document.getElementById("framesProcessed");
const noHelmetCount = document.getElementById("noHelmetCount");
const noVestCount = document.getElementById("noVestCount");

const processedVideo = document.getElementById("processedVideo");
const downloadBtn = document.getElementById("downloadBtn");
const detectionList = document.getElementById("detectionList");

const ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
const MAX_UPLOAD_MB = 50;
const POLL_INTERVAL_MS = 3000;
const MAX_WAIT_MS = 15 * 60 * 1000;

videoInput.addEventListener("change", () => {
    const file = videoInput.files[0];

    fileName.textContent = file
        ? `${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MB`
        : "No video selected";

    message.className = "message";

    if (file) {
        message.textContent = "Video selected. Click Analyze video to start.";
    }
});

analyzeBtn.addEventListener("click", analyzeVideo);

function setMessage(text, type = "") {
    message.textContent = text;
    message.className = `message ${type}`.trim();
}

function setLoading(loading) {
    analyzeBtn.disabled = loading;
    analyzeBtn.textContent = loading
        ? "Processing video... Please wait"
        : "Analyze video →";
}

async function readJsonResponse(response) {
    const text = await response.text();

    let data;

    try {
        data = JSON.parse(text);
    } catch {
        console.error("Non-JSON server response:", text.slice(0, 500));

        throw new Error(
            `Server returned a non-JSON response (HTTP ${response.status}). Check Render Logs.`
        );
    }

    if (!response.ok) {
        throw new Error(
            data.detail || `Server error: HTTP ${response.status}`
        );
    }

    return data;
}

function showStatus(status) {
    const value = String(status || "UNKNOWN").toUpperCase();

    statusCard.className = "status-card";
    reportBadge.className = "status-badge";
    statusTitle.textContent = "";
    statusMessage.textContent = "";

    if (value === "VIOLATION") {
        statusCard.classList.add("violation");
        reportBadge.classList.add("violation");
        reportBadge.textContent = "VIOLATION";
        statusTitle.textContent = "SAFETY VIOLATION DETECTED";
        statusMessage.textContent =
            "NoHelmet or NoVest was detected in at least one sampled frame.";
    } else if (value === "SAFE") {
        statusCard.classList.add("safe");
        reportBadge.classList.add("safe");
        reportBadge.textContent = "PPE DETECTED";
        statusTitle.textContent = "PPE DETECTED";
        statusMessage.textContent =
            "Helmet or Vest was detected in the sampled frames.";
    } else {
        statusCard.classList.add("review");
        reportBadge.classList.add("review");
        reportBadge.textContent = "CHECK VIDEO";
        statusTitle.textContent = "PPE STATUS NOT CONFIRMED";
        statusMessage.textContent =
            "The model could not confirm PPE compliance.";
    }
}

function renderDetections(detections) {
    detectionList.replaceChildren();

    if (!Array.isArray(detections) || detections.length === 0) {
        detectionList.textContent =
            "No detections were returned by the model.";
        return;
    }

    const shown = detections.slice(0, 100);

    for (const item of shown) {
        const name = String(item.class_name || item.class || "Unknown");

        const type = ["NoHelmet", "NoVest"].includes(name)
            ? "violation"
            : (["Helmet", "Vest"].includes(name) ? "safe" : "review");

        const card = document.createElement("div");
        card.className = `detection-item ${type}`;

        const left = document.createElement("span");

        left.textContent =
            `${item.frame_number ? `Frame ${item.frame_number}` : "Detected object"} · ` +
            `${Number(item.timestamp_seconds || 0).toFixed(2)}s · ${name}`;

        const right = document.createElement("strong");

        right.textContent =
            `${(Number(item.confidence || 0) * 100).toFixed(1)}%`;

        card.append(left, right);
        detectionList.appendChild(card);
    }

    if (detections.length > shown.length) {
        const note = document.createElement("p");
        note.className = "muted small";
        note.textContent =
            `Showing ${shown.length} of ${detections.length} detections.`;
        detectionList.appendChild(note);
    }
}

async function waitForVideoJob(jobId) {
    const startTime = Date.now();

    while (Date.now() - startTime < MAX_WAIT_MS) {
        await new Promise(resolve =>
            setTimeout(resolve, POLL_INTERVAL_MS)
        );

        const response = await fetch(
            `/api/analyze/video/${encodeURIComponent(jobId)}`,
            { cache: "no-store" }
        );

        const data = await readJsonResponse(response);

        if (data.status === "completed") {
            return data.result;
        }

        if (data.status === "failed") {
            throw new Error(data.error || "Video processing failed.");
        }

        setMessage(
            data.status === "queued"
                ? "Video is queued. Please wait..."
                : "Video is processing on the server. Please wait..."
        );
    }

    throw new Error(
        "Processing is taking too long. Check Render Logs before retrying."
    );
}

async function analyzeVideo() {
    const file = videoInput.files[0];

    if (!file) {
        setMessage("Please select a video first.", "error");
        return;
    }

    const extension = "." + file.name.split(".").pop().toLowerCase();

    if (!ALLOWED_EXTENSIONS.includes(extension)) {
        setMessage(
            "Unsupported format. Select MP4, AVI, MOV, MKV or WEBM.",
            "error"
        );
        return;
    }

    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
        setMessage(
            `Please select a video smaller than ${MAX_UPLOAD_MB} MB.`,
            "error"
        );
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setMessage("Uploading video. Please wait...");

    emptyState.hidden = true;
    reportContent.hidden = true;

    processedVideo.pause();
    processedVideo.removeAttribute("src");
    processedVideo.load();
    downloadBtn.hidden = true;

    try {
        const uploadResponse = await fetch("/api/analyze/video", {
            method: "POST",
            body: formData
        });

        const job = await readJsonResponse(uploadResponse);

        if (!job.job_id) {
            throw new Error("The server did not return a video job ID.");
        }

        setMessage("Video uploaded. Processing has started...");

        const result = await waitForVideoJob(job.job_id);

        reportContent.hidden = false;

        showStatus(result.overall_status);

        detectionCount.textContent =
            Number(result.detection_count || 0).toLocaleString();

        framesProcessed.textContent =
            Number(result.frames_processed || 0).toLocaleString();

        noHelmetCount.textContent =
            Number(result.no_helmet_count || 0).toLocaleString();

        noVestCount.textContent =
            Number(result.no_vest_count || 0).toLocaleString();

        renderDetections(result.detections);

        if (!result.output_url) {
            throw new Error("The server did not return an output video URL.");
        }

        const videoUrl = result.output_url +
            (result.output_url.includes("?") ? "&" : "?") +
            "t=" + Date.now();

        processedVideo.src = videoUrl;
        processedVideo.load();

        downloadBtn.href = videoUrl;
        downloadBtn.download = result.output_file || "processed_video.mp4";
        downloadBtn.hidden = false;

        setMessage(
            "Video analysis completed. Play the processed video to review the labels.",
            "success"
        );

    } catch (error) {
        console.error("Video analysis error:", error);

        setMessage(
            "Analysis failed: " + (error.message || "Unknown error"),
            "error"
        );

        emptyState.hidden = false;
        reportContent.hidden = true;

    } finally {
        setLoading(false);
    }
}