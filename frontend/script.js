
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

videoInput.addEventListener("change", () => {
    const file = videoInput.files[0];

    if (!file) {
        fileName.textContent = "No video selected";
        return;
    }

    fileName.textContent =
        `${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MB`;

    message.className = "message";
    message.textContent = "Video selected. Click Analyze video to start.";
});

analyzeBtn.addEventListener("click", analyzeVideo);

function setMessage(text, type = "") {
    message.textContent = text;
    message.className = `message ${type}`.trim();
}

function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    analyzeBtn.textContent = isLoading
        ? "Processing video... Please wait"
        : "Analyze video →";
}

function showStatus(status) {
    const normalized = String(status || "UNKNOWN").toUpperCase();

    statusCard.className = "status-card";
    reportBadge.className = "status-badge";
    statusTitle.textContent = "";
    statusMessage.textContent = "";

    if (normalized === "VIOLATION") {
        statusCard.classList.add("violation");
        reportBadge.classList.add("violation");
        reportBadge.textContent = "VIOLATION";
        statusTitle.textContent = "SAFETY VIOLATION DETECTED";
        statusMessage.textContent =
            "NoHelmet or NoVest was detected in at least one sampled frame. Review the video.";
    } else if (normalized === "SAFE") {
        statusCard.classList.add("safe");
        reportBadge.classList.add("safe");
        reportBadge.textContent = "PPE DETECTED";
        statusTitle.textContent = "PPE DETECTED";
        statusMessage.textContent =
            "Helmet or Vest was detected, and no violation was detected in the sampled frames.";
    } else {
        statusCard.classList.add("review");
        reportBadge.classList.add("review");
        reportBadge.textContent = "CHECK VIDEO";
        statusTitle.textContent = "PPE STATUS NOT CONFIRMED";
        statusMessage.textContent =
            "The model could not confirm PPE compliance. Please review the video.";
    }
}

function renderDetections(detections) {
    detectionList.replaceChildren();

    if (!Array.isArray(detections) || detections.length === 0) {
        detectionList.textContent = "No detections were returned by the model.";
        return;
    }

    const sorted = [...detections].sort((a, b) => {
        const frameA = Number(a.frame_number || 0);
        const frameB = Number(b.frame_number || 0);
        return frameA - frameB;
    });

    // Keep the report readable for long videos.
    const shown = sorted.slice(0, 100);

    for (const item of shown) {
        const name = String(item.class_name || item.class || "Unknown");
        const isViolation = ["NoHelmet", "NoVest"].includes(name);
        const isSafe = ["Helmet", "Vest"].includes(name);

        const card = document.createElement("div");
        card.className = "detection-item";

        if (isViolation) {
            card.classList.add("violation");
        } else if (isSafe) {
            card.classList.add("safe");
        } else {
            card.classList.add("review");
        }

        const left = document.createElement("span");
        const frame = item.frame_number
            ? `Frame ${item.frame_number}`
            : "Detected object";
        const seconds = Number(item.timestamp_seconds || 0).toFixed(2);

        left.textContent = `${frame} · ${seconds}s · ${name}`;

        const right = document.createElement("strong");
        const confidence = Number(item.confidence || 0);
        right.textContent = `${(confidence * 100).toFixed(1)}%`;

        card.append(left, right);
        detectionList.appendChild(card);
    }

    if (sorted.length > shown.length) {
        const note = document.createElement("p");
        note.className = "muted small";
        note.textContent =
            `Showing ${shown.length} of ${sorted.length} detections.`;
        detectionList.appendChild(note);
    }
}

async function analyzeVideo() {
    const file = videoInput.files[0];

    if (!file) {
        setMessage("Please select a video first.", "error");
        return;
    }

    const extension = "." + file.name.split(".").pop().toLowerCase();

    if (!ALLOWED_EXTENSIONS.includes(extension)) {
        setMessage("Unsupported video format. Select MP4, AVI, MOV, MKV or WEBM.", "error");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setMessage("Uploading and processing your video. Please wait...");

    emptyState.hidden = true;
    reportContent.hidden = true;

    // Clear the previous processed video.
    processedVideo.pause();
    processedVideo.removeAttribute("src");
    processedVideo.load();
    downloadBtn.hidden = true;

    try {
        const response = await fetch("/api/analyze/video", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || `Server error: ${response.status}`
            );
        }

        reportContent.hidden = false;

        showStatus(data.overall_status);

        detectionCount.textContent = Number(
            data.detection_count || 0
        ).toLocaleString();

        framesProcessed.textContent = Number(
            data.frames_processed || 0
        ).toLocaleString();

        noHelmetCount.textContent = Number(
            data.no_helmet_count || 0
        ).toLocaleString();

        noVestCount.textContent = Number(
            data.no_vest_count || 0
        ).toLocaleString();

        renderDetections(data.detections);

        if (!data.output_url) {
            throw new Error("The server did not return a processed video URL.");
        }

        const videoUrl = data.output_url +
            (data.output_url.includes("?") ? "&" : "?") +
            "t=" + Date.now();

        processedVideo.src = videoUrl;
        processedVideo.load();

        downloadBtn.href = data.output_url;
        downloadBtn.download = data.output_file || "processed_video.mp4";
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
