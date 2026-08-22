// Frontend Application Logic
document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const previewCard = document.getElementById("file-preview-card");
  const previewThumb = document.getElementById("preview-thumbnail");
  const previewFilename = document.getElementById("preview-filename");
  const previewFilesize = document.getElementById("preview-filesize");
  const removeFileBtn = document.getElementById("remove-file-btn");
  const dropZoneContent = document.querySelector(".drop-zone-content");
  
  const analyzeBtn = document.getElementById("analyze-btn");
  const btnText = document.querySelector(".btn-text");
  const btnSpinner = document.querySelector(".btn-spinner");
  
  const confSlider = document.getElementById("conf-slider");
  const confVal = document.getElementById("conf-val");
  const iouSlider = document.getElementById("iou-slider");
  const iouVal = document.getElementById("iou-val");

  const metricCount = document.getElementById("metric-count");
  const metricSeverity = document.getElementById("metric-severity");
  const metricArea = document.getElementById("metric-area");
  const metricLatency = document.getElementById("metric-latency");

  const emptyState = document.getElementById("empty-state");
  const annotatedImg = document.getElementById("annotated-image");
  const tableBody = document.getElementById("defects-table-body");
  const jsonOutput = document.getElementById("json-output");
  const apiStatusText = document.getElementById("api-status-text");

  let selectedFile = null;

  // Check API health status
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      if (data.status === "healthy") {
        apiStatusText.textContent = "System Ready (Model Cached)";
      } else {
        apiStatusText.textContent = "System Initializing...";
      }
    } catch {
      apiStatusText.textContent = "API Offline";
    }
  }
  checkHealth();

  // Slider Listeners
  confSlider.addEventListener("input", (e) => {
    confVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  iouSlider.addEventListener("input", (e) => {
    iouVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // Browse & Drop Handling
  browseBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  function handleFile(file) {
    if (!file) return;

    if (!file.type.match(/^image\/(jpeg|png|webp)$/)) {
      alert("Unsupported file type. Please upload a JPG, PNG, or WEBP image.");
      return;
    }

    selectedFile = file;
    previewFilename.textContent = file.name;
    previewFilesize.textContent = `${(file.size / 1024).toFixed(1)} KB`;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewThumb.src = e.target.result;
    };
    reader.readAsDataURL(file);

    dropZoneContent.style.display = "none";
    previewCard.style.display = "flex";
    analyzeBtn.disabled = false;
  }

  removeFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = "";
    previewCard.style.display = "none";
    dropZoneContent.style.display = "block";
    analyzeBtn.disabled = true;
  });

  // Analyze Defect
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    btnText.textContent = "Inspecting...";
    btnSpinner.style.display = "inline-block";

    const formData = new FormData();
    formData.append("file", selectedFile);

    const conf = parseFloat(confSlider.value);
    const iou = parseFloat(iouSlider.value);
    const url = `/predict?conf_threshold=${conf}&iou_threshold=${iou}&include_image=true`;

    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server error: ${response.statusText}`);
      }

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      alert(`Defect Analysis Failed: ${err.message}`);
    } finally {
      analyzeBtn.disabled = false;
      btnText.textContent = "⚡ Analyze Surface Defect";
      btnSpinner.style.display = "none";
    }
  });

  function renderResults(data) {
    // 1. Metrics
    metricCount.textContent = data.defect_count;
    metricSeverity.textContent = data.overall_severity.toUpperCase();
    metricSeverity.className = `metric-value badge badge-${data.overall_severity.toLowerCase()}`;
    metricArea.textContent = `${data.total_affected_area_percent.toFixed(2)}%`;
    metricLatency.textContent = `${data.inference_time_ms} ms`;

    // 2. Image
    if (data.annotated_image_base64) {
      annotatedImg.src = `data:image/jpeg;base64,${data.annotated_image_base64}`;
      annotatedImg.style.display = "block";
      emptyState.style.display = "none";
    }

    // 3. Table
    tableBody.innerHTML = "";
    if (data.detections.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="5" class="table-empty">No surface defects detected at current threshold.</td></tr>`;
    } else {
      data.detections.forEach((d) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${d.class_name}</strong></td>
          <td>${(d.confidence * 100).toFixed(1)}%</td>
          <td><span class="badge-${d.severity}">${d.severity.toUpperCase()}</span></td>
          <td>${d.area_percentage.toFixed(2)}% (${d.area_pixels} px)</td>
          <td>[${d.bbox.x1}, ${d.bbox.y1}, ${d.bbox.x2}, ${d.bbox.y2}]</td>
        `;
        tableBody.appendChild(tr);
      });
    }

    // 4. Raw JSON
    const displayData = { ...data };
    if (displayData.annotated_image_base64) {
      displayData.annotated_image_base64 = `[Base64 Image String (${displayData.annotated_image_base64.length} bytes)]`;
    }
    jsonOutput.innerHTML = `<code>${JSON.stringify(displayData, null, 2)}</code>`;
  }
});
