// Explainable CNN dashboard: drawing, API calls and rendering.
"use strict";

const CONFIG = JSON.parse(document.getElementById("page-config").textContent);
const METHOD_COLORS = {
    saliency: "#1f77b4",
    smoothgrad: "#9467bd",
    integrated_gradients: "#ff7f0e",
    grad_cam: "#2ca02c",
    occlusion: "#d62728",
};
const RANDOM_COLOR = "#7f7f7f";

const state = {
    dataset: document.querySelector('input[name="dataset"]:checked').value,
    lastInput: null,   // {image: dataURL} or {sample_index: n}
    busy: false,
};

// ---------- helpers ----------

const $ = (selector, root = document) => root.querySelector(selector);

function setStatus(message, isError = false) {
    const status = $("#status");
    status.textContent = message;
    status.classList.toggle("error", isError);
}

async function postJson(url, body) {
    const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
}

// Matplotlib-style "jet" colormap, t in [0, 1].
function jet(t) {
    const clamp = (v) => Math.max(0, Math.min(1, v));
    return [clamp(1.5 - Math.abs(4 * t - 3)), clamp(1.5 - Math.abs(4 * t - 2)), clamp(1.5 - Math.abs(4 * t - 1))]
        .map((v) => Math.round(v * 255));
}

// Draw a 28x28 grayscale image, optionally with a heatmap blended on top,
// scaled up to the canvas without smoothing so real pixels stay visible.
function drawGrid(canvas, gray, heatmap = null) {
    const size = gray.length;
    const small = document.createElement("canvas");
    small.width = small.height = size;
    const ctx = small.getContext("2d");
    const pixels = ctx.createImageData(size, size);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const i = (y * size + x) * 4;
            const g = gray[y][x] * 255;
            let rgb = [g, g, g];
            if (heatmap) {
                const t = heatmap[y][x];
                const alpha = 0.25 + 0.6 * t;   // stronger colour where evidence is strong
                rgb = jet(t).map((c, k) => (1 - alpha) * rgb[k] * 0.6 + alpha * c);
            }
            pixels.data.set([rgb[0], rgb[1], rgb[2], 255], i);
        }
    }
    ctx.putImageData(pixels, 0, 0);
    const target = canvas.getContext("2d");
    target.imageSmoothingEnabled = false;
    target.clearRect(0, 0, canvas.width, canvas.height);
    target.drawImage(small, 0, 0, canvas.width, canvas.height);
}

// ---------- drawing canvas ----------

const drawCanvas = $("#draw-canvas");
const drawCtx = drawCanvas.getContext("2d");
let drawing = false;
let hasDrawing = false;

function canvasPoint(event) {
    const rect = drawCanvas.getBoundingClientRect();
    return {
        x: (event.clientX - rect.left) * (drawCanvas.width / rect.width),
        y: (event.clientY - rect.top) * (drawCanvas.height / rect.height),
    };
}

drawCanvas.addEventListener("pointerdown", (event) => {
    drawing = true;
    drawCanvas.setPointerCapture(event.pointerId);
    const p = canvasPoint(event);
    drawCtx.beginPath();
    drawCtx.moveTo(p.x, p.y);
    drawCtx.lineTo(p.x + 0.1, p.y + 0.1);
    drawCtx.stroke();
    hasDrawing = true;
});

drawCanvas.addEventListener("pointermove", (event) => {
    if (!drawing) return;
    const p = canvasPoint(event);
    drawCtx.lineTo(p.x, p.y);
    drawCtx.stroke();
});

function endStroke() {
    if (!drawing) return;
    drawing = false;
    // Explain automatically shortly after each stroke.
    clearTimeout(endStroke.timer);
    endStroke.timer = setTimeout(explainDrawing, 400);
}
drawCanvas.addEventListener("pointerup", endStroke);
drawCanvas.addEventListener("pointercancel", endStroke);

function resetDrawCanvas() {
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    // White stroke on a transparent canvas: the server reads the alpha channel.
    drawCtx.strokeStyle = "#fff";
    drawCtx.lineWidth = ["mnist", "decoy_mnist"].includes(state.dataset) ? 18 : 22;  // digit-like strokes
    drawCtx.lineCap = "round";
    drawCtx.lineJoin = "round";
    hasDrawing = false;
}

// ---------- actions ----------

function explainDrawing() {
    if (!hasDrawing) {
        setStatus("Draw something first.", true);
        return;
    }
    runExplain({image: drawCanvas.toDataURL("image/png")});
}

function explainSample() {
    const size = CONFIG.datasets[state.dataset].test_size;
    resetDrawCanvas();
    runExplain({sample_index: Math.floor(Math.random() * size)});
}

function explainUpload(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
        setStatus("Please choose an image file.", true);
        return;
    }
    const reader = new FileReader();
    reader.onload = () => {
        resetDrawCanvas();
        runExplain({image: reader.result});
    };
    reader.onerror = () => setStatus("Could not read the file.", true);
    reader.readAsDataURL(file);
}

async function runExplain(input) {
    if (state.busy) return;
    state.busy = true;
    state.lastInput = input;
    setStatus("Computing explanations…");
    try {
        const result = await postJson("/api/explain", {dataset: state.dataset, ...input});
        renderResult(result);
        $("#sanity-btn").disabled = false;
        setStatus("Done.");
    } catch (error) {
        setStatus(error.message, true);
    } finally {
        state.busy = false;
    }
}

async function runSanity() {
    if (!state.lastInput || state.busy) return;
    state.busy = true;
    setStatus("Running sanity check on a randomized model…");
    try {
        const result = await postJson("/api/sanity", {dataset: state.dataset, ...state.lastInput});
        renderSanity(result);
        setStatus("Sanity check done.");
    } catch (error) {
        setStatus(error.message, true);
    } finally {
        state.busy = false;
    }
}

// ---------- rendering ----------

function renderResult(result) {
    state.lastInputGrid = result.input;
    drawGrid($("#input-preview"), result.input);

    // Prediction and probabilities
    const pred = result.prediction;
    let html = `Prediction: <span class="label">${escapeHtml(pred.label)}</span> ` +
        `(${(pred.confidence * 100).toFixed(1)}% confidence)`;
    if (result.true_label) {
        const correct = result.true_label.index === pred.index;
        html += `<br><span class="truth ${correct ? "correct" : "wrong"}">True label: ` +
            `${escapeHtml(result.true_label.label)} (${correct ? "correct" : "wrong"})</span>`;
    }
    $("#prediction").innerHTML = html;

    const classes = CONFIG.datasets[result.dataset].classes;
    $("#probabilities").innerHTML = result.probabilities.map((p, i) => `
        <div class="prob-row ${i === pred.index ? "top" : ""}">
            <span>${escapeHtml(classes[i])}</span>
            <div class="prob-bar" aria-hidden="true"><div style="width:${(p * 100).toFixed(1)}%"></div></div>
            <span class="value">${(p * 100).toFixed(1)}%</span>
        </div>`).join("");

    // Explanation cards
    const best = result.explanations.reduce((a, b) => (b.deletion_auc < a.deletion_auc ? b : a));
    for (const item of result.explanations) {
        const card = $(`.card[data-method="${item.method}"]`);
        drawGrid($(".heatmap", card), result.input, item.heatmap);
        $(".auc", card).textContent =
            `${item.deletion_auc.toFixed(3)} (random ${result.random_baseline.deletion_auc.toFixed(3)})`;
        $(".badge", card).hidden = item !== best;
        $(".sanity", card).hidden = true;
    }
    $("#sanity-note").hidden = true;

    renderChart(result);
}

function renderSanity(result) {
    for (const item of result.explanations) {
        const card = $(`.card[data-method="${item.method}"]`);
        drawGrid($(".sanity-heatmap", card), state.lastInputGrid, item.heatmap);
        const cell = $(".correlation", card);
        if (item.correlation === null) {
            // A blank map shares nothing with the original explanation.
            cell.textContent = "empty map (changed: passes)";
            cell.className = "correlation good";
        } else {
            // 0.5 is only a visual guide; the thesis reports the raw values.
            const passes = Math.abs(item.correlation) < 0.5;
            cell.textContent = `${item.correlation.toFixed(3)} (${passes ? "changed: passes" : "similar: fails"})`;
            cell.className = `correlation ${passes ? "good" : "bad"}`;
        }
        $(".sanity", card).hidden = false;
    }
    $("#sanity-note").hidden = false;
}

function renderChart(result) {
    const canvas = $("#deletion-chart");
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const m = {left: 48, right: 12, top: 12, bottom: 40};
    const px = (x) => m.left + x * (W - m.left - m.right);
    const py = (y) => H - m.bottom - y * (H - m.top - m.bottom);

    ctx.clearRect(0, 0, W, H);
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = "#5b6776";
    ctx.strokeStyle = "#e3e7eb";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const v = i / 5;
        ctx.beginPath(); ctx.moveTo(px(0), py(v)); ctx.lineTo(px(1), py(v)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(px(v), py(0)); ctx.lineTo(px(v), py(1)); ctx.stroke();
        ctx.textAlign = "right"; ctx.fillText(v.toFixed(1), px(0) - 6, py(v) + 4);
        ctx.textAlign = "center"; ctx.fillText(v.toFixed(1), px(v), py(0) + 16);
    }
    ctx.fillText("Fraction of pixels removed (most important first)", px(0.5), H - 6);
    ctx.save();
    ctx.translate(12, py(0.5)); ctx.rotate(-Math.PI / 2);
    ctx.fillText("Probability of predicted class", 0, 0);
    ctx.restore();

    const series = result.explanations.map((e) => ({
        name: e.title, curve: e.deletion_curve, auc: e.deletion_auc, color: METHOD_COLORS[e.method], dashed: false,
    }));
    series.push({
        name: "Random (baseline)", curve: result.random_baseline.deletion_curve,
        auc: result.random_baseline.deletion_auc, color: RANDOM_COLOR, dashed: true,
    });

    for (const s of series) {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2;
        ctx.setLineDash(s.dashed ? [6, 4] : []);
        ctx.beginPath();
        s.curve.forEach((y, i) => {
            const x = i / (s.curve.length - 1);
            if (i === 0) ctx.moveTo(px(x), py(y)); else ctx.lineTo(px(x), py(y));
        });
        ctx.stroke();
    }
    ctx.setLineDash([]);

    $("#chart-legend").innerHTML = series.map((s) => `
        <li><span class="swatch ${s.dashed ? "dashed" : ""}" style="background:${s.color}"></span>
        ${escapeHtml(s.name)}: AUC ${s.auc.toFixed(3)}</li>`).join("");
}

function clearResults() {
    $("#prediction").textContent = "No input yet.";
    $("#probabilities").innerHTML = "";
    $("#chart-legend").innerHTML = "";
    const chart = $("#deletion-chart");
    chart.getContext("2d").clearRect(0, 0, chart.width, chart.height);
    for (const canvas of document.querySelectorAll(".heatmap, #input-preview")) {
        canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    }
    for (const card of document.querySelectorAll(".card")) {
        $(".auc", card).textContent = "–";
        $(".badge", card).hidden = true;
        $(".sanity", card).hidden = true;
    }
    $("#sanity-note").hidden = true;
    $("#sanity-btn").disabled = true;
    state.lastInput = null;
}

async function loadBenchmark() {
    const target = $("#benchmark");
    try {
        const response = await fetch(`/api/benchmark/${state.dataset}`);
        const data = await response.json();
        if (!data.rows || data.rows.length === 0) {
            target.innerHTML = `<p class="hint">${escapeHtml(data.message || "No results.")}</p>`;
            return;
        }
        const columns = data.columns;
        const cell = (v) => {
            if (v === null || v === undefined) return "–";
            return typeof v === "number" ? v.toFixed(3) : escapeHtml(v);
        };
        target.innerHTML = `<table>
            <thead><tr>${columns.map(([, t]) => `<th scope="col">${escapeHtml(t)}</th>`).join("")}</tr></thead>
            <tbody>${data.rows.map((row) => `<tr>${columns.map(([k]) => `<td>${cell(row[k])}</td>`).join("")}</tr>`).join("")}</tbody>
        </table>`;
    } catch (error) {
        target.innerHTML = `<p class="hint">Could not load benchmark results.</p>`;
    }
}

function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}

// ---------- wiring ----------

const HINTS = {
    mnist: "Draw a digit in the box, upload an image, or pick a test image.",
    fashion_mnist: "Pick a test image or upload a clothing photo on a plain background. Drawing a simple outline also works.",
    pneumonia_mnist: "Pick a test chest X-ray. Research demo only, not a diagnostic tool.",
    decoy_mnist: "This model learned a shortcut: a white patch whose position gives away the class. " +
        "Test images put the patch at a WRONG class's position, so the model is fooled; see which methods find the patch. " +
        "Drawings have no patch, so the model usually fails on them.",
    decoy_pneumonia: "This model learned a shortcut: a corner marker that gives away the label. " +
        "Test X-rays have the marker in the WRONG corner, so the model is fooled; see which methods highlight it. Research demo only.",
};

function updateHint() {
    $("#draw-hint").textContent = HINTS[state.dataset] || "Pick a test image or upload an image.";
}

$("#explain-btn").addEventListener("click", explainDrawing);
$("#clear-btn").addEventListener("click", () => { resetDrawCanvas(); clearResults(); setStatus(""); });
$("#sample-btn").addEventListener("click", explainSample);
$("#sanity-btn").addEventListener("click", runSanity);
$("#upload-input").addEventListener("change", (event) => {
    explainUpload(event.target.files[0]);
    event.target.value = "";  // allow re-uploading the same file
});

for (const radio of document.querySelectorAll('input[name="dataset"]')) {
    radio.addEventListener("change", () => {
        state.dataset = radio.value;
        resetDrawCanvas();
        clearResults();
        updateHint();
        setStatus("");
        loadBenchmark();
    });
}

resetDrawCanvas();
updateHint();
loadBenchmark();
