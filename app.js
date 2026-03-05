/* ══════════════════════════════════════════════════════════════════════
   FamilyFace — app.js
   Engine : face-api.js  (68 facial landmarks + 128-dim descriptor)
   Weights : jsdelivr CDN — no WASM, loads in ~5 s
   Runs 100 % in-browser. Zero data leaves your device.
   ══════════════════════════════════════════════════════════════════════ */

'use strict';

// CDN sources tried in order. The npm package bundles the weights, so it loads
// faster and more reliably than the GitHub CDN.
const MODEL_URLS = [
  'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights',
  'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights',
];

// ── 68-point landmark index groups ────────────────────────────────────────────
// face-api.js uses dlib's 68-point model:
// 0-16 jaw · 17-21 R-brow · 22-26 L-brow · 27-35 nose · 36-47 eyes · 48-67 mouth
const FEATURES = [
  {
    key: 'eyes',
    label: 'Eyes',
    emoji: '👁️',
    indices: [36,37,38,39,40,41, 42,43,44,45,46,47],
    note: '12 landmark pts — eye shape & lid curvature',
  },
  {
    key: 'eyebrows',
    label: 'Eyebrows',
    emoji: '〰️',
    indices: [17,18,19,20,21, 22,23,24,25,26],
    note: '10 landmark pts — arch height & shape',
  },
  {
    key: 'nose',
    label: 'Nose',
    emoji: '👃',
    indices: [27,28,29,30,31,32,33,34,35],
    note: '9 landmark pts — bridge, tip & nostril width',
  },
  {
    key: 'mouth',
    label: 'Lips & Mouth',
    emoji: '👄',
    indices: [48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67],
    note: '20 landmark pts — lip shape, cupid\'s bow & width',
  },
  {
    key: 'jawline',
    label: 'Face Shape',
    emoji: '🔷',
    indices: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
    note: '17 landmark pts — jawline width & chin shape',
  },
];

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  modelsReady: false,
  images:     { parent1: null, parent2: null, child: null },
};

const $           = id => document.getElementById(id);
const modelBanner = $('model-banner');
const modelText   = $('model-status-text');
const analyzeBtn  = $('analyze-btn');
const resetBtn    = $('reset-btn');
const ctaHint     = $('cta-hint');
const resultsEl   = $('results-section');

// ══════════════════════════════════════════════════════════════════════════════
// 1. LOAD MODELS  (SSD face detector + 68-pt landmarks + 128-d descriptor)
// ══════════════════════════════════════════════════════════════════════════════
async function loadModels() {
  // Reset to spinner in case this is a retry
  modelBanner.classList.remove('ready', 'hidden');
  const spinnerEl = modelBanner.querySelector('span, .spinner');
  if (spinnerEl && !spinnerEl.classList.contains('spinner')) {
    spinnerEl.outerHTML = '<div class="spinner"></div>';
  }
  state.modelsReady = false;
  updateButtons();

  const TIMEOUT_MS = 25_000; // 25 s per CDN source before trying next

  for (let i = 0; i < MODEL_URLS.length; i++) {
    const url = MODEL_URLS[i];
    const label = i === 0 ? 'primary server' : `backup server ${i}`;
    try {
      // Load the three models one-by-one so the user sees real progress
      modelText.textContent = `[1/3] Loading face detector (${label})…`;
      await loadOneWithTimeout(faceapi.nets.ssdMobilenetv1, url, TIMEOUT_MS);

      modelText.textContent = `[2/3] Loading landmark model (${label})…`;
      await loadOneWithTimeout(faceapi.nets.faceLandmark68Net, url, TIMEOUT_MS);

      modelText.textContent = `[3/3] Loading recognition model (${label})…`;
      await loadOneWithTimeout(faceapi.nets.faceRecognitionNet, url, TIMEOUT_MS);

      // ── Success ──
      state.modelsReady = true;
      modelBanner.classList.add('ready');
      modelBanner.querySelector('.spinner').outerHTML = '<span style="font-size:1.1rem">✅</span>';
      modelText.textContent = 'Models ready — upload your family photos below';
      updateButtons();
      setTimeout(() => modelBanner.classList.add('hidden'), 4000);
      return; // done

    } catch (err) {
      console.warn(`CDN ${label} failed:`, err.message);
      if (i < MODEL_URLS.length - 1) {
        modelText.textContent = `Server ${i + 1} timed out — trying next…`;
        await new Promise(r => setTimeout(r, 600)); // brief pause before retry
      }
    }
  }

  // ── All sources failed ──
  modelBanner.querySelector('.spinner').outerHTML = '<span style="font-size:1.1rem">❌</span>';
  modelText.innerHTML =
    'Could not load models. Check your internet connection and '
    + '<button id="retry-btn" style="'
    + 'background:#7c3aed;color:#fff;border:none;border-radius:6px;'
    + 'padding:3px 12px;cursor:pointer;font-weight:700;margin-left:6px'
    + '" onclick="loadModels()">Retry</button>';
}

/** Wrap a single net.loadFromUri() with a timeout so hangs don't block forever. */
function loadOneWithTimeout(net, url, ms) {
  return Promise.race([
    net.loadFromUri(url),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`timed out after ${ms / 1000} s`)), ms)
    ),
  ]);
}

// ══════════════════════════════════════════════════════════════════════════════
// 2. UPLOAD / DRAG-DROP
// ══════════════════════════════════════════════════════════════════════════════
['parent1', 'parent2', 'child'].forEach(slot => {
  const drop  = $(`drop-${slot}`);
  const input = $(`input-${slot}`);

  drop.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('drag-over'); });
  drop.addEventListener('dragleave', ()  => drop.classList.remove('drag-over'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f?.type.startsWith('image/')) loadImageFile(slot, f);
  });
  input.addEventListener('change', () => { if (input.files[0]) loadImageFile(slot, input.files[0]); });
});

document.querySelectorAll('.remove-btn').forEach(btn =>
  btn.addEventListener('click', () => clearSlot(btn.dataset.slot))
);

function loadImageFile(slot, file) {
  const reader = new FileReader();
  reader.onload = ev => {
    const img = new Image();
    img.onload = () => {
      state.images[slot] = img;
      renderPreview(slot, img);
      updateButtons();
      resultsEl.style.display = 'none';
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

function renderPreview(slot, img) {
  const canvas = $(`canvas-${slot}`);
  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.getContext('2d').drawImage(img, 0, 0);
  $(`drop-${slot}`).style.display = 'none';
  $(`wrap-${slot}`).style.display = 'block';
  $(`tag-${slot}`).textContent    = '';
  const old = $(`acc-badge-${slot}`);
  if (old) old.remove();
}

function clearSlot(slot) {
  state.images[slot] = null;
  $(`wrap-${slot}`).style.display  = 'none';
  $(`drop-${slot}`).style.display  = 'flex';
  $(`input-${slot}`).value         = '';
  $(`tag-${slot}`).textContent     = '';
  const old = $(`acc-badge-${slot}`);
  if (old) old.remove();
  updateButtons();
  resultsEl.style.display = 'none';
}

// ── Reset All ────────────────────────────────────────────────────────────────
resetBtn.addEventListener('click', () =>
  ['parent1', 'parent2', 'child'].forEach(clearSlot)
);

function updateButtons() {
  const hasChild  = !!state.images.child;
  const hasParent = !!(state.images.parent1 || state.images.parent2);
  const hasAny    = !!(state.images.child || state.images.parent1 || state.images.parent2);

  analyzeBtn.disabled = !(state.modelsReady && hasChild && hasParent);
  resetBtn.disabled   = !hasAny;

  if (!state.modelsReady) ctaHint.textContent = 'Downloading models…';
  else if (!hasChild)     ctaHint.textContent = "Upload the child's photo to continue";
  else if (!hasParent)    ctaHint.textContent = "Upload at least one parent's photo to continue";
  else                    ctaHint.textContent = 'Ready — click Analyze Family Resemblance';
}

// ══════════════════════════════════════════════════════════════════════════════
// 3. ANALYZE
// ══════════════════════════════════════════════════════════════════════════════
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  analyzeBtn.disabled = true;
  resultsEl.style.display = 'block';
  resultsEl.innerHTML = analyzingHTML();
  setTimeout(() => resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);

  try {
    const detections = {};
    const slots = ['child', 'parent1', 'parent2'].filter(s => state.images[s]);

    for (const slot of slots) {
      const det = await faceapi
        .detectSingleFace(state.images[slot], new faceapi.SsdMobilenetv1Options({ minConfidence: 0.3 }))
        .withFaceLandmarks()
        .withFaceDescriptor();
      detections[slot] = det || null;
    }

    if (!detections.child) {
      showError("No face detected in the child's photo. Try a clearer, front-facing photo.");
      analyzeBtn.disabled = false;
      return;
    }

    const hasP1 = !!detections.parent1;
    const hasP2 = !!detections.parent2;

    if (!hasP1 && !hasP2) {
      showError("No faces detected in parent photos. Try clearer, front-facing photos.");
      analyzeBtn.disabled = false;
      return;
    }

    // Draw landmark overlays + accuracy badges
    const accuracies = {};
    for (const slot of slots) {
      const det = detections[slot];
      if (det) {
        drawLandmarks(slot, det);
        accuracies[slot] = computeAccuracy(det, state.images[slot]);
        showAccuracyBadge(slot, accuracies[slot].pct);
        $(`tag-${slot}`).textContent = `✅ 68 landmarks`;
      } else {
        $(`tag-${slot}`).textContent = '⚠️ No face found';
      }
    }

    const scores = computeScores(detections, hasP1, hasP2);
    renderResults(scores, accuracies, hasP1, hasP2);
    analyzeBtn.disabled = false;

  } catch (err) {
    console.error(err);
    showError(`Analysis failed: ${err.message}`);
    analyzeBtn.disabled = false;
  }
}

// ── Draw landmarks on canvas ──────────────────────────────────────────────────
function drawLandmarks(slot, det) {
  const canvas = $(`canvas-${slot}`);
  const img    = state.images[slot];
  const ctx    = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  const box  = det.detection.box;
  const pts  = det.landmarks.positions;
  const dotR = Math.max(1.5, img.naturalWidth / 380);
  const lw   = Math.max(1.5, img.naturalWidth / 400);

  const slotColor = { parent1: '#2563eb', parent2: '#db2777', child: '#059669' };
  const primary   = slotColor[slot] || '#7c3aed';

  // Bounding box
  ctx.strokeStyle = primary;
  ctx.lineWidth   = lw * 1.6;
  ctx.strokeRect(box.x, box.y, box.width, box.height);

  // Feature regions with distinct colours
  const regionColors = [
    { label: 'jaw',      color: '#10b981', idx: range(0,  17) },
    { label: 'r-brow',   color: '#8b5cf6', idx: range(17, 22) },
    { label: 'l-brow',   color: '#8b5cf6', idx: range(22, 27) },
    { label: 'nose',     color: '#f59e0b', idx: range(27, 36) },
    { label: 'r-eye',    color: '#3b82f6', idx: range(36, 42) },
    { label: 'l-eye',    color: '#3b82f6', idx: range(42, 48) },
    { label: 'mouth-out',color: '#ef4444', idx: range(48, 60) },
    { label: 'mouth-in', color: '#f87171', idx: range(60, 68) },
  ];

  regionColors.forEach(({ color, idx }) => {
    ctx.fillStyle = color + 'dd';
    idx.forEach(i => {
      const p = pts[i];
      if (!p) return;
      ctx.beginPath();
      ctx.arc(p.x, p.y, dotR, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}

function range(start, end) {
  return Array.from({ length: end - start }, (_, i) => start + i);
}

function showAccuracyBadge(slot, pct) {
  const wrap  = $(`wrap-${slot}`);
  const level = pct >= 75 ? 'high' : pct >= 50 ? 'medium' : 'low';
  const icon  = pct >= 75 ? '✅' : pct >= 50 ? '⚠️' : '❌';
  const el    = document.createElement('div');
  el.id        = `acc-badge-${slot}`;
  el.className = `acc-photo-badge ${level}`;
  el.textContent = `${icon} ${pct}% quality`;
  wrap.appendChild(el);
}

// ══════════════════════════════════════════════════════════════════════════════
// 4. ACCURACY — face size, detection confidence, frontality
// ══════════════════════════════════════════════════════════════════════════════
function computeAccuracy(det, img) {
  const box  = det.detection.box;
  const pts  = det.landmarks.positions;
  const conf = det.detection.score;                            // 0–1 SSD confidence

  // Face size vs image area
  const imgArea   = img.naturalWidth * img.naturalHeight;
  const faceArea  = box.width * box.height;
  const sizeScore = Math.min(1, faceArea / imgArea / 0.09);   // 9 % = full score

  // Eye-roll (head tilt)
  const rEyeC = centroid(pts.slice(36, 42));
  const lEyeC = centroid(pts.slice(42, 48));
  const tiltDeg = Math.abs(Math.atan2(lEyeC.y - rEyeC.y, lEyeC.x - rEyeC.x) * 180 / Math.PI);
  const rollScore = Math.max(0, 1 - tiltDeg / 25);

  // Nose yaw (centering between eyes)
  const noseTip  = pts[30];
  const eyeMidX  = (rEyeC.x + lEyeC.x) / 2;
  const yawNorm  = Math.abs(noseTip.x - eyeMidX) / Math.max(1, box.width);
  const yawScore = Math.max(0, 1 - yawNorm / 0.15);

  const total = conf * 0.20 + sizeScore * 0.30 + rollScore * 0.25 + yawScore * 0.25;
  const pct   = Math.round(Math.min(100, total * 100));

  const tips = [];
  if (sizeScore  < 0.5) tips.push('face too small — move closer or crop tighter');
  if (rollScore  < 0.6) tips.push('head tilted — keep it level');
  if (yawScore   < 0.6) tips.push('face turned sideways — look straight at the camera');
  if (conf       < 0.6) tips.push('photo may be blurry or poorly lit');

  return {
    pct,
    factors: {
      confidence: Math.round(conf * 100),
      size:       Math.round(sizeScore  * 100),
      roll:       Math.round(rollScore  * 100),
      yaw:        Math.round(yawScore   * 100),
    },
    tips,
  };
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. SIMILARITY SCORING
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Overall similarity = weighted average of the feature-level scores.
 * Weights reflect each feature's prominence in perceived resemblance.
 * Using the same numbers as the bars guarantees the gauges and the
 * inheritance summary always tell the same story.
 */
const FEATURE_WEIGHTS = {
  eyes:     0.25,   // most distinctive feature
  nose:     0.25,   // strong genetic marker
  mouth:    0.20,
  jawline:  0.15,
  eyebrows: 0.15,
};

function overallFromFeatures(featureArr, parent) {
  let total = 0, wSum = 0;
  featureArr.forEach(f => {
    const score = f.scores[parent];
    if (score == null) return;
    const w = FEATURE_WEIGHTS[f.key] ?? 0.20;
    total += score * w;
    wSum  += w;
  });
  return wSum > 0 ? Math.round(total / wSum) : 0;
}

/**
 * IOD-normalised landmark comparison for a feature point set.
 * Returns 0–100 %.
 */
function featureSim(childNorm, parentNorm, indices) {
  let total = 0;
  for (const i of indices) {
    total += dist2d(childNorm[i], parentNorm[i]);
  }
  const avgDist = total / indices.length;
  // After IOD-normalisation, avg dist for close family ≈ 0.05–0.18
  // Map [0, 0.30] → [100, 0] with exponential for smoother spread
  return Math.max(0, Math.min(100, Math.round(100 * Math.exp(-avgDist / 0.12))));
}

/**
 * Normalise positions to be translation- + scale-invariant.
 * Origin = mid-eye point; scale = inter-ocular distance (IOD).
 */
function normaliseLandmarks(positions) {
  const rEyeC = centroid(positions.slice(36, 42));
  const lEyeC = centroid(positions.slice(42, 48));
  const iod   = dist2d(rEyeC, lEyeC);
  if (iod < 1) return positions;
  const cx = (rEyeC.x + lEyeC.x) / 2;
  const cy = (rEyeC.y + lEyeC.y) / 2;
  return positions.map(p => ({ x: (p.x - cx) / iod, y: (p.y - cy) / iod }));
}

function computeScores(detections, hasP1, hasP2) {
  const childNorm = normaliseLandmarks(detections.child.landmarks.positions);

  const result = {
    overall:  {},
    features: FEATURES.map(f => ({ ...f, scores: {} })),
  };

  // Pass 1: compute per-feature scores for every parent
  ['parent1', 'parent2'].forEach(p => {
    if (!detections[p]) return;
    const parNorm = normaliseLandmarks(detections[p].landmarks.positions);
    result.features.forEach(f => {
      f.scores[p] = featureSim(childNorm, parNorm, f.indices);
    });
  });

  // Pass 2: overall = weighted average of the SAME feature scores
  // → gauges and inheritance summary now always agree
  ['parent1', 'parent2'].forEach(p => {
    if (!detections[p]) return;
    result.overall[p] = overallFromFeatures(result.features, p);
  });

  return result;
}

// ── Math helpers ─────────────────────────────────────────────────────────────
function centroid(pts) {
  const s = pts.reduce((a, p) => ({ x: a.x + p.x, y: a.y + p.y }), { x: 0, y: 0 });
  return { x: s.x / pts.length, y: s.y / pts.length };
}
function dist2d(a, b) { return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2); }

// ══════════════════════════════════════════════════════════════════════════════
// 6. RENDER RESULTS
// ══════════════════════════════════════════════════════════════════════════════
function renderResults(scores, accuracies, hasP1, hasP2) {
  resultsEl.innerHTML = `
    <div class="results-header">
      <h2>Family Resemblance Report</h2>
      <p>Overall scores are a <strong>weighted average of the 5 feature scores</strong> below — every number on this page comes from the same landmark geometry</p>
    </div>
    ${buildConfidenceCard(accuracies, hasP1, hasP2)}
    ${buildScoreCards(scores.overall, hasP1, hasP2)}
    ${buildFeatureBreakdown(scores.features, hasP1, hasP2)}
    ${buildInheritanceSummary(scores.features, hasP1, hasP2)}
  `;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    animateGauges(scores.overall, hasP1, hasP2);
    document.querySelectorAll('[data-fill-pct]').forEach(el => {
      el.style.width = el.dataset.fillPct + '%';
    });
  }));
}

// ── Confidence / Accuracy card ────────────────────────────────────────────────
function buildConfidenceCard(accuracies, hasP1, hasP2) {
  const slots = [
    { key: 'child',   label: '👶 Child' },
    { key: 'parent1', label: '👤 Parent 1', skip: !hasP1 },
    { key: 'parent2', label: '👤 Parent 2', skip: !hasP2 },
  ].filter(s => !s.skip && accuracies[s.key]);

  const minScore  = Math.min(...slots.map(s => accuracies[s.key].pct));
  const overallMsg = minScore >= 75 ? '🎯 Results are highly reliable'
                   : minScore >= 50 ? '📋 Results are a reasonable estimate'
                   : '⚠️ Results may be less accurate — see improvement tips below';

  const rows = slots.map(({ key, label }) => {
    const { pct, factors, tips } = accuracies[key];
    const level = pct >= 75 ? 'high' : pct >= 50 ? 'medium' : 'low';
    const badge = pct >= 75 ? '✅ High quality' : pct >= 50 ? '⚠️ Fair quality' : '❌ Low quality';
    const tip   = tips.length ? `💡 ${tips.join(' · ')}` : '✓ Great photo for analysis';
    const factorLine = `Detection: ${factors.confidence}% · Size: ${factors.size}% · Frontality: ${Math.round((factors.roll + factors.yaw) / 2)}%`;
    return `
      <div class="conf-row">
        <div class="conf-who">${label}</div>
        <div class="conf-bar-wrap">
          <div class="conf-bar-track"><div class="conf-bar-fill conf-${level}" data-fill-pct="${pct}"></div></div>
          <span class="conf-pct">${pct}%</span>
          <span class="conf-badge conf-badge-${level}">${badge}</span>
        </div>
        <div class="conf-factors">${factorLine}</div>
        <div class="conf-tip">${tip}</div>
      </div>`;
  }).join('');

  return `
    <div class="confidence-card">
      <h3>📊 Analysis Confidence</h3>
      <p class="conf-subtitle">How reliable these results are, based on photo quality</p>
      <div class="conf-rows">${rows}</div>
      <div class="conf-overall"><strong>Overall result confidence: ${minScore}%</strong> — ${overallMsg}</div>
    </div>`;
}

// ── Overall score gauges ──────────────────────────────────────────────────────
function buildScoreCards(overall, hasP1, hasP2) {
  let h = '<div class="score-cards">';
  if (hasP1) h += gaugeHTML('p1', 'Child vs Parent 1', overall.parent1 ?? 0, 'p1');
  if (hasP1 && hasP2) {
    const avg = Math.round(((overall.parent1 ?? 0) + (overall.parent2 ?? 0)) / 2);
    h += gaugeHTML('neutral', 'Family Blend', avg, 'neutral');
  }
  if (hasP2) h += gaugeHTML('p2', 'Child vs Parent 2', overall.parent2 ?? 0, 'p2');
  return h + '</div>';
}

function gaugeHTML(id, label, pct, arc) {
  const tag = pct >= 65 ? '🔥 Strong resemblance' : pct >= 45 ? '😊 Good resemblance' : '🌱 Some resemblance';
  return `
    <div class="score-card ${id}-card">
      <div class="score-label">${label}</div>
      <div class="gauge-wrap">
        <svg class="gauge-svg" viewBox="0 0 110 110">
          <circle class="gauge-bg" cx="55" cy="55" r="45"/>
          <circle class="gauge-arc ${arc}" id="arc-${id}" cx="55" cy="55" r="45"/>
        </svg>
        <div class="gauge-pct" id="pct-${id}">0%</div>
      </div>
      <div class="score-sublabel">${tag}</div>
    </div>`;
}

function animateGauges(overall, hasP1, hasP2) {
  const C = 283;
  const go = (id, pct) => {
    const arc = document.getElementById(`arc-${id}`);
    const el  = document.getElementById(`pct-${id}`);
    if (!arc || !el) return;
    arc.style.strokeDashoffset = C - (C * pct) / 100;
    let v = 0;
    const tick = () => { v = Math.min(v + 2, pct); el.textContent = v + '%'; if (v < pct) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  };
  if (hasP1) go('p1', overall.parent1 ?? 0);
  if (hasP2) go('p2', overall.parent2 ?? 0);
  if (hasP1 && hasP2) go('neutral', Math.round(((overall.parent1 ?? 0) + (overall.parent2 ?? 0)) / 2));
}

// ── Feature breakdown ─────────────────────────────────────────────────────────
function buildFeatureBreakdown(features, hasP1, hasP2) {
  let h = `<div class="features-card"><h3>Feature-by-Feature Breakdown</h3>`;
  features.forEach((f, i) => {
    if (i > 0) h += '<hr class="feature-divider"/>';
    const p1s = f.scores.parent1 ?? null, p2s = f.scores.parent2 ?? null;
    const winner = resolveWinner(p1s, p2s, hasP1, hasP2);
    h += `
      <div class="feature-row">
        <div class="feature-name">
          <span class="feature-emoji">${f.emoji}</span>
          <div><div>${f.label}</div><div class="feature-note">${f.note}</div></div>
        </div>
        <div class="feature-bars">
    `;
    if (hasP1 && p1s !== null) h += barRow('p1', 'Parent 1', p1s);
    if (hasP2 && p2s !== null) h += barRow('p2', 'Parent 2', p2s);
    if (winner) h += `<div class="bar-row"><span style="margin-left:70px">${winnerBadge(winner)}</span></div>`;
    h += '</div></div>';
  });
  return h + '</div>';
}

function barRow(cls, label, pct) {
  return `
    <div class="bar-row">
      <span class="bar-label ${cls}">${label}</span>
      <div class="bar-track"><div class="bar-fill ${cls}" data-fill-pct="${pct}"></div></div>
      <span class="bar-pct ${cls}">${pct}%</span>
    </div>`;
}

function resolveWinner(p1, p2, hasP1, hasP2) {
  if (!hasP1 || !hasP2 || p1 === null || p2 === null) return null;
  if (Math.abs(p1 - p2) < 6) return 'tie';
  return p1 > p2 ? 'p1' : 'p2';
}

function winnerBadge(w) {
  if (w === 'tie') return `<span class="winner-badge tie">≈ Equal from both</span>`;
  return `<span class="winner-badge ${w}">▶ From ${w === 'p1' ? 'Parent 1' : 'Parent 2'}</span>`;
}

// ── Inheritance summary ───────────────────────────────────────────────────────
function buildInheritanceSummary(features, hasP1, hasP2) {
  if (!hasP1 || !hasP2) return '';
  const p1w = [], p2w = [], ties = [];
  features.forEach(f => {
    const d = Math.abs((f.scores.parent1 ?? 0) - (f.scores.parent2 ?? 0));
    if (d < 6) ties.push(f);
    else if ((f.scores.parent1 ?? 0) > (f.scores.parent2 ?? 0)) p1w.push(f);
    else p2w.push(f);
  });
  const dom = p1w.length > p2w.length ? 'Parent 1'
            : p2w.length > p1w.length ? 'Parent 2'
            : 'both parents equally';
  const pills = [
    ...p1w.map(f  => `<span class="pill p1">${f.emoji} ${f.label} → Parent 1</span>`),
    ...p2w.map(f  => `<span class="pill p2">${f.emoji} ${f.label} → Parent 2</span>`),
    ...ties.map(f => `<span class="pill unique">${f.emoji} ${f.label} → Blend of both</span>`),
  ].join('');
  return `
    <div class="inheritance-card">
      <h3>🧬 Inheritance Summary</h3>
      <p style="margin-bottom:18px;opacity:.85">Your child resembles <strong>${dom}</strong> the most
        (${p1w.length} feature${p1w.length !== 1 ? 's' : ''} from Parent 1 ·
         ${p2w.length} feature${p2w.length !== 1 ? 's' : ''} from Parent 2 ·
         ${ties.length} blend${ties.length !== 1 ? 's' : ''})
      </p>
      <div class="inheritance-pills">${pills}</div>
    </div>`;
}

// ══════════════════════════════════════════════════════════════════════════════
// 7. UI HELPERS
// ══════════════════════════════════════════════════════════════════════════════
function analyzingHTML() {
  return `
    <div class="analyzing-overlay">
      <div class="spinner" style="color:#7c3aed"></div>
      <h3>Analyzing faces…</h3>
      <p>Detecting landmarks and computing face descriptors</p>
    </div>`;
}
function showError(msg) {
  resultsEl.innerHTML = `<div class="error-box">❌ ${msg}</div>`;
}

// ══════════════════════════════════════════════════════════════════════════════
// START
// ══════════════════════════════════════════════════════════════════════════════
loadModels();
