/* ══════════════════════════════════════════════════════════════════════
   FamilyFace — app.js
   Engine: Native MediaPipe FaceMesh (no TensorFlow.js required)
   478 facial landmark points · runs 100 % in-browser
   ══════════════════════════════════════════════════════════════════════ */

'use strict';

// CDN base for MediaPipe WASM + model files (same version as the script tag)
const MP_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619';

// ══════════════════════════════════════════════════════════════════════════════
// LANDMARK INDEX REGIONS  (MediaPipe 478-point mesh)
// Coords are NORMALISED [0,1] relative to image width / height
// ══════════════════════════════════════════════════════════════════════════════
const REGIONS = {
  rightEye:    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  leftEye:     [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
  rightIris:   [468, 469, 470, 471, 472],
  leftIris:    [473, 474, 475, 476, 477],
  rightEyebrow:[46,  53,  52,  65,  55,  70,  63, 105,  66, 107],
  leftEyebrow: [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
  noseBridge:  [6, 197, 195, 5, 4, 1, 19, 94, 125, 141, 142],
  noseTip:     [97, 98, 99, 100, 101, 102, 2, 326, 327, 328, 329, 330, 331, 294, 278, 237, 79],
  outerLips:   [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
  innerLips:   [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
  faceOval:    [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
  chin:        [152, 377, 148, 400, 176, 149, 150, 136, 172, 58, 132, 175, 396],
  rightCheek:  [234, 93, 137, 177, 215, 138, 135, 169, 210, 214, 207, 205, 36, 142, 126, 116, 117, 118],
  leftCheek:   [454, 323, 366, 401, 435, 367, 364, 394, 430, 434, 427, 425, 266, 371, 355, 345, 346, 347],
  forehead:    [10, 109, 67, 103, 54, 21, 162, 127, 356, 389, 251, 284, 332, 297, 338, 69, 108, 151, 337, 299],
};

const FEATURES = [
  { key: 'eyes',      label: 'Eyes & Irises',    emoji: '👁️', regionKeys: ['rightEye','leftEye','rightIris','leftIris'],   note: '42 pts — eye shape, lid curvature, iris size' },
  { key: 'eyebrows',  label: 'Eyebrows',          emoji: '〰️', regionKeys: ['rightEyebrow','leftEyebrow'],                  note: '20 pts — arch height, thickness, shape' },
  { key: 'nose',      label: 'Nose',              emoji: '👃', regionKeys: ['noseBridge','noseTip'],                        note: '28 pts — bridge height, tip width, nostril flare' },
  { key: 'lips',      label: 'Lips & Mouth',      emoji: '👄', regionKeys: ['outerLips','innerLips'],                       note: '40 pts — cupid's bow, lip fullness, mouth width' },
  { key: 'faceShape', label: 'Face Shape & Jaw',  emoji: '🔷', regionKeys: ['faceOval','chin'],                            note: '49 pts — jawline, chin shape, face width' },
  { key: 'cheeks',    label: 'Cheeks & Temples',  emoji: '✨', regionKeys: ['rightCheek','leftCheek'],                     note: '36 pts — cheekbone height & prominence' },
  { key: 'forehead',  label: 'Forehead',           emoji: '🧠', regionKeys: ['forehead'],                                   note: '20 pts — forehead height, brow ridge' },
];

// ══════════════════════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════════════════════
const state = {
  fm:          null,   // FaceMesh instance
  modelsReady: false,
  images:     { parent1: null, parent2: null, child: null },
  landmarks:  { parent1: null, parent2: null, child: null },
};

const $          = id => document.getElementById(id);
const modelBanner = $('model-banner');
const modelText   = $('model-status-text');
const analyzeBtn  = $('analyze-btn');
const resetBtn    = $('reset-btn');
const ctaHint     = $('cta-hint');
const resultsEl   = $('results-section');

// ══════════════════════════════════════════════════════════════════════════════
// 1. INIT — create FaceMesh and pre-warm (downloads WASM + model in background)
// ══════════════════════════════════════════════════════════════════════════════
async function init() {
  try {
    modelText.textContent = 'Downloading MediaPipe FaceMesh model…';

    state.fm = new FaceMesh({
      locateFile: file => `${MP_CDN}/${file}`,
    });

    state.fm.setOptions({
      maxNumFaces:            1,
      refineLandmarks:        true,   // enables 478-point mesh + iris
      minDetectionConfidence: 0.5,
      minTrackingConfidence:  0.5,
    });

    // Pre-warm: send a blank canvas so WASM + model download now, not on first click
    await sendImage(createBlankCanvas());

    state.modelsReady = true;
    modelBanner.classList.add('ready');
    modelBanner.querySelector('.spinner').outerHTML = '<span style="font-size:1.1rem">✅</span>';
    modelText.textContent = 'MediaPipe FaceMesh ready (478 landmarks per face)';
    updateButtons();
    setTimeout(() => modelBanner.classList.add('hidden'), 4000);

  } catch (err) {
    modelText.textContent = `❌ Failed to load: ${err.message}`;
    console.error(err);
  }
}

/** Wrap FaceMesh.send() in a Promise that resolves with the landmark array (or null). */
function sendImage(imgOrCanvas) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Detection timed out after 20 s')), 20_000);

    state.fm.onResults(results => {
      clearTimeout(timer);
      // multiFaceLandmarks[0] is an array of 478 {x,y,z} objects in [0,1] coords
      resolve(results.multiFaceLandmarks?.[0] ?? null);
    });

    state.fm.send({ image: imgOrCanvas }).catch(err => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

function createBlankCanvas() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  return c;
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
    if (f?.type.startsWith('image/')) loadImage(slot, f);
  });
  input.addEventListener('change', () => { if (input.files[0]) loadImage(slot, input.files[0]); });
});

document.querySelectorAll('.remove-btn').forEach(btn =>
  btn.addEventListener('click', () => clearSlot(btn.dataset.slot))
);

function loadImage(slot, file) {
  const reader = new FileReader();
  reader.onload = ev => {
    const img = new Image();
    img.onload = () => {
      state.images[slot]    = img;
      state.landmarks[slot] = null;
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
  state.images[slot]    = null;
  state.landmarks[slot] = null;
  $(`wrap-${slot}`).style.display = 'none';
  $(`drop-${slot}`).style.display = 'flex';
  $(`input-${slot}`).value        = '';
  $(`tag-${slot}`).textContent    = '';
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

  if (!state.modelsReady) ctaHint.textContent = 'Downloading model…';
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
    const slots = ['child', 'parent1', 'parent2'].filter(s => state.images[s]);

    // ── Run detection on each uploaded image (sequential — MediaPipe is stateful)
    for (const slot of slots) {
      const lms = await sendImage(state.images[slot]);
      state.landmarks[slot] = lms;

      if (lms) {
        drawLandmarks(slot, lms);
        const acc = computePhotoAccuracy(lms, state.images[slot]);
        showAccuracyBadge(slot, acc.pct);
        $(`tag-${slot}`).textContent = `✅ ${lms.length} landmarks`;
      } else {
        $(`tag-${slot}`).textContent = '⚠️ No face found';
      }
    }

    // ── Validate required faces ──
    if (!state.landmarks.child) {
      showError("No face detected in the child's photo. Try a clearer, front-facing photo.");
      analyzeBtn.disabled = false;
      return;
    }

    const hasP1 = !!state.landmarks.parent1;
    const hasP2 = !!state.landmarks.parent2;

    if (!hasP1 && !hasP2) {
      showError("No faces detected in the parent photos. Try clearer, front-facing photos.");
      analyzeBtn.disabled = false;
      return;
    }

    // ── Compute scores & render ──
    const accuracies = {};
    slots.forEach(s => {
      if (state.landmarks[s]) accuracies[s] = computePhotoAccuracy(state.landmarks[s], state.images[s]);
    });

    const scores = computeScores(hasP1, hasP2);
    renderResults(scores, accuracies, hasP1, hasP2);
    analyzeBtn.disabled = false;

  } catch (err) {
    console.error(err);
    showError(`Analysis failed: ${err.message}`);
    analyzeBtn.disabled = false;
  }
}

// ── Draw coloured landmark mesh on the photo canvas ──────────────────────────
// MediaPipe landmarks are normalised [0,1] → multiply by canvas dimensions
function drawLandmarks(slot, lms) {
  const canvas = $(`canvas-${slot}`);
  const img    = state.images[slot];
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  ctx.drawImage(img, 0, 0);

  const dotR = Math.max(1.2, W / 500);
  const slotColor = { parent1: '#2563eb', parent2: '#db2777', child: '#059669' };

  // Bounding box from face-oval landmarks
  const ovalPts = REGIONS.faceOval.map(i => lms[i]).filter(Boolean);
  const xs = ovalPts.map(p => p.x * W), ys = ovalPts.map(p => p.y * H);
  const bx = Math.min(...xs), by = Math.min(...ys);
  const bw = Math.max(...xs) - bx, bh = Math.max(...ys) - by;
  ctx.strokeStyle = slotColor[slot] || '#7c3aed';
  ctx.lineWidth   = Math.max(1.5, W / 400);
  ctx.strokeRect(bx, by, bw, bh);

  // Colour-coded regions
  const regionColors = {
    rightEye: '#3b82f6', leftEye: '#3b82f6',
    rightIris: '#1d4ed8', leftIris: '#1d4ed8',
    rightEyebrow: '#8b5cf6', leftEyebrow: '#8b5cf6',
    noseBridge: '#f59e0b', noseTip: '#f59e0b',
    outerLips: '#ef4444', innerLips: '#f87171',
    faceOval: '#10b981', chin: '#059669',
    rightCheek: '#ec4899', leftCheek: '#ec4899',
    forehead: '#14b8a6',
  };

  Object.entries(regionColors).forEach(([region, color]) => {
    ctx.fillStyle = color + 'cc';
    (REGIONS[region] ?? []).forEach(i => {
      const p = lms[i];
      if (!p) return;
      ctx.beginPath();
      ctx.arc(p.x * W, p.y * H, dotR, 0, Math.PI * 2);
      ctx.fill();
    });
  });
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
// 4. ACCURACY — based on face size, frontality and landmark count
// ══════════════════════════════════════════════════════════════════════════════
// MediaPipe landmark coords are normalised [0,1]; aspect ratio differences
// between x and y are minor for portrait photos and don't affect relative scores.
function computePhotoAccuracy(lms, img) {
  // ── Face size (bounding box of face oval in normalised coords) ──
  const ovalPts = REGIONS.faceOval.map(i => lms[i]).filter(Boolean);
  const xs = ovalPts.map(p => p.x), ys = ovalPts.map(p => p.y);
  const faceW = Math.max(...xs) - Math.min(...xs);
  const faceH = Math.max(...ys) - Math.min(...ys);
  const faceArea  = faceW * faceH;              // fraction of image area
  const sizeScore = Math.min(1, faceArea / 0.09); // 9 % fills image = full score

  // ── Eye tilt (roll) ──
  const rEyeC = meanPoint(REGIONS.rightEye.map(i => lms[i]).filter(Boolean));
  const lEyeC = meanPoint(REGIONS.leftEye.map(i => lms[i]).filter(Boolean));
  const tiltDeg = Math.abs(Math.atan2(lEyeC.y - rEyeC.y, lEyeC.x - rEyeC.x) * 180 / Math.PI);
  const rollScore = Math.max(0, 1 - tiltDeg / 25);

  // ── Nose-centering between eyes (yaw) ──
  const noseTip  = lms[4];
  const eyeMidX  = (rEyeC.x + lEyeC.x) / 2;
  const yawNorm  = Math.abs((noseTip?.x ?? eyeMidX) - eyeMidX) / Math.max(0.01, faceW);
  const yawScore = Math.max(0, 1 - yawNorm / 0.18);

  // ── Landmark coverage ──
  const landmarkScore = Math.min(1, lms.length / 468);

  const total = sizeScore * 0.35 + rollScore * 0.25 + yawScore * 0.25 + landmarkScore * 0.15;
  const pct   = Math.round(Math.min(100, total * 100));

  const tips = [];
  if (sizeScore  < 0.5) tips.push('face too small — move closer or crop tighter');
  if (rollScore  < 0.6) tips.push('head is tilted — keep it level');
  if (yawScore   < 0.6) tips.push('face turned to the side — look straight at camera');

  return {
    pct,
    factors: {
      size:      Math.round(sizeScore * 100),
      roll:      Math.round(rollScore * 100),
      yaw:       Math.round(yawScore  * 100),
      landmarks: Math.round(landmarkScore * 100),
    },
    tips,
  };
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. SIMILARITY SCORING
// Normalise landmarks by inter-ocular distance (IOD) so comparisons are
// scale- and translation-invariant.
// ══════════════════════════════════════════════════════════════════════════════
function normaliseLandmarks(lms) {
  const rEyeC = meanPoint(REGIONS.rightEye.map(i => lms[i]).filter(Boolean));
  const lEyeC = meanPoint(REGIONS.leftEye.map(i => lms[i]).filter(Boolean));
  const iod   = dist2d(rEyeC, lEyeC);
  if (iod < 0.005) return lms;                          // degenerate — too small
  const cx = (rEyeC.x + lEyeC.x) / 2;
  const cy = (rEyeC.y + lEyeC.y) / 2;
  return lms.map(p => p ? { x: (p.x - cx) / iod, y: (p.y - cy) / iod } : null);
}

/** Overall similarity using all 468 face-mesh landmarks → 0-100 % */
function overallSim(n1, n2) {
  let total = 0, count = 0;
  const len = Math.min(468, n1.length, n2.length);
  for (let i = 0; i < len; i++) {
    if (!n1[i] || !n2[i]) continue;
    total += dist2d(n1[i], n2[i]);
    count++;
  }
  if (!count) return 0;
  // After IOD-normalisation, avg dist range: 0 (identical) → ~0.14 (unrelated)
  return Math.max(0, Math.min(100, Math.round((1 - total / count / 0.14) * 100)));
}

/** Feature-level similarity for a list of region keys → 0-100 % */
function featureSim(n1, n2, regionKeys) {
  const indices = regionKeys.flatMap(k => REGIONS[k] ?? []);
  let total = 0, count = 0;
  for (const i of indices) {
    if (!n1[i] || !n2[i]) continue;
    total += dist2d(n1[i], n2[i]);
    count++;
  }
  if (!count) return 50;
  return Math.max(0, Math.min(100, Math.round((1 - total / count / 0.20) * 100)));
}

function computeScores(hasP1, hasP2) {
  const childN = normaliseLandmarks(state.landmarks.child);
  const result = { overall: {}, features: FEATURES.map(f => ({ ...f, scores: {} })) };

  ['parent1', 'parent2'].forEach(p => {
    if (!state.landmarks[p]) return;
    const parN = normaliseLandmarks(state.landmarks[p]);
    result.overall[p] = overallSim(childN, parN);
    result.features.forEach(f => { f.scores[p] = featureSim(childN, parN, f.regionKeys); });
  });
  return result;
}

// ── Maths helpers ─────────────────────────────────────────────────────────────
function meanPoint(pts) {
  if (!pts.length) return { x: 0, y: 0 };
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
      <p>Based on <strong>478 MediaPipe facial landmark points</strong> per photo</p>
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

// ── Confidence card ───────────────────────────────────────────────────────────
function buildConfidenceCard(accuracies, hasP1, hasP2) {
  const slots = [
    { key: 'child',   label: '👶 Child' },
    { key: 'parent1', label: '👤 Parent 1', skip: !hasP1 },
    { key: 'parent2', label: '👤 Parent 2', skip: !hasP2 },
  ].filter(s => !s.skip && accuracies[s.key]);

  const minScore = Math.min(...slots.map(s => accuracies[s.key].pct));
  const overallMsg = minScore >= 75 ? '🎯 Results are highly reliable'
                   : minScore >= 50 ? '📋 Results are a reasonable estimate'
                   : '⚠️ Results may be less accurate — see improvement tips below';

  const rows = slots.map(({ key, label }) => {
    const { pct, factors, tips } = accuracies[key];
    const level = pct >= 75 ? 'high' : pct >= 50 ? 'medium' : 'low';
    const badge = pct >= 75 ? '✅ High quality' : pct >= 50 ? '⚠️ Fair quality' : '❌ Low quality';
    const tip   = tips.length ? `💡 ${tips.join(' · ')}` : '✓ Great photo for analysis';
    const factorLine = `Face size: ${factors.size}% · Frontality: ${Math.round((factors.roll + factors.yaw) / 2)}% · Landmarks: ${factors.landmarks}%`;
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

// ── Overall gauge cards ───────────────────────────────────────────────────────
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
  const tag = pct >= 70 ? '🔥 Strong resemblance' : pct >= 45 ? '😊 Good resemblance' : '🌱 Some resemblance';
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
    if (d < 6)                            ties.push(f);
    else if (f.scores.parent1 > f.scores.parent2) p1w.push(f);
    else                                  p2w.push(f);
  });
  const dom = p1w.length > p2w.length ? 'Parent 1' : p2w.length > p1w.length ? 'Parent 2' : 'both parents equally';
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
      <p>Running MediaPipe FaceMesh on all photos</p>
    </div>`;
}

function showError(msg) {
  resultsEl.innerHTML = `<div class="error-box">❌ ${msg}</div>`;
}

// ══════════════════════════════════════════════════════════════════════════════
// START
// ══════════════════════════════════════════════════════════════════════════════
init();
