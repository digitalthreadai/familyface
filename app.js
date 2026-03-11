/* ══════════════════════════════════════════════════════════════════════
   Resemble — app.js
   Engine : face-api.js  (68 facial landmarks + 128-dim descriptor)
   Weights : jsdelivr CDN — no WASM, loads in ~5 s
   Runs 100 % in-browser. Zero data leaves your device.
   ══════════════════════════════════════════════════════════════════════ */

'use strict';

// ── CDN sources tried in order ───────────────────────────────────────────────
const MODEL_URLS = [
  'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights',
  'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights',
];

// ── 68-point landmark index groups ───────────────────────────────────────────
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

const FEATURE_WEIGHTS = {
  eyes:     0.25,
  nose:     0.25,
  mouth:    0.20,
  jawline:  0.15,
  eyebrows: 0.15,
};

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  modelsReady: false,
  currentView: 'landing',
  family: { parent1: null, parent2: null, child: null },
  celeb:  { you: null, celebrity: null },
  camera: { stream: null, slot: null, mode: null, detecting: false },
};

const $ = id => document.getElementById(id);
const appEl     = $('app');
const modelBanner = $('model-banner');
const modelText   = $('model-status-text');

// ══════════════════════════════════════════════════════════════════════════════
// 1. MODEL LOADING
// ══════════════════════════════════════════════════════════════════════════════
async function loadModels() {
  modelBanner.classList.remove('ready', 'hidden');
  const spinnerEl = modelBanner.querySelector('.spinner');
  if (!spinnerEl) {
    const s = document.createElement('div');
    s.className = 'spinner';
    modelBanner.insertBefore(s, modelText);
  }
  state.modelsReady = false;

  const TIMEOUT_MS = 25_000;

  for (let i = 0; i < MODEL_URLS.length; i++) {
    const url = MODEL_URLS[i];
    const label = i === 0 ? 'primary server' : `backup server ${i}`;
    try {
      modelText.textContent = `[1/3] Loading face detector (${label})…`;
      await loadOneWithTimeout(faceapi.nets.ssdMobilenetv1, url, TIMEOUT_MS);

      modelText.textContent = `[2/3] Loading landmark model (${label})…`;
      await loadOneWithTimeout(faceapi.nets.faceLandmark68Net, url, TIMEOUT_MS);

      modelText.textContent = `[3/3] Loading recognition model (${label})…`;
      await loadOneWithTimeout(faceapi.nets.faceRecognitionNet, url, TIMEOUT_MS);

      state.modelsReady = true;
      modelBanner.classList.add('ready');
      const sp = modelBanner.querySelector('.spinner');
      if (sp) sp.outerHTML = '<span style="font-size:1.1rem">✅</span>';
      modelText.textContent = 'AI models ready — start comparing faces';
      setTimeout(() => modelBanner.classList.add('hidden'), 4000);
      return;

    } catch (err) {
      console.warn(`CDN ${label} failed:`, err.message);
      if (i < MODEL_URLS.length - 1) {
        modelText.textContent = `Server ${i + 1} timed out — trying next…`;
        await new Promise(r => setTimeout(r, 600));
      }
    }
  }

  const sp = modelBanner.querySelector('.spinner');
  if (sp) sp.outerHTML = '<span style="font-size:1.1rem">❌</span>';
  modelText.innerHTML =
    'Could not load models. Check your internet connection and '
    + '<button id="retry-btn" style="'
    + 'background:#7c3aed;color:#fff;border:none;border-radius:6px;'
    + 'padding:3px 12px;cursor:pointer;font-weight:700;margin-left:6px'
    + '" onclick="loadModels()">Retry</button>';
}

function loadOneWithTimeout(net, url, ms) {
  return Promise.race([
    net.loadFromUri(url),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`timed out after ${ms / 1000} s`)), ms)
    ),
  ]);
}

// ══════════════════════════════════════════════════════════════════════════════
// 2. ROUTER
// ══════════════════════════════════════════════════════════════════════════════
function navigate(view) {
  state.currentView = view;
  window.location.hash = view === 'landing' ? '' : view;
  render();
  window.scrollTo(0, 0);
}

function render() {
  // Update nav links
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.toggle('active', link.dataset.nav === state.currentView);
  });

  switch (state.currentView) {
    case 'family':    renderFamilyPage(); break;
    case 'celebrity': renderCelebrityPage(); break;
    default:          renderLandingPage(); break;
  }
}

// Nav link clicks
document.querySelectorAll('[data-nav]').forEach(el => {
  el.addEventListener('click', e => {
    e.preventDefault();
    navigate(el.dataset.nav === 'landing' ? 'landing' : el.dataset.nav);
  });
});

// Hash change
window.addEventListener('hashchange', () => {
  const hash = window.location.hash.replace('#', '') || 'landing';
  if (hash !== state.currentView) {
    state.currentView = hash;
    render();
  }
});

// ══════════════════════════════════════════════════════════════════════════════
// 3. LANDING PAGE
// ══════════════════════════════════════════════════════════════════════════════
function renderLandingPage() {
  appEl.innerHTML = `
    <div class="landing">
      <section class="hero">
        <div class="hero-bg"></div>
        <div class="hero-badge">
          <span class="hero-badge-dot"></span>
          AI-Powered Face Analysis
        </div>
        <h1>Discover Your <span class="gradient-text">Facial Resemblance</span></h1>
        <p class="hero-sub">
          Compare facial features with family members or celebrities using advanced
          AI landmark detection. 68-point analysis, 100% private — nothing leaves your device.
        </p>
        <div class="hero-actions">
          <button class="btn btn-primary" onclick="navigate('family')">
            👨‍👩‍👧 Family Comparison
          </button>
          <button class="btn btn-secondary" onclick="navigate('celebrity')">
            ⭐ Celebrity Match
          </button>
        </div>
        <div class="hero-stats">
          <div class="hero-stat">
            <div class="hero-stat-num">68</div>
            <div class="hero-stat-label">Facial Landmarks</div>
          </div>
          <div class="hero-stat">
            <div class="hero-stat-num">5</div>
            <div class="hero-stat-label">Feature Regions</div>
          </div>
          <div class="hero-stat">
            <div class="hero-stat-num">128D</div>
            <div class="hero-stat-label">Face Descriptor</div>
          </div>
          <div class="hero-stat">
            <div class="hero-stat-num">0</div>
            <div class="hero-stat-label">Data Uploaded</div>
          </div>
        </div>
      </section>

      <section class="features-section">
        <h2 class="features-title">Choose Your Comparison</h2>
        <p class="features-subtitle">Two powerful ways to discover facial resemblance</p>
        <div class="feature-cards-grid">
          <a class="feature-card" href="#family" onclick="event.preventDefault();navigate('family')">
            <div class="feature-card-icon family">👨‍👩‍👧</div>
            <div class="feature-card-title">Family Resemblance</div>
            <div class="feature-card-desc">
              Upload photos of a child and up to two parents. See which parent
              each feature comes from with detailed breakdown and inheritance summary.
            </div>
            <div class="feature-card-cta">Get Started →</div>
          </a>
          <a class="feature-card" href="#celebrity" onclick="event.preventDefault();navigate('celebrity')">
            <div class="feature-card-icon celebrity">⭐</div>
            <div class="feature-card-title">Celebrity Match</div>
            <div class="feature-card-desc">
              Compare your face with any celebrity. Upload both photos and discover
              your resemblance percentage with feature-by-feature analysis.
            </div>
            <div class="feature-card-cta">Try It →</div>
          </a>
        </div>
      </section>

      <section class="privacy-banner">
        <div class="privacy-items">
          <div class="privacy-item">
            <span class="privacy-icon">🔒</span>
            <span>100% Private</span>
          </div>
          <div class="privacy-item">
            <span class="privacy-icon">📱</span>
            <span>Works Offline</span>
          </div>
          <div class="privacy-item">
            <span class="privacy-icon">⚡</span>
            <span>Instant Results</span>
          </div>
          <div class="privacy-item">
            <span class="privacy-icon">🧠</span>
            <span>AI-Powered</span>
          </div>
        </div>
      </section>

      <footer class="app-footer">
        Built with face-api.js · <a href="https://github.com/digitalthreadai">DigitalThread AI</a>
      </footer>
    </div>
  `;
}

// ══════════════════════════════════════════════════════════════════════════════
// 4. FAMILY COMPARISON PAGE
// ══════════════════════════════════════════════════════════════════════════════
function renderFamilyPage() {
  appEl.innerHTML = `
    <div class="compare-page">
      <div class="page-header">
        <a href="#" class="page-back" onclick="event.preventDefault();navigate('landing')">←</a>
        <div>
          <div class="page-title">Family Resemblance</div>
          <div class="page-subtitle">Upload a child and up to two parents to compare</div>
        </div>
      </div>

      <div class="upload-grid family-grid">
        ${uploadCardHTML('parent1', 'Parent 1', 'p1', '👤', false, 'family')}
        ${uploadCardHTML('child', 'Child', 'child', '👶', false, 'family')}
        ${uploadCardHTML('parent2', 'Parent 2', 'p2', '👤', true, 'family')}
      </div>

      <div class="cta-row">
        <div class="btn-group">
          <button class="btn btn-primary" id="analyze-family-btn" disabled>
            🔬 Analyze Resemblance
          </button>
          <button class="btn btn-ghost" id="reset-family-btn" disabled>
            ↺ Reset
          </button>
        </div>
        <div class="cta-hint" id="family-hint">Upload photos to get started</div>
      </div>

      <div id="family-results" class="results-section" style="display:none"></div>
    </div>
  `;

  bindUploadSlots(['parent1', 'parent2', 'child'], 'family');
  $('analyze-family-btn').addEventListener('click', () => runFamilyAnalysis());
  $('reset-family-btn').addEventListener('click', () => resetMode('family'));
  updateFamilyButtons();
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. CELEBRITY COMPARISON PAGE
// ══════════════════════════════════════════════════════════════════════════════
function renderCelebrityPage() {
  appEl.innerHTML = `
    <div class="compare-page">
      <div class="page-header">
        <a href="#" class="page-back" onclick="event.preventDefault();navigate('landing')">←</a>
        <div>
          <div class="page-title">Celebrity Match</div>
          <div class="page-subtitle">Compare your face with any celebrity</div>
        </div>
      </div>

      <div class="upload-grid celeb-grid">
        ${uploadCardHTML('you', 'Your Photo', 'you', '🧑', false, 'celeb')}
        ${uploadCardHTML('celebrity', 'Celebrity', 'celeb', '⭐', false, 'celeb')}
      </div>

      <div class="cta-row">
        <div class="btn-group">
          <button class="btn btn-primary" id="analyze-celeb-btn" disabled>
            ⭐ Compare Faces
          </button>
          <button class="btn btn-ghost" id="reset-celeb-btn" disabled>
            ↺ Reset
          </button>
        </div>
        <div class="cta-hint" id="celeb-hint">Upload both photos to compare</div>
      </div>

      <div id="celeb-results" class="results-section" style="display:none"></div>
    </div>
  `;

  bindUploadSlots(['you', 'celebrity'], 'celeb');
  $('analyze-celeb-btn').addEventListener('click', () => runCelebAnalysis());
  $('reset-celeb-btn').addEventListener('click', () => resetMode('celeb'));
  updateCelebButtons();
}

// ══════════════════════════════════════════════════════════════════════════════
// 6. SHARED UPLOAD CARD HTML
// ══════════════════════════════════════════════════════════════════════════════
function uploadCardHTML(slot, label, colorClass, emoji, optional, mode) {
  const highlight = slot === 'child' || slot === 'you' ? ' highlight' : '';
  return `
    <div class="upload-card${highlight}" id="card-${slot}">
      <div class="card-label">
        <span class="dot dot-${colorClass}"></span>
        <span class="label-${colorClass}">${label}</span>
        ${optional ? '<span class="optional-badge">Optional</span>' : ''}
      </div>

      <div class="drop-zone" id="drop-${slot}">
        <input type="file" accept="image/*" class="file-input" id="input-${slot}" />
        <div class="drop-icon">${emoji}</div>
        <div class="drop-text">Drop photo or tap to upload</div>
        <div class="drop-hint">Front-facing works best</div>
        <div class="drop-actions">
          <button class="drop-btn" onclick="event.stopPropagation();openCamera('${slot}','${mode}')">📷 Camera</button>
        </div>
      </div>

      <div class="photo-wrap" id="wrap-${slot}" style="display:none">
        <canvas class="photo-canvas" id="canvas-${slot}"></canvas>
        <div class="face-tag" id="tag-${slot}"></div>
        <button class="remove-btn" data-slot="${slot}" data-mode="${mode}">×</button>
      </div>
    </div>
  `;
}

// ── Bind upload events ───────────────────────────────────────────────────────
function bindUploadSlots(slots, mode) {
  slots.forEach(slot => {
    const drop  = $(`drop-${slot}`);
    const input = $(`input-${slot}`);

    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
    drop.addEventListener('drop', e => {
      e.preventDefault();
      drop.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f?.type.startsWith('image/')) loadImageFile(slot, f, mode);
    });
    input.addEventListener('change', () => {
      if (input.files[0]) loadImageFile(slot, input.files[0], mode);
    });
  });

  // Remove buttons
  appEl.querySelectorAll(`.remove-btn[data-mode="${mode}"]`).forEach(btn =>
    btn.addEventListener('click', () => clearSlot(btn.dataset.slot, btn.dataset.mode))
  );
}

function loadImageFile(slot, file, mode) {
  const reader = new FileReader();
  reader.onload = ev => {
    const img = new Image();
    img.onload = () => {
      state[mode][slot] = img;
      renderPreview(slot);
      updateModeButtons(mode);
      const resultsEl = $(mode === 'family' ? 'family-results' : 'celeb-results');
      if (resultsEl) resultsEl.style.display = 'none';
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

function renderPreview(slot) {
  const mode = state.family[slot] !== undefined ? 'family' : 'celeb';
  const img  = state[mode][slot];
  if (!img) return;
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

function clearSlot(slot, mode) {
  state[mode][slot] = null;
  $(`wrap-${slot}`).style.display  = 'none';
  $(`drop-${slot}`).style.display  = 'flex';
  $(`input-${slot}`).value         = '';
  $(`tag-${slot}`).textContent     = '';
  const old = $(`acc-badge-${slot}`);
  if (old) old.remove();
  updateModeButtons(mode);
  const resultsEl = $(mode === 'family' ? 'family-results' : 'celeb-results');
  if (resultsEl) resultsEl.style.display = 'none';
}

function resetMode(mode) {
  const slots = mode === 'family' ? ['parent1', 'parent2', 'child'] : ['you', 'celebrity'];
  slots.forEach(s => clearSlot(s, mode));
}

function updateModeButtons(mode) {
  if (mode === 'family') updateFamilyButtons();
  else updateCelebButtons();
}

function updateFamilyButtons() {
  const btn = $('analyze-family-btn');
  const rst = $('reset-family-btn');
  const hint = $('family-hint');
  if (!btn) return;

  const hasChild  = !!state.family.child;
  const hasParent = !!(state.family.parent1 || state.family.parent2);
  const hasAny    = hasChild || hasParent;

  btn.disabled = !(state.modelsReady && hasChild && hasParent);
  rst.disabled = !hasAny;

  if (!state.modelsReady)  hint.textContent = 'Downloading AI models…';
  else if (!hasChild)      hint.textContent = "Upload the child's photo to continue";
  else if (!hasParent)     hint.textContent = "Upload at least one parent's photo";
  else                     hint.textContent = 'Ready — click Analyze Resemblance';
}

function updateCelebButtons() {
  const btn = $('analyze-celeb-btn');
  const rst = $('reset-celeb-btn');
  const hint = $('celeb-hint');
  if (!btn) return;

  const hasYou   = !!state.celeb.you;
  const hasCeleb = !!state.celeb.celebrity;
  const hasAny   = hasYou || hasCeleb;

  btn.disabled = !(state.modelsReady && hasYou && hasCeleb);
  rst.disabled = !hasAny;

  if (!state.modelsReady)  hint.textContent = 'Downloading AI models…';
  else if (!hasYou)        hint.textContent = 'Upload your photo to continue';
  else if (!hasCeleb)      hint.textContent = "Upload the celebrity's photo";
  else                     hint.textContent = 'Ready — click Compare Faces';
}

// ══════════════════════════════════════════════════════════════════════════════
// 7. CAMERA CAPTURE
// ══════════════════════════════════════════════════════════════════════════════
const cameraModal   = $('camera-modal');
const cameraVideo   = $('camera-video');
const cameraOverlay = $('camera-overlay');
const cameraStatus  = $('camera-status');
const cameraQuality = $('camera-quality');
const captureBtn    = $('camera-capture-btn');
const cameraClose   = $('camera-close');
const scanLine      = $('scan-line');

async function openCamera(slot, mode) {
  state.camera.slot = slot;
  state.camera.mode = mode;
  captureBtn.disabled = true;
  captureBtn.classList.remove('ready');
  cameraStatus.textContent = 'Starting camera…';
  cameraQuality.textContent = '';
  cameraModal.style.display = 'flex';

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
    state.camera.stream = stream;
    cameraVideo.srcObject = stream;
    await cameraVideo.play();

    // Set overlay canvas to match video
    cameraOverlay.width  = cameraVideo.videoWidth;
    cameraOverlay.height = cameraVideo.videoHeight;

    cameraStatus.textContent = 'Position your face within the oval';
    scanLine.classList.add('active');

    // Start face detection loop
    state.camera.detecting = true;
    detectFaceLoop();

  } catch (err) {
    console.error('Camera error:', err);
    cameraStatus.textContent = 'Camera access denied. Please allow camera permissions.';
  }
}

async function detectFaceLoop() {
  if (!state.camera.detecting) return;

  const ctx = cameraOverlay.getContext('2d');
  ctx.clearRect(0, 0, cameraOverlay.width, cameraOverlay.height);

  if (state.modelsReady && cameraVideo.readyState >= 2) {
    try {
      const det = await faceapi
        .detectSingleFace(cameraVideo, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.3 }))
        .withFaceLandmarks();

      if (det) {
        // Draw landmarks on overlay
        const pts = det.landmarks.positions;
        const dotR = 3;

        const colors = [
          { idx: range(0,17),  color: '#10b981' },
          { idx: range(17,27), color: '#8b5cf6' },
          { idx: range(27,36), color: '#f59e0b' },
          { idx: range(36,48), color: '#3b82f6' },
          { idx: range(48,68), color: '#ef4444' },
        ];
        colors.forEach(({ idx, color }) => {
          ctx.fillStyle = color;
          idx.forEach(i => {
            const p = pts[i];
            if (!p) return;
            ctx.beginPath();
            ctx.arc(p.x, p.y, dotR, 0, Math.PI * 2);
            ctx.fill();
          });
        });

        // Assess quality
        const box     = det.detection.box;
        const vidArea = cameraVideo.videoWidth * cameraVideo.videoHeight;
        const faceArea = box.width * box.height;
        const sizeOk  = faceArea / vidArea > 0.04;

        const rEyeC = centroid(pts.slice(36, 42));
        const lEyeC = centroid(pts.slice(42, 48));
        const tiltDeg = Math.abs(Math.atan2(lEyeC.y - rEyeC.y, lEyeC.x - rEyeC.x) * 180 / Math.PI);
        const rollOk = tiltDeg < 15;

        const noseTip = pts[30];
        const eyeMidX = (rEyeC.x + lEyeC.x) / 2;
        const yawNorm = Math.abs(noseTip.x - eyeMidX) / Math.max(1, box.width);
        const yawOk   = yawNorm < 0.12;

        const ready = sizeOk && rollOk && yawOk && det.detection.score > 0.5;

        captureBtn.disabled = !ready;
        captureBtn.classList.toggle('ready', ready);

        if (ready) {
          cameraStatus.textContent = 'Face detected — tap to capture';
          cameraQuality.textContent = `Quality: ${Math.round(det.detection.score * 100)}% confidence`;
        } else {
          const tips = [];
          if (!sizeOk) tips.push('move closer');
          if (!rollOk) tips.push('keep head level');
          if (!yawOk)  tips.push('look straight at camera');
          cameraStatus.textContent = 'Adjusting… ' + tips.join(', ');
          cameraQuality.textContent = '';
        }
      } else {
        captureBtn.disabled = true;
        captureBtn.classList.remove('ready');
        cameraStatus.textContent = 'No face detected — make sure your face is visible';
        cameraQuality.textContent = '';
      }
    } catch (err) {
      console.warn('Detection error:', err);
    }
  }

  requestAnimationFrame(detectFaceLoop);
}

function captureFrame() {
  if (captureBtn.disabled) return;

  const canvas = document.createElement('canvas');
  canvas.width  = cameraVideo.videoWidth;
  canvas.height = cameraVideo.videoHeight;
  const ctx = canvas.getContext('2d');
  // Mirror the capture to match the preview
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(cameraVideo, 0, 0);

  const img = new Image();
  img.onload = () => {
    const slot = state.camera.slot;
    const mode = state.camera.mode;
    state[mode][slot] = img;
    closeCamera();
    renderPreview(slot);
    updateModeButtons(mode);
  };
  img.src = canvas.toDataURL('image/jpeg', 0.92);
}

function closeCamera() {
  state.camera.detecting = false;
  scanLine.classList.remove('active');
  if (state.camera.stream) {
    state.camera.stream.getTracks().forEach(t => t.stop());
    state.camera.stream = null;
  }
  cameraVideo.srcObject = null;
  cameraModal.style.display = 'none';
  const ctx = cameraOverlay.getContext('2d');
  ctx.clearRect(0, 0, cameraOverlay.width, cameraOverlay.height);
}

captureBtn.addEventListener('click', captureFrame);
cameraClose.addEventListener('click', closeCamera);

// ══════════════════════════════════════════════════════════════════════════════
// 8. FAMILY ANALYSIS
// ══════════════════════════════════════════════════════════════════════════════
async function runFamilyAnalysis() {
  const btn = $('analyze-family-btn');
  const resultsEl = $('family-results');
  btn.disabled = true;
  resultsEl.style.display = 'block';
  resultsEl.innerHTML = analyzingHTML();
  setTimeout(() => resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);

  try {
    const detections = {};
    const slots = ['child', 'parent1', 'parent2'].filter(s => state.family[s]);

    for (const slot of slots) {
      const det = await faceapi
        .detectSingleFace(state.family[slot], new faceapi.SsdMobilenetv1Options({ minConfidence: 0.3 }))
        .withFaceLandmarks()
        .withFaceDescriptor();
      detections[slot] = det || null;
    }

    if (!detections.child) {
      showError(resultsEl, "No face detected in the child's photo. Try a clearer, front-facing photo.");
      btn.disabled = false;
      return;
    }

    const hasP1 = !!detections.parent1;
    const hasP2 = !!detections.parent2;

    if (!hasP1 && !hasP2) {
      showError(resultsEl, "No faces detected in parent photos. Try clearer, front-facing photos.");
      btn.disabled = false;
      return;
    }

    const accuracies = {};
    for (const slot of slots) {
      const det = detections[slot];
      if (det) {
        drawLandmarks(slot, det, 'family');
        accuracies[slot] = computeAccuracy(det, state.family[slot]);
        showAccuracyBadge(slot, accuracies[slot].pct);
        $(`tag-${slot}`).textContent = '✅ 68 landmarks';
      } else {
        $(`tag-${slot}`).textContent = '⚠️ No face found';
      }
    }

    const scores = computeScores(detections, hasP1, hasP2);
    renderFamilyResults(resultsEl, scores, accuracies, hasP1, hasP2);
    btn.disabled = false;

  } catch (err) {
    console.error(err);
    showError(resultsEl, `Analysis failed: ${err.message}`);
    btn.disabled = false;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 9. CELEBRITY ANALYSIS
// ══════════════════════════════════════════════════════════════════════════════
async function runCelebAnalysis() {
  const btn = $('analyze-celeb-btn');
  const resultsEl = $('celeb-results');
  btn.disabled = true;
  resultsEl.style.display = 'block';
  resultsEl.innerHTML = analyzingHTML();
  setTimeout(() => resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);

  try {
    const youDet = await faceapi
      .detectSingleFace(state.celeb.you, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.3 }))
      .withFaceLandmarks()
      .withFaceDescriptor();

    const celebDet = await faceapi
      .detectSingleFace(state.celeb.celebrity, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.3 }))
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!youDet) {
      showError(resultsEl, "No face detected in your photo. Try a clearer, front-facing photo.");
      btn.disabled = false;
      return;
    }
    if (!celebDet) {
      showError(resultsEl, "No face detected in the celebrity photo. Try a clearer photo.");
      btn.disabled = false;
      return;
    }

    // Draw landmarks
    drawLandmarks('you', youDet, 'celeb');
    drawLandmarks('celebrity', celebDet, 'celeb');

    const accYou   = computeAccuracy(youDet, state.celeb.you);
    const accCeleb = computeAccuracy(celebDet, state.celeb.celebrity);
    showAccuracyBadge('you', accYou.pct);
    showAccuracyBadge('celebrity', accCeleb.pct);
    $('tag-you').textContent       = '✅ 68 landmarks';
    $('tag-celebrity').textContent  = '✅ 68 landmarks';

    // Compute scores using the same engine
    const youNorm   = normaliseLandmarks(youDet.landmarks.positions);
    const celebNorm = normaliseLandmarks(celebDet.landmarks.positions);

    const featureScores = FEATURES.map(f => ({
      ...f,
      score: featureSim(youNorm, celebNorm, f.indices),
    }));

    // Overall = weighted average
    let total = 0, wSum = 0;
    featureScores.forEach(f => {
      const w = FEATURE_WEIGHTS[f.key] ?? 0.20;
      total += f.score * w;
      wSum  += w;
    });
    const overall = wSum > 0 ? Math.round(total / wSum) : 0;

    // Descriptor distance (128-dim Euclidean) as extra data
    const descDist = faceapi.euclideanDistance(youDet.descriptor, celebDet.descriptor);
    const descSim  = Math.max(0, Math.round((1 - descDist / 1.5) * 100));

    renderCelebResults(resultsEl, overall, descSim, featureScores,
      { you: accYou, celebrity: accCeleb });
    btn.disabled = false;

  } catch (err) {
    console.error(err);
    showError(resultsEl, `Analysis failed: ${err.message}`);
    btn.disabled = false;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 10. SIMILARITY ENGINE (preserved exactly)
// ══════════════════════════════════════════════════════════════════════════════
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

function featureSim(childNorm, parentNorm, indices) {
  let total = 0;
  for (const i of indices) {
    total += dist2d(childNorm[i], parentNorm[i]);
  }
  const avgDist = total / indices.length;
  return Math.max(0, Math.min(100, Math.round(100 * Math.exp(-avgDist / 0.12))));
}

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

  ['parent1', 'parent2'].forEach(p => {
    if (!detections[p]) return;
    const parNorm = normaliseLandmarks(detections[p].landmarks.positions);
    result.features.forEach(f => {
      f.scores[p] = featureSim(childNorm, parNorm, f.indices);
    });
  });

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
function range(start, end) {
  return Array.from({ length: end - start }, (_, i) => start + i);
}

// ══════════════════════════════════════════════════════════════════════════════
// 11. ACCURACY SCORING (preserved exactly)
// ══════════════════════════════════════════════════════════════════════════════
function computeAccuracy(det, img) {
  const box  = det.detection.box;
  const pts  = det.landmarks.positions;
  const conf = det.detection.score;

  const imgArea   = img.naturalWidth * img.naturalHeight;
  const faceArea  = box.width * box.height;
  const sizeScore = Math.min(1, faceArea / imgArea / 0.09);

  const rEyeC = centroid(pts.slice(36, 42));
  const lEyeC = centroid(pts.slice(42, 48));
  const tiltDeg = Math.abs(Math.atan2(lEyeC.y - rEyeC.y, lEyeC.x - rEyeC.x) * 180 / Math.PI);
  const rollScore = Math.max(0, 1 - tiltDeg / 25);

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
// 12. DRAW LANDMARKS ON CANVAS
// ══════════════════════════════════════════════════════════════════════════════
function drawLandmarks(slot, det, mode) {
  const canvas = $(`canvas-${slot}`);
  const img    = state[mode][slot];
  const ctx    = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  const box  = det.detection.box;
  const pts  = det.landmarks.positions;
  const dotR = Math.max(1.5, img.naturalWidth / 380);
  const lw   = Math.max(1.5, img.naturalWidth / 400);

  const slotColor = {
    parent1: '#2563eb', parent2: '#db2777', child: '#059669',
    you: '#7c3aed', celebrity: '#f59e0b'
  };
  const primary = slotColor[slot] || '#7c3aed';

  ctx.strokeStyle = primary;
  ctx.lineWidth   = lw * 1.6;
  ctx.strokeRect(box.x, box.y, box.width, box.height);

  const regionColors = [
    { color: '#10b981', idx: range(0,  17) },
    { color: '#8b5cf6', idx: range(17, 27) },
    { color: '#f59e0b', idx: range(27, 36) },
    { color: '#3b82f6', idx: range(36, 48) },
    { color: '#ef4444', idx: range(48, 60) },
    { color: '#f87171', idx: range(60, 68) },
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

function showAccuracyBadge(slot, pct) {
  const wrap  = $(`wrap-${slot}`);
  if (!wrap) return;
  const level = pct >= 75 ? 'high' : pct >= 50 ? 'medium' : 'low';
  const icon  = pct >= 75 ? '✅' : pct >= 50 ? '⚠️' : '❌';
  const el    = document.createElement('div');
  el.id        = `acc-badge-${slot}`;
  el.className = `acc-photo-badge ${level}`;
  el.textContent = `${icon} ${pct}% quality`;
  wrap.appendChild(el);
}

// ══════════════════════════════════════════════════════════════════════════════
// 13. RENDER FAMILY RESULTS
// ══════════════════════════════════════════════════════════════════════════════
function renderFamilyResults(el, scores, accuracies, hasP1, hasP2) {
  el.innerHTML = `
    <div class="results-header">
      <h2>Family Resemblance Report</h2>
      <p>Overall scores are a <strong>weighted average of the 5 feature scores</strong> below</p>
    </div>
    ${buildConfidenceCard(accuracies, hasP1, hasP2, 'family')}
    ${buildFamilyGauges(scores.overall, hasP1, hasP2)}
    ${buildFeatureBreakdown(scores.features, hasP1, hasP2)}
    ${buildInheritanceSummary(scores.features, hasP1, hasP2)}
  `;
  animateResults(el);
  animateFamilyGauges(scores.overall, hasP1, hasP2);
}

function buildFamilyGauges(overall, hasP1, hasP2) {
  let h = '<div class="score-cards">';
  if (hasP1) h += gaugeHTML('p1', 'Child vs Parent 1', overall.parent1 ?? 0, 'p1');
  if (hasP1 && hasP2) {
    const avg = Math.round(((overall.parent1 ?? 0) + (overall.parent2 ?? 0)) / 2);
    h += gaugeHTML('neutral', 'Family Blend', avg, 'neutral');
  }
  if (hasP2) h += gaugeHTML('p2', 'Child vs Parent 2', overall.parent2 ?? 0, 'p2');
  return h + '</div>';
}

function animateFamilyGauges(overall, hasP1, hasP2) {
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

// ══════════════════════════════════════════════════════════════════════════════
// 14. RENDER CELEBRITY RESULTS
// ══════════════════════════════════════════════════════════════════════════════
function renderCelebResults(el, overall, descSim, featureScores, accuracies) {
  const tag = overall >= 65 ? '🔥 Strong resemblance'
            : overall >= 45 ? '😊 Good resemblance'
            : '🌱 Some resemblance';

  el.innerHTML = `
    <div class="results-header">
      <h2>Celebrity Match Report</h2>
      <p>Landmark-based analysis with 128-dimension face descriptor comparison</p>
    </div>
    ${buildCelebConfidenceCard(accuracies)}
    <div class="score-cards">
      ${gaugeHTML('celeb-main', 'Landmark Similarity', overall, 'celeb')}
      ${gaugeHTML('celeb-desc', 'Neural Similarity', descSim, 'celeb')}
    </div>
    ${buildCelebFeatureBreakdown(featureScores)}
    <div class="inheritance-card">
      <h3>⭐ Match Summary</h3>
      <p style="margin-bottom:18px;opacity:.85">
        Your overall resemblance is <strong>${overall}%</strong> based on facial landmarks
        and <strong>${descSim}%</strong> based on the neural face descriptor. ${tag}
      </p>
      <div class="inheritance-pills">
        ${featureScores.map(f => {
          const cls = f.score >= 60 ? 'p1' : f.score >= 40 ? 'unique' : 'p2';
          const label = f.score >= 60 ? 'Similar' : f.score >= 40 ? 'Moderate' : 'Different';
          return `<span class="pill ${cls}">${f.emoji} ${f.label}: ${f.score}% (${label})</span>`;
        }).join('')}
      </div>
    </div>
  `;

  animateResults(el);

  // Animate celeb gauges
  const C = 283;
  const go = (id, pct) => {
    const arc = document.getElementById(`arc-${id}`);
    const pctEl = document.getElementById(`pct-${id}`);
    if (!arc || !pctEl) return;
    arc.style.strokeDashoffset = C - (C * pct) / 100;
    let v = 0;
    const tick = () => { v = Math.min(v + 2, pct); pctEl.textContent = v + '%'; if (v < pct) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  };
  go('celeb-main', overall);
  go('celeb-desc', descSim);
}

function buildCelebConfidenceCard(accuracies) {
  const slots = [
    { key: 'you',       label: '🧑 You' },
    { key: 'celebrity', label: '⭐ Celebrity' },
  ].filter(s => accuracies[s.key]);

  const minScore = Math.min(...slots.map(s => accuracies[s.key].pct));
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

function buildCelebFeatureBreakdown(featureScores) {
  let h = `<div class="features-card-result"><h3>Feature-by-Feature Breakdown</h3>`;
  featureScores.forEach((f, i) => {
    if (i > 0) h += '<hr class="feature-divider"/>';
    h += `
      <div class="feature-row">
        <div class="feature-name">
          <span class="feature-emoji">${f.emoji}</span>
          <div><div>${f.label}</div><div class="feature-note">${f.note}</div></div>
        </div>
        <div class="feature-bars">
          ${barRow('celeb', 'Match', f.score)}
        </div>
      </div>`;
  });
  return h + '</div>';
}

// ══════════════════════════════════════════════════════════════════════════════
// 15. SHARED RESULT RENDERING
// ══════════════════════════════════════════════════════════════════════════════
function gaugeHTML(id, label, pct, arc) {
  const tag = pct >= 65 ? '🔥 Strong resemblance' : pct >= 45 ? '😊 Good resemblance' : '🌱 Some resemblance';
  return `
    <div class="score-card ${arc}-card">
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

function buildConfidenceCard(accuracies, hasP1, hasP2, mode) {
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

function buildFeatureBreakdown(features, hasP1, hasP2) {
  let h = `<div class="features-card-result"><h3>Feature-by-Feature Breakdown</h3>`;
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

// ── Animate results (bar fills) ──────────────────────────────────────────────
function animateResults(container) {
  requestAnimationFrame(() => requestAnimationFrame(() => {
    container.querySelectorAll('[data-fill-pct]').forEach(el => {
      el.style.width = el.dataset.fillPct + '%';
    });
  }));
}

// ── UI Helpers ───────────────────────────────────────────────────────────────
function analyzingHTML() {
  return `
    <div class="analyzing-overlay">
      <div class="spinner" style="width:48px;height:48px;border-width:4px;color:var(--primary-2)"></div>
      <h3>Analyzing faces…</h3>
      <p>Detecting 68 landmarks and computing face descriptors</p>
    </div>`;
}
function showError(el, msg) {
  el.innerHTML = `<div class="error-box">❌ ${msg}</div>`;
}

// ══════════════════════════════════════════════════════════════════════════════
// START
// ══════════════════════════════════════════════════════════════════════════════
(function init() {
  // Determine initial view from hash
  const hash = window.location.hash.replace('#', '') || 'landing';
  state.currentView = ['family', 'celebrity'].includes(hash) ? hash : 'landing';
  render();
  loadModels();
})();
