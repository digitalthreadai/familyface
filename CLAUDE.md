# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Resemble** — a fully client-side PWA that compares facial landmarks between people using AI. Two modes: Family Resemblance (child vs parents) and Celebrity Match (you vs any celebrity). Zero backend. No build step.

## Running

Open `index.html` directly in a browser, or serve it with any static server:

```bash
npx serve .
# or
python -m http.server 8080
```

## Architecture

Four files, no dependencies to install:

- **`index.html`** — shell with navbar, model loading banner, `#app` container (SPA target), and camera modal. Loads `face-api.js` from jsDelivr CDN.
- **`app.js`** — all logic including router, page renderers, camera capture, and analysis engine. Self-contained, no modules/bundler.
- **`style.css`** — dark glass morphism theme. CSS variables drive the entire colour system.
- **`manifest.json`** — PWA manifest for installable mobile app.

### Routing

Hash-based SPA routing: `#landing`, `#family`, `#celebrity`. The `navigate()` function updates `window.location.hash` and calls `render()`, which swaps content inside `#app`. No framework.

### app.js pipeline

1. **`loadModels()`** — downloads three face-api.js neural net weights from jsDelivr (primary) then GitHub CDN (fallback), each with a 25 s timeout.

2. **Router** — `navigate(view)` renders landing, family, or celebrity pages into `#app`. Nav links and hash changes both trigger routing.

3. **Landing page** — hero section, feature cards linking to both modes, privacy banner, footer. All rendered as innerHTML.

4. **Family mode** — three upload slots (parent1, child, parent2) with drag-drop, file input, and camera capture. Analysis runs the full engine and shows confidence card, gauge charts, feature breakdown, and inheritance summary.

5. **Celebrity mode** — two upload slots (you, celebrity). Uses the same landmark engine plus 128-dim Euclidean descriptor distance for a "Neural Similarity" score.

6. **Camera capture** — `getUserMedia` with real-time face-api.js detection loop. Face oval guide SVG, scan line animation, quality assessment (size, roll, yaw). Capture button enables only when face quality passes thresholds.

7. **Analysis engine** (preserved from original):
   - `normaliseLandmarks()`: translates to mid-eye origin, scales by IOD.
   - `featureSim()`: avg Euclidean distance mapped via `exp(-d/0.12)` → 0–100%.
   - `overallFromFeatures()`: weighted average using `FEATURE_WEIGHTS`.
   - `computeAccuracy()`: face size, SSD confidence, head roll, nose yaw → quality score.

### Key constants

| Constant | Location | Purpose |
|---|---|---|
| `FEATURES` | top of app.js | 5 facial regions with their 68-pt landmark index ranges |
| `FEATURE_WEIGHTS` | app.js | eyes 25%, nose 25%, mouth 20%, jawline 15%, eyebrows 15% |
| `MODEL_URLS` | app.js | CDN fallback order for face-api.js weights |

### State structure

```js
state = {
  modelsReady: boolean,
  currentView: 'landing' | 'family' | 'celebrity',
  family: { parent1, parent2, child },  // HTMLImageElement | null
  celeb:  { you, celebrity },           // HTMLImageElement | null
  camera: { stream, slot, mode, detecting },
}
```

### Colour system

Parent 1 = `--p1` (blue), Parent 2 = `--p2` (pink), Child = `--child` (green), Celebrity = `--celeb` (amber), You = `--primary` (purple). CSS classes and JS slot names follow these conventions.

### Fonts

- **Space Grotesk** — headings/display (`--font-display`)
- **Inter** — body text (`--font-body`)
