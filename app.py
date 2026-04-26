

"""
GarbageAI — Live Waste Classifier
Calls Groq API directly from browser JS — no Python backend needed for inference.

Install:
    pip install streamlit

Run:
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"

st.set_page_config(
    page_title="♻️ GarbageAI Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
</style>
""", unsafe_allow_html=True)

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Space Grotesk', sans-serif; }

body {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1421 50%, #0a1628 100%);
    color: #e8edf5; min-height: 100vh; padding: 1.2rem 1.5rem;
}

.title-bar {
    background: linear-gradient(90deg, #1a2744, #0d1f3c);
    border: 1px solid #2a4080; border-radius: 16px;
    padding: 1rem 1.5rem; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.title-bar h1 {
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(90deg, #4ECDC4, #45B7D1, #96CEB4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.title-bar p { color: #8899bb; font-size: 0.82rem; margin-top: 2px; }
.status-badge {
    margin-left: auto; display: inline-flex; align-items: center; gap: 6px;
    background: #0a1a0a; border: 1px solid #1a4a1a;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.72rem; color: #4dbb4d; font-weight: 600;
}
.pulse {
    width: 7px; height: 7px; border-radius: 50%; background: #4dbb4d;
    animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.8)} }

.main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }

.panel-header {
    background: #1a2744; padding: 8px 14px;
    display: flex; align-items: center; gap: 6px;
    font-size: 0.75rem; font-weight: 600; color: #8899bb;
    text-transform: uppercase; letter-spacing: .08em;
    border-radius: 12px 12px 0 0; border: 1.5px solid #1e3060; border-bottom: 0;
}
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
.dot-r{background:#ff5f57} .dot-y{background:#febc2e} .dot-g{background:#28c840}

.cam-wrap {
    border: 1.5px solid #1e3060; border-top: 0;
    border-radius: 0 0 12px 12px; overflow: hidden; position: relative;
    background: #000;
}
#video {
    width: 100%; max-height: 340px; object-fit: cover;
    display: block; background: #000;
}
#canvas { display: none; }
.scan-badge {
    position: absolute; top: 10px; right: 10px;
    background: rgba(10,14,26,.88); border: 1px solid #2a4080;
    border-radius: 8px; padding: 3px 10px;
    font-size: 0.68rem; color: #4ECDC4;
    font-family: 'JetBrains Mono', monospace; font-weight: 600;
}
.timer-bar-wrap { height: 3px; background: #1e2840; width: 100%; }
.timer-bar { height: 3px; background: linear-gradient(90deg,#4ECDC4,#45B7D1); width: 100%; }

.results-panel {
    border: 1.5px solid #1e3060; border-radius: 12px;
    background: #0d1421; padding: 1.2rem; min-height: 360px;
    display: flex; flex-direction: column; gap: .8rem;
}

.pred-card {
    background: linear-gradient(135deg, #1a2744, #0d1a33);
    border: 2px solid; border-radius: 14px; padding: 1.2rem;
    text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,.4);
}
.pred-icon { font-size: 3rem; margin-bottom: .3rem; }
.pred-label { font-size: 1.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; }
.pred-conf { font-size: .9rem; color: #8899bb; font-family: 'JetBrains Mono', monospace; }

.bars-title { font-size: .72rem; font-weight: 600; color: #8899bb;
    text-transform: uppercase; letter-spacing: .06em; }
.conf-row {
    display: flex; align-items: center; gap: .6rem;
    padding: 6px 10px; background: #0a0e1a;
    border-radius: 7px; border: 1px solid #1e3060; margin-bottom: .4rem;
}
.conf-lbl { width: 85px; font-size: .75rem; font-weight: 600;
    color: #8899bb; text-transform: uppercase; }
.conf-bg { flex:1; background:#1e2840; border-radius:4px; height:9px; overflow:hidden; }
.conf-fill { height:100%; border-radius:4px; transition:width .5s ease; }
.conf-val { width:48px; text-align:right;
    font-family:'JetBrains Mono',monospace; font-size:.75rem; font-weight:600; }

.tip-box {
    background: #0a1628; border: 1px solid #1e3060; border-left: 4px solid;
    border-radius: 8px; padding: .75rem 1rem; font-size: .8rem; color: #a0b0cc;
}
.tip-box strong { color: #e8edf5; }

.metrics { display: grid; grid-template-columns: repeat(3,1fr); gap: .5rem; }
.metric-pill {
    background: #1a2744; border: 1px solid #2a4080;
    border-radius: 8px; padding: .5rem; text-align: center;
    font-size: .7rem; color: #8899bb;
}
.metric-pill span { display:block; font-size:1rem; font-weight:700; color:#4ECDC4; }

.waiting {
    flex:1; display:flex; flex-direction:column; align-items:center;
    justify-content:center; color:#8899bb; text-align:center; gap:.5rem;
}
.waiting .big { font-size:3rem; }
.waiting .t1 { font-size:.95rem; font-weight:600; color:#a0b0cc; }
.waiting .t2 { font-size:.78rem; }

.spinner {
    width: 32px; height: 32px; border: 3px solid #1e3060;
    border-top-color: #4ECDC4; border-radius: 50%;
    animation: spin .8s linear infinite; margin: 0 auto 6px auto;
}
@keyframes spin { to{ transform:rotate(360deg) } }
</style>
</head>
<body>

<div class="title-bar">
    <div style="font-size:2rem">♻️</div>
    <div>
        <h1>GarbageAI Classifier</h1>
        <p>Real-time waste classification · 5 categories · Vision AI · Smart Dustbin</p>
    </div>
    <div class="status-badge"><div class="pulse"></div> LIVE</div>
</div>

<div class="main-grid">

    <!-- CAMERA -->
    <div>
        <div class="panel-header">
            <span class="dot dot-r"></span>
            <span class="dot dot-y"></span>
            <span class="dot dot-g"></span>
            &nbsp;LIVE CAMERA FEED — AUTO SCANNING
        </div>
        <div class="cam-wrap">
            <video id="video" autoplay playsinline muted></video>
            <canvas id="canvas"></canvas>
            <div class="scan-badge" id="scanBadge">● STARTING</div>
        </div>
        <div class="timer-bar-wrap"><div class="timer-bar" id="timerBar"></div></div>
        <div style="font-size:.7rem;color:#4a5a7a;margin-top:5px;text-align:center;">
            Auto-classifies every 4 seconds · Frames processed via Groq Vision API
        </div>
    </div>

    <!-- RESULTS -->
    <div>
        <div style="font-size:1.1rem;font-weight:700;margin-bottom:.8rem;color:#e8edf5;">
            🔍 Classification Results
        </div>
        <div class="results-panel" id="resultsPanel">
            <div class="waiting" id="waitMsg">
                <div class="big">📷</div>
                <div class="t1">Waiting for camera…</div>
                <div class="t2">Allow camera access — classification starts automatically</div>
            </div>
        </div>
    </div>

</div>

<div style="border-top:1px solid #1e3060;padding-top:.8rem;margin-top:1rem;
     display:flex;justify-content:space-between;font-size:.7rem;color:#4a5a7a;">
    <div>♻️ GarbageAI · Smart Dustbin Waste Classifier</div>
    <div>🪟 Glass &nbsp;|&nbsp; 📄 Paper &nbsp;|&nbsp; ⚙️ Metal &nbsp;|&nbsp; 🌿 Organic &nbsp;|&nbsp; 🧴 Plastic</div>
</div>

<script>
const GROQ_KEY  = \"""" + GROQ_API_KEY + """\";
const GROQ_MODEL= \"""" + GROQ_MODEL + """\";
const INTERVAL  = 4000;

const CLASSES = ["glass","paper","metal","organic","plastic"];
const ICONS   = {glass:"🪟",paper:"📄",metal:"⚙️",organic:"🌿",plastic:"🧴"};
const COLORS  = {glass:"#4ECDC4",paper:"#45B7D1",metal:"#95A5A6",organic:"#2ECC71",plastic:"#E67E22"};
const TIPS    = {
    glass:   "Rinse before recycling. Remove lids. Place in glass recycling bin.",
    paper:   "Keep dry. Flatten boxes. Remove food residue before recycling.",
    metal:   "Rinse cans. Both aluminium & steel are recyclable.",
    organic: "Compost it! Great for garden bins or municipal composting.",
    plastic: "Check the resin code (1-7). Rinse containers before recycling."
};

const video    = document.getElementById('video');
const canvas   = document.getElementById('canvas');
const badge    = document.getElementById('scanBadge');
const timerBar = document.getElementById('timerBar');
const panel    = document.getElementById('resultsPanel');

let busy = false;
let lastHTML = '';

const SYSTEM = `You are a precise waste material classifier.
Look at the image and classify the primary object into ONE of: glass, paper, metal, organic, plastic.
Assign integer confidence scores (0-100) to each — they MUST sum to exactly 100.
Consider: transparency=glass, fibrous/foldable=paper, shiny/rigid=metal, food/plant=organic, flexible/coloured=plastic.
Reply ONLY in this exact format, no other text:
TOP: <category>
glass: <score>
paper: <score>
metal: <score>
organic: <score>
plastic: <score>`;

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: {ideal:'environment'}, width:{ideal:640}, height:{ideal:480} },
            audio: false
        });
        video.srcObject = stream;
        await video.play();
        badge.textContent = '● LIVE';
        badge.style.color = '#4dbb4d';
        panel.innerHTML = `<div class="waiting"><div class="big">✅</div>
            <div class="t1">Camera active!</div>
            <div class="t2">First classification in 1 second…</div></div>`;
        setTimeout(runCycle, 1000);
    } catch(e) {
        badge.textContent = '● NO CAM';
        badge.style.color = '#ff5f57';
        panel.innerHTML = `<div class="waiting">
            <div class="big">⚠️</div>
            <div class="t1" style="color:#ff7070">Camera Error</div>
            <div class="t2">${e.message}<br>Check browser permissions and reload.</div>
        </div>`;
    }
}

function captureB64() {
    const W = video.videoWidth || 640;
    const H = video.videoHeight || 480;
    canvas.width = W; canvas.height = H;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.82).split(',')[1];
}

function animateBar() {
    timerBar.style.transition = 'none';
    timerBar.style.width = '100%';
    requestAnimationFrame(() => requestAnimationFrame(() => {
        timerBar.style.transition = `width ${INTERVAL}ms linear`;
        timerBar.style.width = '0%';
    }));
}

function parseResult(raw, elapsed) {
    const scores = {};
    let topClass = null;
    for (const line of raw.split('\\n')) {
        const l = line.trim().toLowerCase();
        if (l.startsWith('top:')) topClass = l.split(':')[1].trim();
        for (const cls of CLASSES) {
            if (l.startsWith(cls + ':')) scores[cls] = parseFloat(l.split(':')[1]) || 0;
        }
    }
    for (const cls of CLASSES) if (!(cls in scores)) scores[cls] = 0;
    const total = Object.values(scores).reduce((a,b)=>a+b,0) || 1;
    for (const cls of CLASSES) scores[cls] = Math.round(scores[cls]/total*1000)/10;
    if (!CLASSES.includes(topClass)) topClass = CLASSES.reduce((a,b)=>scores[a]>scores[b]?a:b);
    return { topClass, scores, elapsed };
}

function buildResultHTML(r) {
    const color = COLORS[r.topClass];
    const conf  = r.scores[r.topClass];
    const bars  = CLASSES.map(cls => {
        const c = COLORS[cls]; const s = r.scores[cls];
        const top = cls === r.topClass;
        const bold = top ? 'font-weight:700;color:#e8edf5;' : '';
        return `<div class="conf-row">
            <div class="conf-lbl" style="${bold}">${ICONS[cls]} ${cls}</div>
            <div class="conf-bg"><div class="conf-fill" style="width:${s}%;background:${c};"></div></div>
            <div class="conf-val" style="color:${c};${bold}">${s.toFixed(1)}%</div>
        </div>`;
    }).join('');
    const probs = CLASSES.map(c=>r.scores[c]/100);
    const ent   = -probs.reduce((a,p)=>a+(p>0?p*Math.log(p+1e-9):0),0);
    const cert  = Math.max(0,Math.round(100-ent*40));
    return `
    <div class="pred-card" style="border-color:${color};box-shadow:0 0 30px ${color}22;">
        <div class="pred-icon">${ICONS[r.topClass]}</div>
        <div class="pred-label" style="color:${color}">${r.topClass.toUpperCase()}</div>
        <div class="pred-conf">Confidence: ${conf.toFixed(1)}%</div>
    </div>
    <div class="bars-title">All Category Scores</div>
    ${bars}
    <div class="tip-box" style="border-left-color:${color};">
        <strong>♻️ Disposal Tip:</strong> ${TIPS[r.topClass]}
    </div>
    <div class="metrics">
        <div class="metric-pill"><span>${r.elapsed} ms</span>Response Time</div>
        <div class="metric-pill"><span>${conf.toFixed(1)}%</span>Top Confidence</div>
        <div class="metric-pill"><span>${cert}%</span>Certainty Index</div>
    </div>`;
}

async function runCycle() {
    if (busy || video.readyState < 2) { setTimeout(runCycle, 500); return; }
    busy = true;
    badge.textContent = '⟳ SCANNING';
    badge.style.color = '#febc2e';
    animateBar();

    // Show spinner but keep last result visible
    panel.innerHTML = lastHTML + `<div style="padding:.6rem 0;text-align:center;">
        <div class="spinner"></div>
        <div style="font-size:.72rem;color:#8899bb;margin-top:4px;">Classifying…</div>
    </div>`;

    const t0  = performance.now();
    const b64 = captureB64();

    try {
        const resp = await fetch("https://api.groq.com/openai/v1/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type":  "application/json",
                "Authorization": "Bearer " + GROQ_KEY
            },
            body: JSON.stringify({
                model: GROQ_MODEL,
                max_tokens: 120,
                temperature: 0.05,
                messages: [{
                    role: "user",
                    content: [
                        { type: "text", text: SYSTEM + "\\n\\nClassify this waste item now." },
                        { type: "image_url", image_url: { url: "data:image/jpeg;base64," + b64 } }
                    ]
                }]
            })
        });

        const elapsed = Math.round(performance.now() - t0);
        const json    = await resp.json();

        if (!resp.ok) throw new Error(json.error?.message || resp.statusText);

        const result = parseResult(json.choices[0].message.content.trim(), elapsed);
        lastHTML = buildResultHTML(result);
        panel.innerHTML = lastHTML;
        badge.textContent = '✓ ' + result.topClass.toUpperCase();
        badge.style.color = COLORS[result.topClass];

    } catch(err) {
        panel.innerHTML = `<div style="background:#1a0a0a;border:1px solid #4a1a1a;
            border-radius:10px;padding:1rem;color:#ff7070;font-size:.8rem;word-break:break-all;">
            ❌ <strong>API Error:</strong> ${err.message}
        </div>` + lastHTML;
        badge.textContent = '● ERROR';
        badge.style.color = '#ff5f57';
    }

    busy = false;
    setTimeout(() => {
        badge.textContent = '● LIVE';
        badge.style.color = '#4dbb4d';
        setTimeout(runCycle, INTERVAL);
    }, 500);
}

startCamera();
</script>
</body>
</html>
"""

components.html(HTML, height=800, scrolling=False)