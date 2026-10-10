import html
import io
import json
import re
import time
from collections import Counter

import requests
import streamlit as st
import streamlit.components.v1 as components

# =============================================================
# Configuration
# =============================================================
st.set_page_config(
    page_title="Pegasus Summarizer",
    page_icon="📝",
    layout="wide",
)

API_BASE = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_BASE}/predict"
HEALTH_URL = f"{API_BASE}/health"

MAX_WORDS = 5000      # max words accepted in the UI
CHUNK_WORDS = 350     # Pegasus handles ~512 tokens, so long texts are split
READING_WPM = 238     # average adult reading speed
HISTORY_LIMIT = 15

SAMPLE_TEXT = (
    "The James Webb Space Telescope, launched in December 2021, is the largest "
    "and most powerful space telescope ever built. It observes the universe in "
    "infrared light, which allows it to see through clouds of dust and gas and "
    "to detect the faint glow of the earliest galaxies. Since beginning science "
    "operations in 2022, it has captured detailed images of distant galaxies, "
    "analysed the atmospheres of planets orbiting other stars, and revealed new "
    "details about how stars and planetary systems form. Scientists say the "
    "observatory will keep reshaping our understanding of cosmic history for "
    "at least the next two decades."
)

STOPWORDS = set(
    """
    about above after again against all also and any are because been before
    being below between both but can could did does doing down during each few
    for from further had has have having her here hers him his how into its
    itself just more most myself not now off once only other our ours out over
    own same she should some such than that the their theirs them then there
    these they this those through too under until very was were what when
    where which while who whom why will with would you your yours said says
    """.split()
)

# =============================================================
# Themes (CSS variables)
# =============================================================
THEMES = {
    "light": {
        "bg": "#faf8ff",
        "glow1": "#fde7f5",
        "glow2": "#e4dcff",
        "surface": "#ffffff",
        "surface2": "#f4f1ff",
        "border": "#ddd6f8",
        "text": "#1d1a36",
        "heading": "#14112b",
        "muted": "#5b5775",
        "muted2": "#7d7897",
        "placeholder": "#9a96b4",
        "accent": "#6c4df6",
        "accent-text": "#3b2fb8",
        "chip": "#ece7ff",
        "shadow": "rgba(84, 63, 200, 0.10)",
        "good": "#0f9d58",
        "bad": "#d93025",
        "grad1": "#7c5cff",
        "grad2": "#4a35d6",
        "mark": "#fff0a8",
        "mark-text": "#4a3b00",
    },
    "dark": {
        "bg": "#0d0a1f",
        "glow1": "rgba(236, 72, 153, 0.20)",
        "glow2": "rgba(124, 92, 255, 0.26)",
        "surface": "#171331",
        "surface2": "#1f1a40",
        "border": "#352e68",
        "text": "#ece9ff",
        "heading": "#ffffff",
        "muted": "#a8a2d0",
        "muted2": "#8c85b8",
        "placeholder": "#6f6a9a",
        "accent": "#8b6cff",
        "accent-text": "#c9bcff",
        "chip": "#2a2358",
        "shadow": "rgba(0, 0, 0, 0.45)",
        "good": "#3ddc97",
        "bad": "#ff7b7b",
        "grad1": "#8b6cff",
        "grad2": "#5a45e6",
        "mark": "#5b4a00",
        "mark-text": "#ffe58a",
    },
}


def theme_vars(name: str) -> str:
    values = ";".join(f"--{k}:{v}" for k, v in THEMES[name].items())
    return ":root{" + values + ";color-scheme:" + name + "}"


STATIC_CSS = """
#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background:
        radial-gradient(900px 500px at 8% 0%, var(--glow1) 0%, transparent 60%),
        radial-gradient(900px 500px at 95% 10%, var(--glow2) 0%, transparent 60%),
        var(--bg);
}
.block-container {max-width: 1150px; padding-top: 1.2rem;}

/* Hidden helper iframes (scripts) must not take space */
.st-key-fx_host {position: absolute; width: 1px; height: 0; margin: 0; overflow: hidden;}

/* Base colors */
.stApp, .stApp p, .stApp label, .stApp span, .stApp li {color: var(--text);}
.stApp h1, .stApp h2, .stApp h3 {color: var(--heading);}

/* Theme switch: circular reveal (View Transitions API) */
::view-transition-old(root), ::view-transition-new(root) {
    animation: none; mix-blend-mode: normal;
}
::view-transition-old(root) {z-index: 1;}
::view-transition-new(root) {z-index: 2;}

/* Fallback fade for browsers without View Transitions */
@supports not (view-transition-name: none) {
    .result-card, .stat, .feature, .status, .chip, textarea, button,
    div[data-testid="stExpander"] {
        transition: background-color .4s ease, color .4s ease, border-color .4s ease;
    }
}

/* Top bar */
.brand {display: flex; align-items: center; gap: 10px; font-weight: 800;
        font-size: 1.15rem; color: var(--heading);}
.logo {
    width: 34px; height: 34px; border-radius: 10px; color: #fff !important;
    background: linear-gradient(135deg, var(--grad1), var(--grad2));
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 800; box-shadow: 0 6px 16px var(--shadow);
}
.status {
    font-size: 0.85rem; font-weight: 600; padding: 5px 12px; float: right;
    border-radius: 999px; background: var(--surface); border: 1px solid var(--border);
}
.status.on {color: var(--good) !important;}
.status.off {color: var(--bad) !important;}

/* Hero (plays once on load) */
.hero {text-align: center; margin: 1.2rem 0 1.2rem 0; animation: rise .7s ease both;}
@keyframes rise {from {opacity: 0; transform: translateY(14px);} to {opacity: 1; transform: none;}}
.hero h1 {
    font-size: 2.8rem; font-weight: 800; letter-spacing: -0.02em;
    color: var(--heading); margin-bottom: 0.4rem;
}
.hero p {color: var(--muted); font-size: 1.1rem; margin: 0;}
.badge {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    background: var(--chip); color: var(--accent-text); font-size: 0.8rem;
    font-weight: 600; margin-bottom: 0.9rem;
}
.features {
    display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;
    margin: 1.1rem 0 1.4rem 0;
}
.feature {
    background: var(--surface); border: 1px solid var(--border); border-radius: 999px;
    padding: 6px 14px; font-size: 0.85rem; color: var(--accent-text); font-weight: 600;
}

/* Tabs */
button[data-baseweb="tab"] p {color: var(--muted) !important; font-weight: 600;}
button[data-baseweb="tab"][aria-selected="true"] p {color: var(--accent-text) !important;}
div[data-baseweb="tab-highlight"] {background-color: var(--accent) !important;}
div[data-baseweb="tab-border"] {background-color: var(--border) !important;}

/* Text area as an editor card */
div[data-testid="stTextArea"] > div,
div[data-testid="stTextArea"] div[data-baseweb="textarea"],
div[data-testid="stTextArea"] div[data-baseweb="base-input"] {
    background: var(--surface) !important; border-radius: 16px !important;
    border: none !important;
}
div[data-testid="stTextArea"] textarea {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    caret-color: var(--accent);
    background: var(--surface) !important;
    border-radius: 16px; border: 1px solid var(--border) !important;
    font-size: 1.05rem; line-height: 1.7; padding: 1.1rem 1.2rem;
    box-shadow: 0 10px 34px var(--shadow);
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: var(--placeholder) !important; -webkit-text-fill-color: var(--placeholder) !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 22%, transparent);
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; font-weight: 600; padding: 0.65rem 1rem;
    background: var(--surface); color: var(--accent-text) !important;
    border: 1px solid var(--border);
}
.stButton > button p, .stDownloadButton > button p {color: var(--accent-text) !important;}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent); background: var(--surface2); color: var(--accent-text) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--grad1), var(--grad2), var(--grad1));
    background-size: 180% 180%; background-position: 0% 50%;
    border: none; color: #ffffff !important; font-size: 1.05rem;
    padding: 0.8rem 1rem; box-shadow: 0 10px 24px var(--shadow);
    transition: background-position .6s ease, transform .15s ease;
}
.stButton > button[kind="primary"]:hover {
    background-position: 100% 50%; transform: translateY(-1px);
    color: #ffffff !important;
}
.stButton > button[kind="primary"] p {color: #ffffff !important;}

/* Expanders and file uploader */
div[data-testid="stExpander"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
}
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] details > summary {
    background: var(--surface) !important; border-radius: 12px;
}
div[data-testid="stExpander"] details > summary:hover {background: var(--surface2) !important;}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p {color: var(--accent-text) !important; font-weight: 600;}
div[data-testid="stExpander"] svg {color: var(--accent-text) !important; fill: var(--accent-text) !important;}
section[data-testid="stFileUploaderDropzone"] {
    background: var(--surface2); border: 1.5px dashed var(--accent); border-radius: 12px;
}
section[data-testid="stFileUploaderDropzone"] * {color: var(--accent-text) !important;}
section[data-testid="stFileUploaderDropzone"] button {
    background: var(--surface); border: 1px solid var(--border);
}

/* Alerts */
div[data-testid="stAlert"] {
    background: var(--surface2) !important; border: 1px solid var(--border);
    border-radius: 12px;
}
div[data-testid="stAlert"] * {color: var(--text) !important;}

/* Pill switches (st.radio) */
div[data-testid="stRadio"] > div {gap: 0.4rem;}
div[data-testid="stRadio"] label {
    background: var(--surface); border: 1px solid var(--border); border-radius: 999px;
    padding: 4px 14px; cursor: pointer;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: var(--chip); border-color: var(--accent);
}
div[data-testid="stRadio"] label p {font-size: 0.9rem; font-weight: 600; color: var(--accent-text) !important;}
div[data-testid="stRadio"] label > div:first-child {display: none;}

/* Panels and titles */
.panel-title {font-size: 1.05rem; font-weight: 700; color: var(--heading); margin: 0 0 0.5rem 2px;}
.mini-label {font-size: 0.82rem; color: var(--muted2) !important; margin: 0.7rem 0 0.3rem 2px;}

/* Result card */
.result-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 1.2rem 1.3rem; height: 320px; overflow-y: auto;
    box-sizing: border-box; line-height: 1.8; font-size: 1.02rem;
    color: var(--text); box-shadow: 0 8px 30px var(--shadow);
}
.result-card ul {margin: 0; padding-left: 1.2rem;}
.result-card li {margin-bottom: 0.45rem;}
.placeholder {color: var(--placeholder) !important;}

/* Typewriter-style word reveal + key-term highlight */
.w.fx {opacity: 0; display: inline-block; animation: wordin .35s ease forwards;}
@keyframes wordin {from {opacity: 0; transform: translateY(5px);} to {opacity: 1; transform: none;}}
.w.hl {
    background: var(--mark); color: var(--mark-text) !important;
    border-radius: 5px; padding: 0 4px;
}

/* Stats, meter, chips */
.stats {display: flex; gap: 10px; margin-top: 0.8rem; flex-wrap: wrap;}
.stat {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 8px 14px; font-size: 0.85rem; color: var(--muted);
    box-shadow: 0 4px 14px var(--shadow);
}
.stat b {color: var(--heading); font-size: 1rem; margin-right: 4px;}
.stat.good b {color: var(--good);}

.meter {
    margin-top: 0.8rem; padding: 12px 14px; border-radius: 12px;
    background: var(--surface); border: 1px solid var(--border);
}
.meter-row {display: flex; align-items: center; gap: 10px; font-size: 0.82rem; color: var(--muted);}
.meter-row + .meter-row {margin-top: 8px;}
.meter-row span {width: 62px; flex: none;}
.meter-row b {width: 44px; text-align: right; color: var(--heading); flex: none;}
.bar {flex: 1; height: 10px; border-radius: 999px; background: var(--surface2); overflow: hidden;}
.fill {height: 100%; border-radius: 999px; animation: grow 1.1s cubic-bezier(.2,.8,.2,1) both;}
.fill.orig {background: var(--muted2);}
.fill.sum {background: linear-gradient(90deg, var(--grad1), var(--grad2));}
@keyframes grow {from {width: 0;}}

.chips {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 0.4rem;}
.chip {
    background: var(--chip); color: var(--accent-text) !important; border-radius: 999px;
    padding: 4px 12px; font-size: 0.85rem; font-weight: 600;
}

.counter-row {display: flex; justify-content: space-between; margin-top: -0.3rem;
              font-size: 0.85rem;}
.counter {color: var(--muted2) !important;}
.counter.over {color: var(--bad) !important; font-weight: 600;}
.hint {color: var(--placeholder) !important;}

@media (prefers-reduced-motion: reduce) {
    .hero, .w.fx, .fill {animation: none !important; opacity: 1;}
}
"""

# JavaScript injected once into the PARENT page (runs in the page, not in an iframe)
# 1) circular-reveal when the theme button is clicked   2) Ctrl+Enter to summarize
FX_JS = r"""
(function () {
  if (window.__pegasusFx) return;
  window.__pegasusFx = true;

  function bg() {
    return getComputedStyle(document.documentElement).getPropertyValue('--bg').trim();
  }

  document.addEventListener('click', function (e) {
    var btn = e.target.closest && e.target.closest('[class*="st-key-theme_toggle"]');
    if (!btn) return;
    var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!document.startViewTransition || reduce) return;

    var old = bg();
    var x = e.clientX, y = e.clientY;
    var radius = Math.hypot(
      Math.max(x, window.innerWidth - x),
      Math.max(y, window.innerHeight - y)
    );

    var transition = document.startViewTransition(function () {
      return new Promise(function (resolve) {
        var started = Date.now();
        (function wait() {
          if (bg() !== old || Date.now() - started > 3000) {
            setTimeout(resolve, 80);
          } else {
            setTimeout(wait, 30);
          }
        })();
      });
    });

    transition.ready.then(function () {
      document.documentElement.animate(
        {clipPath: ['circle(0px at ' + x + 'px ' + y + 'px)',
                    'circle(' + radius + 'px at ' + x + 'px ' + y + 'px)']},
        {duration: 750, easing: 'cubic-bezier(.4,0,.2,1)',
         pseudoElement: '::view-transition-new(root)'}
      );
    }).catch(function () {});
  }, true);

  document.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' &&
        e.target && e.target.tagName === 'TEXTAREA') {
      setTimeout(function () {
        var buttons = document.querySelectorAll('button');
        for (var i = 0; i < buttons.length; i++) {
          if (buttons[i].innerText.trim() === 'Summarize') { buttons[i].click(); break; }
        }
      }, 400);
    }
  }, true);
})();
"""

ACTION_BAR_HTML = """
<style>
  body {margin: 0; font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;}
  .row {display: flex; gap: 8px;}
  button {
    flex: 1; font: 600 14px system-ui, sans-serif; padding: 10px 12px; cursor: pointer;
    border-radius: 12px; border: 1px solid __BORDER__; background: __SURFACE__;
    color: __ACCENT__; transition: background .15s, border-color .15s, transform .1s;
  }
  button:hover {background: __SURFACE2__; border-color: __ACCENT_BORDER__;}
  button:active {transform: scale(.97);}
</style>
<div class="row">
  <button id="listen">🔊 Listen</button>
  <button id="stop">⏹ Stop</button>
  <button id="copy">📋 Copy</button>
</div>
<script>
  var text = __TEXT__;
  var listen = document.getElementById('listen');
  var copy = document.getElementById('copy');
  listen.onclick = function () {
    try {
      window.speechSynthesis.cancel();
      var u = new SpeechSynthesisUtterance(text.replace(/\\n/g, '. '));
      u.lang = 'en-US';
      window.speechSynthesis.speak(u);
    } catch (e) {}
  };
  document.getElementById('stop').onclick = function () {
    try { window.speechSynthesis.cancel(); } catch (e) {}
  };
  copy.onclick = function () {
    var done = function () {
      copy.textContent = '✅ Copied';
      setTimeout(function () { copy.textContent = '📋 Copy'; }, 1500);
    };
    var fallback = function () {
      var t = document.createElement('textarea');
      t.value = text; document.body.appendChild(t); t.select();
      try { document.execCommand('copy'); done(); } catch (e) {}
      document.body.removeChild(t);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, fallback);
    } else { fallback(); }
  };
</script>
"""

# =============================================================
# Session state
# =============================================================
if "theme" not in st.session_state:
    requested = st.query_params.get("theme", "dark")
    st.session_state.theme = requested if requested in THEMES else "dark"

defaults = {
    "text_input": "",
    "summary": "",
    "last_input": "",
    "elapsed": 0.0,
    "parts": 1,
    "history": [],
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


# =============================================================
# Helpers
# =============================================================
def word_count(value: str) -> int:
    return len(value.split())


@st.cache_data(ttl=10, show_spinner=False)
def api_online() -> bool:
    try:
        return requests.get(HEALTH_URL, timeout=2).ok
    except requests.exceptions.RequestException:
        return False


def mostly_non_english(text: str) -> bool:
    """True when most letters are not Latin (e.g. Arabic): Pegasus is English-only."""
    letters = re.findall(r"[^\W\d_]", text)
    if len(letters) < 20:
        return False
    latin = sum(1 for ch in letters if ch.isascii())
    return latin / len(letters) < 0.6


def key_terms(text: str, n: int = 8) -> list:
    words = re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text.lower())
    counts = Counter(w for w in words if w not in STOPWORDS)
    return [w for w, _ in counts.most_common(n)]


def count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(1, count)


def readability(text: str):
    """Flesch Reading Ease (0-100, higher = easier). Returns None for empty text."""
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return None
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    syllables = sum(count_syllables(w) for w in words)
    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
    return max(0, min(100, round(score)))


def readability_label(score) -> str:
    if score is None:
        return "-"
    if score >= 70:
        return "easy"
    if score >= 50:
        return "standard"
    if score >= 30:
        return "difficult"
    return "very difficult"


def format_summary(summary: str, mode: str) -> str:
    """Return the summary as a paragraph or as one bullet per sentence."""
    if mode == "Bullet points":
        sentences = [
            x.strip() for x in re.split(r"(?<=[.!?])\s+", summary) if x.strip()
        ]
        return "\n".join(sentences)
    return summary


def words_html(text: str, terms: set, highlight: bool, animate: bool, counter: list) -> str:
    """Wrap every word in a span so we can highlight key terms and animate the reveal."""
    spans = []
    for token in text.split():
        core = re.sub(r"[^\w'-]", "", token).lower()
        cls = "w"
        style = ""
        if highlight and core in terms:
            cls += " hl"
        if animate:
            cls += " fx"
            style = f' style="animation-delay:{min(counter[0] * 0.04, 2.2):.2f}s"'
            counter[0] += 1
        spans.append(f'<span class="{cls}"{style}>{html.escape(token)}</span>')
    return " ".join(spans)


def split_chunks(text: str, size: int = CHUNK_WORDS) -> list:
    """Split long text into chunks of at most `size` words on sentence borders."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    chunks, current, count = [], [], 0
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        while len(words) > size:  # a single huge "sentence"
            if current:
                chunks.append(" ".join(current))
                current, count = [], 0
            chunks.append(" ".join(words[:size]))
            words = words[size:]
        if count + len(words) > size and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(" ".join(words))
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def read_uploaded(uploaded) -> str:
    """Extract text from .txt / .md / .pdf / .docx uploads."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()

    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF support needs pypdf: pip install pypdf")
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if name.endswith(".docx"):
        try:
            from docx import Document
        except ImportError:
            raise ValueError("DOCX support needs python-docx: pip install python-docx")
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError("Unsupported file type.")


def call_api(text: str) -> str:
    """Send one chunk to the FastAPI backend and return its summary."""
    try:
        response = requests.post(PREDICT_URL, params={"text": text}, timeout=180)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach the API. Start the FastAPI backend (python app.py) and try again."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("The request timed out. Try again with a shorter text.")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Request failed: {exc}")

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API error ({response.status_code}): {detail}")

    summary = response.json().get("summary", "").replace("<n>", " ").strip()
    if not summary:
        raise RuntimeError("The API returned an empty summary.")
    return summary


def condense(text: str) -> str:
    """Keep summarizing until the text fits in one chunk, then summarize once more."""
    for _ in range(4):  # safety limit
        if word_count(text) <= CHUNK_WORDS:
            break
        text = " ".join(call_api(chunk) for chunk in split_chunks(text))
    return call_api(text)


def run_summary(text: str, concise: bool, on_progress):
    """Summarize text of any length. Returns (summary, number_of_parts, seconds)."""
    started = time.time()
    chunks = [text] if word_count(text) <= CHUNK_WORDS else split_chunks(text)

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        on_progress(i / (len(chunks) + 1), f"Summarizing part {i} of {len(chunks)}...")
        parts.append(call_api(chunk))

    if len(chunks) > 1 and concise:
        on_progress(0.95, "Merging into one short summary...")
        summary = condense(" ".join(parts))
    else:
        summary = " ".join(parts)

    on_progress(1.0, "Done")
    return summary, len(chunks), time.time() - started


def add_history(source: str, summary: str, parts: int):
    entry = {
        "time": time.strftime("%H:%M"),
        "source": source,
        "summary": summary,
        "parts": parts,
    }
    st.session_state.history.insert(0, entry)
    del st.session_state.history[HISTORY_LIMIT:]


def html_frame(doc: str, height: int):
    """Render an HTML snippet in an iframe (st.iframe on new Streamlit, else components.html)."""
    if hasattr(st, "iframe"):
        st.iframe(doc, height=max(1, height))  # st.iframe rejects height=0
    else:
        components.html(doc, height=height)


def action_bar(text: str):
    """Listen / Stop / Copy buttons (browser text-to-speech + clipboard)."""
    t = THEMES[st.session_state.theme]
    payload = json.dumps(text).replace("</", "<\\/")
    doc = (
        ACTION_BAR_HTML.replace("__TEXT__", payload)
        .replace("__BORDER__", t["border"])
        .replace("__SURFACE2__", t["surface2"])
        .replace("__SURFACE__", t["surface"])
        .replace("__ACCENT_BORDER__", t["accent"])
        .replace("__ACCENT__", t["accent-text"])
    )
    html_frame(doc, 48)


# ----- callbacks -----
def toggle_theme():
    new = "light" if st.session_state.theme == "dark" else "dark"
    st.session_state.theme = new
    st.query_params["theme"] = new


def load_sample():
    st.session_state.text_input = SAMPLE_TEXT


def clear_all():
    st.session_state.text_input = ""
    st.session_state.summary = ""
    st.session_state.last_input = ""
    st.session_state.elapsed = 0.0
    st.session_state.parts = 1


def load_file():
    uploaded = st.session_state.get("uploaded_file")
    st.session_state.upload_error = ""
    if uploaded is None:
        return
    try:
        text = read_uploaded(uploaded).strip()
        if not text:
            st.session_state.upload_error = "No readable text found in this file."
        else:
            st.session_state.text_input = text
    except Exception as exc:  # noqa: BLE001
        st.session_state.upload_error = str(exc)


def restore(index: int):
    entry = st.session_state.history[index]
    st.session_state.summary = entry["summary"]
    st.session_state.last_input = entry["source"]
    st.session_state.text_input = entry["source"]
    st.session_state.parts = entry["parts"]
    st.session_state.elapsed = 0.0


# =============================================================
# Theme + scripts
# =============================================================
st.markdown(
    "<style>" + theme_vars(st.session_state.theme) + STATIC_CSS + "</style>",
    unsafe_allow_html=True,
)

# Inject the helper script into the main page once (circular reveal + Ctrl+Enter)
with st.container(key="fx_host"):
    html_frame(
        "<script>(function(){try{var d=window.parent.document;"
        "if(d.getElementById('pegasus-fx'))return;"
        "var s=d.createElement('script');s.id='pegasus-fx';"
        "s.textContent=" + json.dumps(FX_JS) + ";d.head.appendChild(s);}catch(e){}})();</script>",
        0,
    )

# =============================================================
# Top bar + hero
# =============================================================
online = api_online()
status_html = (
    '<span class="status on">● API online</span>'
    if online
    else '<span class="status off">● API offline</span>'
)

nav_left, nav_status, nav_theme = st.columns([5, 2, 2], vertical_alignment="center")
nav_left.markdown(
    '<div class="brand"><span class="logo">S</span>Pegasus Summarizer</div>',
    unsafe_allow_html=True,
)
nav_status.markdown(status_html, unsafe_allow_html=True)
nav_theme.button(
    "☀️ Light mode" if st.session_state.theme == "dark" else "🌙 Dark mode",
    key="theme_toggle",
    on_click=toggle_theme,
    use_container_width=True,
)

st.markdown(
    """
    <div class="hero">
        <span class="badge">Powered by Pegasus</span>
        <h1>Summarize Any Text in Seconds</h1>
        <p>Paste an article, report or essay and get a short summary
        that keeps the main ideas.</p>
    </div>
    <div class="features">
        <span class="feature">Handles long texts</span>
        <span class="feature">Paragraph or bullet points</span>
        <span class="feature">TXT, PDF and DOCX upload</span>
        <span class="feature">Listen to your summary</span>
        <span class="feature">Dark and light mode</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_summarize, tab_history = st.tabs(["Summarize", "History"])

# =============================================================
# Tab 1: Summarize
# =============================================================
with tab_summarize:
    left, right = st.columns(2, gap="large")

    # ---------- Left: input ----------
    with left:
        st.markdown('<div class="panel-title">Your text</div>', unsafe_allow_html=True)
        st.text_area(
            "Your text",
            key="text_input",
            height=320,
            placeholder="Start typing or paste your text here...",
            label_visibility="collapsed",
        )

        words = word_count(st.session_state.text_input)
        over = words > MAX_WORDS
        st.markdown(
            f'<div class="counter-row"><span class="counter {"over" if over else ""}">'
            f'{words:,} / {MAX_WORDS:,} words</span>'
            f'<span class="hint">Ctrl + Enter to summarize</span></div>',
            unsafe_allow_html=True,
        )

        if words and mostly_non_english(st.session_state.text_input):
            st.warning(
                "This text doesn't look English. The Pegasus model only "
                "summarizes English text well."
            )

        b1, b2 = st.columns(2)
        b1.button("Try a sample", on_click=load_sample, use_container_width=True)
        b2.button("Clear", on_click=clear_all, use_container_width=True)

        with st.expander("Upload a file (TXT, MD, PDF, DOCX)"):
            st.file_uploader(
                "Upload a file",
                type=["txt", "md", "pdf", "docx"],
                key="uploaded_file",
                on_change=load_file,
                label_visibility="collapsed",
            )
            if st.session_state.get("upload_error"):
                st.error(st.session_state.upload_error)

        st.markdown('<div class="mini-label">Summary detail</div>', unsafe_allow_html=True)
        detail = st.radio(
            "Summary detail",
            ["Standard", "Concise"],
            horizontal=True,
            key="detail_mode",
            label_visibility="collapsed",
            help=(
                "Standard keeps one summary per part of a long text. "
                "Concise merges them into one short summary."
            ),
        )

        summarize_clicked = st.button(
            "Summarize", type="primary", use_container_width=True
        )

    # ---------- Right: output ----------
    with right:
        st.markdown('<div class="panel-title">Summary</div>', unsafe_allow_html=True)

        if summarize_clicked:
            current = st.session_state.text_input.strip()
            if not current:
                st.warning("Add some text first, then press Summarize.")
            elif over:
                st.warning(f"Text is over {MAX_WORDS:,} words. Shorten it and try again.")
            else:
                progress = st.progress(0.0, text="Starting...")
                try:
                    summary, parts, elapsed = run_summary(
                        current,
                        concise=(detail == "Concise"),
                        on_progress=lambda f, msg: progress.progress(f, text=msg),
                    )
                    st.session_state.summary = summary
                    st.session_state.last_input = current
                    st.session_state.elapsed = elapsed
                    st.session_state.parts = parts
                    st.session_state.fresh = True  # play the reveal animation once
                    add_history(current, summary, parts)
                except RuntimeError as exc:
                    st.error(str(exc))
                finally:
                    progress.empty()

        if st.session_state.summary:
            fmt_col, hl_col = st.columns([3, 2], vertical_alignment="center")
            output_format = fmt_col.radio(
                "Format",
                ["Paragraph", "Bullet points"],
                horizontal=True,
                key="output_format",
                label_visibility="collapsed",
            )
            highlight = hl_col.toggle("Highlight key terms", value=True, key="highlight")

            display_text = format_summary(st.session_state.summary, output_format)
            terms = set(key_terms(st.session_state.last_input))
            animate = st.session_state.pop("fresh", False)
            counter = [0]

            if output_format == "Bullet points":
                items = "".join(
                    f"<li>{words_html(line, terms, highlight, animate, counter)}</li>"
                    for line in display_text.splitlines()
                )
                card_html = f"<ul>{items}</ul>"
            else:
                card_html = words_html(display_text, terms, highlight, animate, counter)

            st.markdown(
                f'<div class="result-card">{card_html}</div>',
                unsafe_allow_html=True,
            )

            original_words = word_count(st.session_state.last_input)
            summary_words = word_count(st.session_state.summary)
            reduction = (
                max(0, round((1 - summary_words / original_words) * 100))
                if original_words
                else 0
            )
            saved_minutes = (original_words - summary_words) / READING_WPM
            saved_label = "<1 min" if saved_minutes < 1 else f"{round(saved_minutes)} min"

            read_before = readability(st.session_state.last_input)
            read_after = readability(st.session_state.summary)
            read_label = (
                f"{read_before} → {read_after}"
                if read_before is not None and read_after is not None
                else "-"
            )
            summary_pct = (
                min(100, max(2, round(summary_words / original_words * 100)))
                if original_words
                else 100
            )

            st.markdown(
                f"""
                <div class="stats">
                    <div class="stat good"><b>{reduction}%</b>shorter</div>
                    <div class="stat good"><b>{saved_label}</b>reading time saved</div>
                    <div class="stat"><b>{read_label}</b>readability ({readability_label(read_after)})</div>
                    <div class="stat"><b>{st.session_state.parts}</b>part(s)</div>
                    <div class="stat"><b>{st.session_state.elapsed:.1f}s</b>time</div>
                </div>
                <div class="meter">
                    <div class="meter-row"><span>Original</span>
                        <div class="bar"><div class="fill orig" style="width:100%"></div></div>
                        <b>{original_words:,}</b></div>
                    <div class="meter-row"><span>Summary</span>
                        <div class="bar"><div class="fill sum" style="width:{summary_pct}%"></div></div>
                        <b>{summary_words:,}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            top_terms = key_terms(st.session_state.last_input)
            if top_terms:
                chips = "".join(f'<span class="chip">{html.escape(t)}</span>' for t in top_terms)
                st.markdown('<div class="mini-label">Key terms</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

            st.write("")
            action_bar(display_text)

            md_text = (
                "# Summary\n\n"
                + (
                    "\n".join(f"- {line}" for line in display_text.splitlines())
                    if output_format == "Bullet points"
                    else display_text
                )
                + "\n"
            )
            d1, d2 = st.columns(2)
            d1.download_button(
                "Download .txt",
                data=display_text,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
            d2.download_button(
                "Download .md",
                data=md_text,
                file_name="summary.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div class="result-card placeholder">'
                "Your summary will appear here.</div>",
                unsafe_allow_html=True,
            )

# =============================================================
# Tab 2: History
# =============================================================
with tab_history:
    if not st.session_state.history:
        st.info("Your recent summaries will show up here during this session.")
    else:
        for i, entry in enumerate(st.session_state.history):
            preview = entry["source"][:70].replace("\n", " ")
            with st.expander(f'{entry["time"]}  ·  {preview}...'):
                st.write(entry["summary"])
                st.button(
                    "Open in editor",
                    key=f"restore_{i}",
                    on_click=restore,
                    args=(i,),
                )