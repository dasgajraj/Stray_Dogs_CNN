import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
import os
import time
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CanineCare AI", page_icon="🐕", layout="wide")

# --- WORLD-CLASS CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #080c10;
        color: #e2e8f0;
    }

    /* ── Root Background ── */
    .stApp {
        background: #080c10;
    }
    .main .block-container {
        padding: 2.5rem 3rem 4rem 3rem;
        max-width: 1400px;
    }

    /* ── Hide default streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── HEADER ── */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 4.2rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -2px;
        color: #f1f5f9;
        margin: 0;
    }
    .hero-title span {
        color: transparent;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #38bdf8;
        margin: 10px 0 6px 0;
    }
    .hero-desc {
        font-size: 0.95rem;
        color: #64748b;
        max-width: 460px;
        line-height: 1.7;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 28px 0;
    }

    /* ── Panel Labels ── */
    .panel-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 12px;
    }

    /* ── Upload Area ── */
    .stFileUploader > div {
        background: rgba(255,255,255,0.03) !important;
        border: 1.5px dashed rgba(255,255,255,0.1) !important;
        border-radius: 16px !important;
        transition: border-color 0.3s;
    }
    .stFileUploader > div:hover {
        border-color: rgba(56,189,248,0.35) !important;
    }
    .stFileUploader label { color: #64748b !important; }

    /* ── Image display ── */
    .stImage img {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.07);
    }

    /* ── CTA Button ── */
    .stButton > button {
        width: 100%;
        height: 3.5em;
        border-radius: 12px;
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: #fff;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 30px rgba(99,102,241,0.15);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(99,102,241,0.35);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* ── CNN Visualizer ── */
    .cnn-section {
        margin: 32px 0 20px 0;
    }
    .cnn-wrap {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 28px 24px 20px 24px;
        overflow-x: auto;
    }
    .cnn-label-title {
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 18px;
    }

    /* ── Result Card ── */
    .result-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px;
        margin-top: 20px;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--result-color, #38bdf8);
        border-radius: 20px 20px 0 0;
    }
    .result-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 6px;
    }
    .result-name {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
    }
    .result-advice {
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.6;
        margin: 0 0 20px 0;
    }
    .confidence-row {
        display: flex;
        justify-content: space-between;
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: #475569;
        margin-bottom: 8px;
    }
    .conf-value {
        color: #e2e8f0;
    }
    .conf-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 99px;
        height: 4px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 99px;
        background: var(--result-color, #38bdf8);
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Prob rows ── */
    .prob-table {
        margin-top: 20px;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .prob-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .prob-name {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: #64748b;
        width: 140px;
        flex-shrink: 0;
    }
    .prob-bar-bg {
        flex: 1;
        background: rgba(255,255,255,0.05);
        border-radius: 99px;
        height: 3px;
        overflow: hidden;
    }
    .prob-bar {
        height: 100%;
        border-radius: 99px;
    }
    .prob-pct {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: #475569;
        width: 42px;
        text-align: right;
        flex-shrink: 0;
    }

    /* ── Legend pills ── */
    .legend-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-top: 18px;
    }
    .legend-card {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.05);
        background: rgba(14,23,42,0.6);
        box-shadow: inset 0 0 0 1px rgba(15,23,42,0.4);
    }
    .legend-dot {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .legend-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: -0.2px;
        color: #e2e8f0;
        margin: 0;
    }
    .legend-desc {
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 1.5px;
        color: #475569;
        margin: 2px 0 0 0;
        text-transform: uppercase;
    }

    /* ── Architecture section ── */
    .arch-section {
        border-radius: 28px;
        padding: 32px;
        background: radial-gradient(circle at 0% 0%, rgba(14,165,233,0.12), transparent 55%),
                    radial-gradient(circle at 100% 0%, rgba(99,102,241,0.12), transparent 50%),
                    rgba(2,6,23,0.85);
        border: 1px solid rgba(148,163,184,0.08);
        box-shadow: 0 20px 45px rgba(2,6,23,0.45);
        position: relative;
        overflow: hidden;
    }
    .arch-section::after {
        content: '';
        position: absolute;
        inset: 12px;
        border-radius: 22px;
        border: 1px solid rgba(248,250,252,0.03);
        pointer-events: none;
    }
    .arch-head {
        display: flex;
        justify-content: space-between;
        gap: 28px;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    .arch-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 3px;
        color: #38bdf8;
        text-transform: uppercase;
        margin: 0 0 10px 0;
    }
    .arch-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.3rem;
        letter-spacing: -1px;
        margin: 0 0 10px 0;
    }
    .arch-desc {
        font-size: 0.92rem;
        color: #94a3b8;
        margin: 0;
        max-width: 520px;
        line-height: 1.7;
    }
    .arch-pill {
        align-self: flex-start;
        padding: 14px 18px;
        border-radius: 16px;
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(148,163,184,0.2);
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #cbd5f5;
        text-align: right;
    }
    .arch-pill span {
        display: block;
        font-size: 0.65rem;
        color: #64748b;
        margin-top: 4px;
    }
    .arch-body {
        margin-top: 28px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        position: relative;
        z-index: 1;
    }

    /* ── Status messages ── */
    .stStatus { border-radius: 12px !important; }
    .stAlert { border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.06) !important; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 1px;
        color: #475569 !important;
    }
    
    /* ── Footer ── */
    .footer-text {
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 2px;
        text-align: center;
        color: #1e293b;
        margin-top: 60px;
        text-transform: uppercase;
    }

    /* ── Progress bar override ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #0ea5e9, #818cf8) !important;
        border-radius: 99px !important;
    }
    .stProgress > div > div {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 99px !important;
        height: 4px !important;
    }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CNN ARCHITECTURE SVG VISUALIZER
# ─────────────────────────────────────────────
CNN_VIZ_HTML = """
<div style="overflow-x:auto; padding-bottom: 8px;">
<svg viewBox="0 0 900 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;min-width:700px;height:160px;font-family:'DM Mono',monospace;">

  <defs>
    <linearGradient id="gConv" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#38bdf8;stop-opacity:0.9"/>
      <stop offset="100%" style="stop-color:#0ea5e9;stop-opacity:0.4"/>
    </linearGradient>
    <linearGradient id="gPool" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#818cf8;stop-opacity:0.9"/>
      <stop offset="100%" style="stop-color:#6366f1;stop-opacity:0.4"/>
    </linearGradient>
    <linearGradient id="gDense" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#34d399;stop-opacity:0.9"/>
      <stop offset="100%" style="stop-color:#10b981;stop-opacity:0.4"/>
    </linearGradient>
    <linearGradient id="gDrop" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#f59e0b;stop-opacity:0.7"/>
      <stop offset="100%" style="stop-color:#d97706;stop-opacity:0.3"/>
    </linearGradient>
    <linearGradient id="gOut" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#fb7185;stop-opacity:0.9"/>
      <stop offset="100%" style="stop-color:#e11d48;stop-opacity:0.4"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <!-- INPUT -->
  <g transform="translate(18,0)">
    <rect x="0" y="30" width="44" height="90" rx="6" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>
    <text x="22" y="136" text-anchor="middle" fill="#475569" font-size="7.5" letter-spacing="0.5">INPUT</text>
    <text x="22" y="146" text-anchor="middle" fill="#334155" font-size="6.5">150×150</text>
    <line x1="14" y1="30" x2="14" y2="120" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>
    <line x1="22" y1="30" x2="22" y2="120" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>
    <line x1="30" y1="30" x2="30" y2="120" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>
    <line x1="0" y1="50" x2="44" y2="50" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>
    <line x1="0" y1="68" x2="44" y2="68" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>
    <line x1="0" y1="86" x2="44" y2="86" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>
    <line x1="0" y1="102" x2="44" y2="102" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>
  </g>

  <!-- Arrow -->
  <line x1="66" y1="75" x2="80" y2="75" stroke="#1e3a4a" stroke-width="1.2"/>
  <polygon points="80,71 86,75 80,79" fill="#1e3a4a"/>

  <!-- CONV1 -->
  <g transform="translate(88,0)" filter="url(#glow)">
    <rect x="0" y="20" width="36" height="100" rx="7" fill="url(#gConv)" opacity="0.85"/>
    <rect x="5" y="24" width="36" height="100" rx="7" fill="url(#gConv)" opacity="0.35"/>
    <rect x="10" y="28" width="36" height="100" rx="7" fill="url(#gConv)" opacity="0.15"/>
    <text x="23" y="136" text-anchor="middle" fill="#38bdf8" font-size="7.5" letter-spacing="0.5">CONV</text>
    <text x="23" y="146" text-anchor="middle" fill="#0369a1" font-size="6.5">32 · 3×3</text>
  </g>

  <!-- Arrow -->
  <line x1="140" y1="75" x2="153" y2="75" stroke="#1e3a4a" stroke-width="1.2"/>
  <polygon points="153,71 159,75 153,79" fill="#1e3a4a"/>

  <!-- POOL1 -->
  <g transform="translate(161,10)" filter="url(#glow)">
    <rect x="0" y="20" width="30" height="80" rx="7" fill="url(#gPool)" opacity="0.85"/>
    <rect x="5" y="24" width="30" height="80" rx="7" fill="url(#gPool)" opacity="0.3"/>
    <text x="20" y="128" text-anchor="middle" fill="#818cf8" font-size="7.5" letter-spacing="0.5">POOL</text>
    <text x="20" y="138" text-anchor="middle" fill="#4338ca" font-size="6.5">2×2</text>
  </g>

  <!-- Arrow -->
  <line x1="197" y1="75" x2="211" y2="75" stroke="#1e3a4a" stroke-width="1.2"/>
  <polygon points="211,71 217,75 211,79" fill="#1e3a4a"/>

  <!-- CONV2 -->
  <g transform="translate(219,5)" filter="url(#glow)">
    <rect x="0" y="20" width="38" height="90" rx="7" fill="url(#gConv)" opacity="0.85"/>
    <rect x="5" y="24" width="38" height="90" rx="7" fill="url(#gConv)" opacity="0.35"/>
    <rect x="10" y="28" width="38" height="90" rx="7" fill="url(#gConv)" opacity="0.15"/>
    <text x="24" y="136" text-anchor="middle" fill="#38bdf8" font-size="7.5" letter-spacing="0.5">CONV</text>
    <text x="24" y="146" text-anchor="middle" fill="#0369a1" font-size="6.5">64 · 3×3</text>
  </g>

  <!-- Arrow -->
  <line x1="272" y1="75" x2="286" y2="75" stroke="#1e3a4a" stroke-width="1.2"/>
  <polygon points="286,71 292,75 286,79" fill="#1e3a4a"/>

  <!-- POOL2 -->
  <g transform="translate(294,15)" filter="url(#glow)">
    <rect x="0" y="20" width="26" height="68" rx="7" fill="url(#gPool)" opacity="0.85"/>
    <rect x="5" y="24" width="26" height="68" rx="7" fill="url(#gPool)" opacity="0.3"/>
    <text x="18" y="105" text-anchor="middle" fill="#818cf8" font-size="7.5" letter-spacing="0.5">POOL</text>
    <text x="18" y="115" text-anchor="middle" fill="#4338ca" font-size="6.5">2×2</text>
  </g>

  <!-- Arrow -->
  <line x1="327" y1="75" x2="341" y2="75" stroke="#1e3a4a" stroke-width="1.2"/>
  <polygon points="341,71 347,75 341,79" fill="#1e3a4a"/>

  <!-- CONV3 -->
  <g transform="translate(349,8)" filter="url(#glow)">
    <rect x="0" y="20" width="40" height="84" rx="7" fill="url(#gConv)" opacity="0.85"/>
    <rect x="5" y="24" width="40" height="84" rx="7" fill="url(#gConv)" opacity="0.35"/>
    <rect x="10" y="28" width="40" height="84" rx="7" fill="url(#gConv)" opacity="0.15"/>
    <text x="25" y="136" text-anchor="middle" fill="#38bdf8" font-size="7.5" letter-spacing="0.5">CONV</text>
    <text x="25" y="146" text-anchor="middle" fill="#0369a1" font-size="6.5">128 · 3×3</text>
  </g>

  <!-- Arrow -->
  <line x1="405" y1="75" x2="419" y2="75" stroke="#1e3a4a" stroke-width="1.2"/>
  <polygon points="419,71 425,75 419,79" fill="#1e3a4a"/>

  <!-- POOL3 -->
  <g transform="translate(427,22)" filter="url(#glow)">
    <rect x="0" y="20" width="22" height="56" rx="7" fill="url(#gPool)" opacity="0.85"/>
    <rect x="5" y="24" width="22" height="56" rx="7" fill="url(#gPool)" opacity="0.3"/>
    <text x="16" y="95" text-anchor="middle" fill="#818cf8" font-size="7.5" letter-spacing="0.5">POOL</text>
    <text x="16" y="105" text-anchor="middle" fill="#4338ca" font-size="6.5">2×2</text>
  </g>

  <!-- FLATTEN arrow -->
  <line x1="455" y1="75" x2="470" y2="75" stroke="#1e3a4a" stroke-width="1.2" stroke-dasharray="3,2"/>
  <polygon points="470,71 476,75 470,79" fill="#1e3a4a"/>
  <text x="462" y="68" text-anchor="middle" fill="#334155" font-size="6">flatten</text>

  <!-- DENSE1 -->
  <g transform="translate(478,0)" filter="url(#glow)">
    <rect x="0" y="25" width="16" height="98" rx="6" fill="url(#gDense)" opacity="0.9"/>
    <circle cx="8" cy="38" r="2.2" fill="rgba(255,255,255,0.5)"/>
    <circle cx="8" cy="50" r="2.2" fill="rgba(255,255,255,0.5)"/>
    <circle cx="8" cy="62" r="2.2" fill="rgba(255,255,255,0.5)"/>
    <circle cx="8" cy="74" r="2.2" fill="rgba(255,255,255,0.5)"/>
    <circle cx="8" cy="86" r="2.2" fill="rgba(255,255,255,0.5)"/>
    <circle cx="8" cy="98" r="2.2" fill="rgba(255,255,255,0.5)"/>
    <circle cx="8" cy="110" r="1.5" fill="rgba(255,255,255,0.2)"/>
    <text x="8" y="138" text-anchor="middle" fill="#34d399" font-size="7.5" letter-spacing="0.5">DENSE</text>
    <text x="8" y="148" text-anchor="middle" fill="#065f46" font-size="6.5">512</text>
  </g>

  <!-- Arrow -->
  <line x1="497" y1="75" x2="512" y2="75" stroke="#1e3a4a" stroke-width="1.2"/>
  <polygon points="512,71 518,75 512,79" fill="#1e3a4a"/>

  <!-- DROPOUT -->
  <g transform="translate(520,0)">
    <rect x="0" y="25" width="16" height="98" rx="6" fill="url(#gDrop)" opacity="0.7"/>
    <circle cx="8" cy="38" r="2.2" fill="rgba(255,255,255,0.5)"/>
    <circle cx="8" cy="50" r="2.2" fill="rgba(255,255,255,0.15)"/>
    <circle cx="8" cy="62" r="2.2" fill="rgba(255,255,255,0.5)"/>
    <circle cx="8" cy="74" r="2.2" fill="rgba(255,255,255,0.15)"/>
    <circle cx="8" cy="86" r="2.2" fill="rgba(255,255,255,0.5)"/>
    <circle cx="8" cy="98" r="2.2" fill="rgba(255,255,255,0.15)"/>
    <circle cx="8" cy="110" r="1.5" fill="rgba(255,255,255,0.08)"/>
    <text x="8" y="138" text-anchor="middle" fill="#f59e0b" font-size="7.5" letter-spacing="0.5">DROP</text>
    <text x="8" y="148" text-anchor="middle" fill="#92400e" font-size="6.5">0.5</text>
  </g>

  <!-- Arrow -->
  <line x1="539" y1="75" x2="554" y2="75" stroke="#1e3a4a" stroke-width="1.2"/>
  <polygon points="554,71 560,75 554,79" fill="#1e3a4a"/>

  <!-- OUTPUT -->
  <g transform="translate(562,0)" filter="url(#glow)">
    <rect x="0" y="44" width="16" height="54" rx="6" fill="url(#gOut)" opacity="0.9"/>
    <circle cx="8" cy="56" r="2.5" fill="rgba(255,255,255,0.7)"/>
    <circle cx="8" cy="71" r="2.5" fill="rgba(255,255,255,0.7)"/>
    <circle cx="8" cy="86" r="2.5" fill="rgba(255,255,255,0.7)"/>
    <text x="8" y="114" text-anchor="middle" fill="#fb7185" font-size="7.5" letter-spacing="0.5">SOFTMAX</text>
    <text x="8" y="124" text-anchor="middle" fill="#9f1239" font-size="6.5">3 classes</text>
  </g>

  <!-- LEGEND bottom -->
  <g transform="translate(18, 152)">
    <rect x="0" y="0" width="8" height="5" rx="1.5" fill="url(#gConv)"/>
    <text x="12" y="5" fill="#475569" font-size="6.5">Conv+ReLU</text>
    <rect x="66" y="0" width="8" height="5" rx="1.5" fill="url(#gPool)"/>
    <text x="78" y="5" fill="#475569" font-size="6.5">MaxPool</text>
    <rect x="124" y="0" width="8" height="5" rx="1.5" fill="url(#gDense)"/>
    <text x="136" y="5" fill="#475569" font-size="6.5">Dense</text>
    <rect x="172" y="0" width="8" height="5" rx="1.5" fill="url(#gDrop)"/>
    <text x="184" y="5" fill="#475569" font-size="6.5">Dropout</text>
    <rect x="224" y="0" width="8" height="5" rx="1.5" fill="url(#gOut)"/>
    <text x="236" y="5" fill="#475569" font-size="6.5">Output</text>
  </g>
</svg>
</div>
"""

CLASS_LEGEND_HTML = """
<div class="legend-grid">
    <div class="legend-card">
        <span class="legend-dot" style="background:#ff4b4b;"></span>
        <div>
            <p class="legend-label">Critical Risk</p>
            <p class="legend-desc">Immediate care</p>
        </div>
    </div>
    <div class="legend-card">
        <span class="legend-dot" style="background:#10b981;"></span>
        <div>
            <p class="legend-label">Healthy · Low Risk</p>
            <p class="legend-desc">Monitoring only</p>
        </div>
    </div>
    <div class="legend-card">
        <span class="legend-dot" style="background:#fbbf24;"></span>
        <div>
            <p class="legend-label">Medium Risk</p>
            <p class="legend-desc">Schedule follow-up</p>
        </div>
    </div>
</div>
"""

ARCHITECTURE_SECTION_HTML = f"""
<div class="arch-section">
    <div class="arch-head">
        <div>
            <p class="arch-eyebrow">— Model Architecture</p>
            <h3 class="arch-title">Signal-tuned convolutional chain</h3>
            <p class="arch-desc">Triple convolutional towers compress dermal textures into a dense triage head with dropout regularization for robust three-way risk scoring.</p>
        </div>
        <div class="arch-pill">
            TensorFlow Core
            <span>Input grid · 150×150</span>
        </div>
    </div>
    <div class="arch-body">
        <div class="cnn-wrap">{CNN_VIZ_HTML}</div>
        {CLASS_LEGEND_HTML}
    </div>
</div>
"""

# ─────────────────────────────────────────────
# MODEL DOWNLOAD FROM GOOGLE DRIVE
# ─────────────────────────────────────────────

MODEL_PATH = 'dog_health_cnn.keras'
# Google Drive file ID extracted from the share link
GDRIVE_FILE_ID = '191Qc-HxJPF2MCrK_gGMY64pWLec6Px2D'

def download_model_from_gdrive(file_id: str, dest_path: str) -> bool:
    """
    Downloads a file from Google Drive using gdown.
    Returns True on success, False on failure.
    """
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False, fuzzy=True)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10_000:
            return True
        return False
    except Exception as e:
        st.error(f"gdown error: {e}")
        return False


# ─────────────────────────────────────────────
# ANIMATION ASSETS
# ─────────────────────────────────────────────
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

dog_anim = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_syqnfe7c.json")

# ─────────────────────────────────────────────
# MODEL RECONSTRUCTION + LOADING
# ─────────────────────────────────────────────
def build_model_manually():
    model = models.Sequential([
        layers.Input(shape=(150, 150, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    return model


@st.cache_resource(show_spinner=False)
def load_trained_model():
    """
    Loads the CNN model. Auto-downloads from Google Drive if missing.
    Uses gdown for reliable GDrive large-file support.
    """
    # Auto-install gdown if not in environment
    try:
        import gdown  # noqa: F401
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from Google Drive (first run only)..."):
            success = download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)
        if not success:
            st.error(
                "Could not download model weights from Google Drive. "
                "Please check your internet connection or place dog_health_cnn.keras "
                "manually in the app directory."
            )
            return None

    try:
        model = build_model_manually()
        model.load_weights(MODEL_PATH)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        return None

model = load_trained_model()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_title, col_lottie = st.columns([5, 1])
with col_title:
    st.markdown("<p class='hero-sub'>Deep Learning · Health Classification</p>", unsafe_allow_html=True)
    st.markdown("<h1 class='hero-title'>Canine<span>Care</span> AI</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-desc'>CNN-powered dermal risk classification for stray dog populations "
        "— bacterial, fungal &amp; mange pattern detection.</p>",
        unsafe_allow_html=True
    )
with col_lottie:
    if dog_anim:
        st_lottie(dog_anim, height=130, key="hero_dog")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CNN ARCHITECTURE VISUALIZER
# ─────────────────────────────────────────────
st.markdown(ARCHITECTURE_SECTION_HTML, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN WORKSPACE
# ─────────────────────────────────────────────
left_panel, right_panel = st.columns([1, 1], gap="large")

with left_panel:
    st.markdown("<p class='panel-label'>— 01 / Upload Subject</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Target Image", use_container_width=True)

with right_panel:
    st.markdown("<p class='panel-label'>— 02 / Neural Diagnostics</p>", unsafe_allow_html=True)

    if not uploaded_file:
        st.markdown("""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
            border-radius:16px;padding:40px 28px;text-align:center;margin-top:4px;">
                <p style="font-family:'DM Mono',monospace;font-size:0.7rem;letter-spacing:2px;
                color:#334155;text-transform:uppercase;margin-bottom:10px;">System Ready</p>
                <p style="color:#1e3a5f;font-size:0.85rem;">Upload an image on the left<br>
                to initiate dermal scanning.</p>
            </div>
        """, unsafe_allow_html=True)

    elif model is None:
        st.error(
            "⚠️ Model not loaded. Place `dog_health_cnn.keras` in the app directory "
            "or ensure internet access so it can be downloaded automatically."
        )

    else:
        if st.button("EXECUTE NEURAL ANALYSIS"):
            with st.status("Initializing Neural Engine...", expanded=True) as status:
                st.write("Loading weight tensors...")
                time.sleep(0.4)
                st.write("Applying feature kernels...")

                # --- PREPROCESSING ---
                img_resized = img.resize((150, 150), Image.Resampling.LANCZOS)
                img_array = np.array(img_resized).astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # --- INFERENCE ---
                predictions = model.predict(img_array)
                result_idx = int(np.argmax(predictions))
                confidence = float(np.max(predictions)) * 100

                status.update(label="Diagnostic Complete", state="complete", expanded=False)

            # --- MAPPING ---
            categories = ["Critical Risk", "Healthy (Low Risk)", "Medium Risk"]
            colors     = ["#ff4b4b", "#10b981", "#fbbf24"]
            icons      = ["⚠️", "✅", "🔍"]
            advice = [
                "URGENT: Bacterial infection or severe mange detected. Immediate veterinary intervention required.",
                "NORMAL: No significant dermal abnormalities detected. General monitoring advised.",
                "CAUTION: Potential fungal or allergic dermatitis patterns observed. Schedule follow-up."
            ]
            cat_bar_colors = ["#ff4b4b", "#10b981", "#fbbf24"]

            # Probability bars HTML
            prob_bars_html = ""
            for i, (cat, prob) in enumerate(zip(categories, predictions[0])):
                pct = float(prob) * 100
                prob_bars_html += f"""
                <div class="prob-row">
                    <span class="prob-name">{cat}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar" style="width:{pct:.1f}%;background:{cat_bar_colors[i]};opacity:0.7;"></div>
                    </div>
                    <span class="prob-pct">{pct:.1f}%</span>
                </div>
                """

            st.markdown(f"""
                <div class="result-card" style="--result-color:{colors[result_idx]};">
                    <p class="result-label">Diagnosis Result</p>
                    <p class="result-name" style="color:{colors[result_idx]};">{icons[result_idx]} {categories[result_idx]}</p>
                    <p class="result-advice">{advice[result_idx]}</p>
                    <div class="confidence-row">
                        <span>AI Confidence</span>
                        <span class="conf-value">{confidence:.2f}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{confidence:.1f}%;background:{colors[result_idx]};"></div>
                    </div>
                    <div class="prob-table">
                        {prob_bars_html}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("Raw Neural Output"):
                st.write(f"Class Probabilities: {predictions}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)

