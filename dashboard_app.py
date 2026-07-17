from pathlib import Path
from collections import Counter
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

from tabpfn_multiclass import build_pipeline as build_tabpfn_pipeline

HERE = Path(__file__).parent

CLASS_META = {
    "stress":  {"label": "Stress",   "icon": "🧠", "color": "#FF453A"},
    "aerobic": {"label": "Ausdauer", "icon": "🚴", "color": "#30D158"},
    "sprint":  {"label": "Sprint",   "icon": "⚡", "color": "#FF9F0A"},
    "rest":    {"label": "Ruhe",     "icon": "🌙", "color": "#64D2FF"},
}
DAY_START_MIN, DAY_END_MIN = 8 * 60, 22 * 60
SMOOTH_K = 5
MIN_BLOCK_MIN = 2.0

BG, CARD, CARD2 = "#000000", "#1C1C1E", "#2C2C2E"
TXT, TXT2, SEP = "#F2F2F7", "#98989D", "#38383A"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": CARD, "axes.facecolor": CARD,
    "savefig.facecolor": CARD,
    "axes.edgecolor": SEP, "axes.linewidth": 0.8,
    "text.color": TXT, "axes.labelcolor": TXT,
    "xtick.color": TXT2, "ytick.color": TXT2,
    "grid.color": "#48484A",
})

st.set_page_config(page_title="Wearable Tagesübersicht", page_icon="⌚", layout="wide")

APPLE_CSS = f"""
<style>
#MainMenu, header, footer {{visibility: hidden;}}
.stApp {{ background: {BG}; }}
.block-container {{ max-width: 1060px; padding-top: 1.4rem; padding-bottom: 3rem; }}
html, body, [class*="css"], .stMarkdown, p, span, div {{
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
  -webkit-font-smoothing: antialiased; color: {TXT};
}}
.hero {{ display:flex; align-items:center; justify-content:space-between;
  background: linear-gradient(135deg, #1C1C1E 0%, #2C2C2E 100%);
  border: 1px solid {SEP};
  color:{TXT}; border-radius:26px; padding:26px 30px; margin-bottom:22px;
  box-shadow: 0 10px 30px rgba(0,0,0,.4); }}
.hero h1 {{ font-size:1.9rem; font-weight:700; letter-spacing:-.03em; margin:0; color:{TXT}; }}
.hero .sub {{ color:{TXT2}; font-size:.98rem; margin-top:4px; }}
.hero .pill {{ background:rgba(255,255,255,.08); border-radius:16px; padding:12px 18px; text-align:center; }}
.hero .pill .k {{ font-size:1.6rem; font-weight:700; color:{TXT}; }}
.hero .pill .l {{ font-size:.72rem; color:{TXT2}; text-transform:uppercase; letter-spacing:.05em;}}
.section-title {{ font-size:1.15rem; font-weight:700; letter-spacing:-.01em; margin:18px 2px 10px; color:{TXT}; }}
.balance-card {{ border-radius:22px; padding:18px 10px; text-align:center; border:1px solid {SEP}; }}
.balance-card .ic {{ font-size:1.9rem; }}
.balance-card .lab {{ font-weight:600; font-size:.9rem; margin-top:2px; }}
.balance-card .val {{ font-size:1.7rem; font-weight:800; letter-spacing:-.02em; color:{TXT}; }}
.balance-card .unit {{ font-size:.8rem; color:{TXT2}; }}
.reco {{ border-radius:22px; padding:20px 24px; font-size:1.02rem; line-height:1.5; color:{TXT}; }}
.stat-chip {{ display:inline-block; background:{CARD2}; border:1px solid {SEP}; border-radius:16px;
  padding:12px 18px; margin:4px 8px 4px 0; text-align:center; }}
.stat-chip .k {{ font-size:1.35rem; font-weight:700; letter-spacing:-.02em; color:{TXT}; }}
.stat-chip .l {{ font-size:.72rem; color:{TXT2}; text-transform:uppercase; letter-spacing:.04em;}}
.legend {{ text-align:center; color:{TXT2}; margin-top:2px; font-size:.92rem; }}
.foot {{ color:#6E6E73; font-size:.82rem; margin-top:16px; }}
div[data-testid="stImage"] img {{ border-radius:20px; }}
label, .stSelectbox label {{ color:{TXT2} !important; }}
</style>
"""
st.markdown(APPLE_CSS, unsafe_allow_html=True)



def available_subjects():
    return [p.name.replace("dashboard_demo_", "").replace("_bundle.joblib", "")
            for p in sorted(HERE.glob("dashboard_demo_*_bundle.joblib"))]


@st.cache_resource(show_spinner="Modell wird geladen …")
def load_model(subject):
    bundle = joblib.load(HERE / f"dashboard_demo_{subject}_bundle.joblib")
    pipe = build_tabpfn_pipeline()
    pipe.fit(bundle["X_train"], bundle["y_train"])
    return bundle, pipe


@st.cache_data(show_spinner=False)
def load_windows(subject):
    return pd.read_csv(HERE / f"dashboard_demo_{subject}_windows.csv", sep=";")


@st.cache_data(show_spinner=False)
def load_signals(subject):
    return pd.read_csv(HERE / f"dashboard_demo_{subject}_signals.csv", sep=";")


@st.cache_data(show_spinner="Vorhersage läuft …")
def predict(subject, _pipe, windows, feature_cols, classes):
    proba = _pipe.predict_proba(windows[feature_cols])
    return np.array([classes[i] for i in proba.argmax(1)])


def min_to_clock(m):
    m = int(round(m)); return f"{m // 60:02d}:{m % 60:02d}"



def session_blocks(win_sess, step_sec):
    
    preds = win_sess["pred"].tolist()
    times = win_sess["clock_center_min"].tolist()
    step = step_sec / 60
    h = SMOOTH_K // 2
    sm = [Counter(preds[max(0, i - h):min(len(preds), i + h + 1)]).most_common(1)[0][0]
          for i in range(len(preds))]
    blocks = []
    for i, p in enumerate(sm):
        t0, t1 = times[i] - step / 2, times[i] + step / 2
        if blocks and blocks[-1]["cls"] == p:
            blocks[-1]["end"] = t1
        else:
            blocks.append({"cls": p, "start": t0, "end": t1})
    changed = True
    while changed and len(blocks) > 1:
        changed = False
        for i, b in enumerate(blocks):
            if b["end"] - b["start"] < MIN_BLOCK_MIN:
                if i == 0:
                    b["cls"] = blocks[1]["cls"]
                elif i == len(blocks) - 1:
                    b["cls"] = blocks[-2]["cls"]
                else:
                    left, right = blocks[i - 1], blocks[i + 1]
                    b["cls"] = (left if (left["end"] - left["start"]) >=
                                (right["end"] - right["start"]) else right)["cls"]
                changed = True
                break
        if changed:
            merged = []
            for b in blocks:
                if merged and merged[-1]["cls"] == b["cls"]:
                    merged[-1]["end"] = b["end"]
                else:
                    merged.append(dict(b))
            blocks = merged
    return blocks


def all_day_blocks(windows, step_sec):
    per_sess, day, spans = {}, [], []
    step = step_sec / 60
    for sess, grp in windows.groupby("session"):
        grp = grp.sort_values("clock_center_min")
        blks = session_blocks(grp, step_sec)
        for b in blks:
            b["session"] = sess
        per_sess[sess] = blks
        day.extend(blks)
        spans.append({"session": sess, "phase_text": grp["phase_text"].iloc[0],
                      "start": grp["clock_center_min"].min() - step / 2,
                      "end":   grp["clock_center_min"].max() + step / 2})
    day.sort(key=lambda b: b["start"])
    spans.sort(key=lambda b: b["start"])
    return day, per_sess, spans



def timeline_range(blocks, pad=30):
    lo = max(0, min(b["start"] for b in blocks) - pad)
    hi = max(b["end"] for b in blocks) + pad
    return lo, hi


def build_overview(day_blocks):
    lo, hi = timeline_range(day_blocks, pad=15)
    fig = go.Figure()
    for b in day_blocks:
        m = CLASS_META[b["cls"]]
        fig.add_trace(go.Bar(
            x=[b["end"] - b["start"]], base=[b["start"]], y=["Tag"],
            orientation="h", width=0.92,
            marker=dict(color=m["color"], cornerradius=3),
            customdata=[[b["session"]]],
            hovertemplate=(f"<b>{m['icon']} {m['label']}</b><br>"
                           f"{min_to_clock(b['start'])} – {min_to_clock(b['end'])}"
                           f"<extra></extra>"),
            showlegend=False,
        ))
    first = ((int(lo) + 29) // 30) * 30
    ticks = list(range(first, int(hi) + 1, 30))
    fig.update_layout(
        barmode="overlay", height=230, bargap=0.0,
        margin=dict(l=10, r=10, t=6, b=6),
        paper_bgcolor=BG, plot_bgcolor=CARD2,
        dragmode=False, clickmode="event+select",
        hoverlabel=dict(bgcolor=CARD2, font=dict(color=TXT, size=14)),
        xaxis=dict(range=[lo, hi], tickvals=ticks,
                   ticktext=[min_to_clock(t) for t in ticks],
                   tickfont=dict(color=TXT, size=14),
                   showgrid=True, gridcolor="#3A3A3C", gridwidth=1,
                   ticks="outside", ticklen=6, tickcolor="#3A3A3C",
                   zeroline=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True, range=[-0.5, 0.5]),
    )
    return fig


def plot_session_big(blocks):
    """Vergrößerte Timeline EINER Session mit genauen Von-Bis-Zeiten je Block."""
    lo, hi = timeline_range(blocks, pad=3)
    fig, ax = plt.subplots(figsize=(13, 2.2), dpi=200)
    ax.axhspan(0.3, 0.95, color=CARD2, zorder=0)
    for b in blocks:
        m = CLASS_META[b["cls"]]
        ax.axvspan(b["start"], b["end"], 0.3, 0.95, color=m["color"], zorder=2)
        mid = (b["start"] + b["end"]) / 2
        if b["end"] - b["start"] >= 3.5:
            ax.text(mid, 0.62, m["label"], ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold", zorder=3)
        ax.text(mid, 0.12, f"{min_to_clock(b['start'])}–{min_to_clock(b['end'])}",
                ha="center", va="center", color=TXT2, fontsize=8.5, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    first = ((int(lo) + 4) // 5) * 5
    ticks = list(range(first, int(hi) + 1, 5))
    ax.set_xticks(ticks)
    ax.set_xticklabels([min_to_clock(t) for t in ticks], fontsize=9)
    ax.tick_params(length=0, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def plot_session_detail(sig_sess, blocks):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 4.3), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    x = sig_sess["clock_min"].to_numpy()
    for b in blocks:
        ax1.axvspan(b["start"], b["end"], color=CLASS_META[b["cls"]]["color"],
                    alpha=0.16, zorder=0)
    ax1.plot(x, sig_sess["hr"], color="#FF453A", lw=1.7, zorder=3)
    ax1.set_ylabel("Herzfrequenz (bpm)", fontsize=10, fontweight="bold")
    ax1.grid(alpha=0.15)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.fill_between(x, sig_sess["move"], color="#8E8E93", alpha=0.7, lw=0)
    ax2.set_ylabel("Bewegung", fontsize=10, fontweight="bold")
    ax2.grid(alpha=0.15)
    ax2.spines[["top", "right"]].set_visible(False)

    ticks = np.linspace(x.min(), x.max(), 8)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([min_to_clock(t) for t in ticks], fontsize=8)
    fig.tight_layout()
    return fig



def main():
    subs = available_subjects()
    if not subs:
        st.error("Keine vorbereiteten Probanden gefunden. Bitte zuerst "
                 "`python dashboard_demo_prep.py <Proband>` ausführen.")
        return

    top_l, top_r = st.columns([3, 1])
    with top_r:
        subject = st.selectbox("Person / Proband", subs, index=0)

    bundle, pipe = load_model(subject)
    windows = load_windows(subject).copy()
    signals = load_signals(subject)
    feature_cols, classes = bundle["feature_cols"], bundle["label_classes"]
    step_sec = bundle.get("step_sec", 20)

    pred = predict(subject, pipe, windows, feature_cols, classes)
    windows["pred"] = pred
    day_blocks, per_sess, _spans = all_day_blocks(windows, step_sec)
    minutes = {c: float((pred == c).sum() * step_sec / 60) for c in CLASS_META}

    with top_l:
        st.markdown(
            f"""<div class="hero">
              <div>
                <h1>Guten Abend, {bundle.get('person_name', 'Demo')} 👋</h1>
                <div class="sub">Deine Wearable-Tagesübersicht &nbsp;·&nbsp; Smartwatch (Puls + Bewegung)</div>
              </div>
              <div style="display:flex; gap:12px;">
                <div class="pill"><div class="k">{minutes['stress']:.0f}</div><div class="l">min Stress</div></div>
                <div class="pill"><div class="k">{minutes['aerobic']+minutes['sprint']:.0f}</div><div class="l">min Sport</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dein Tag im Überblick</div>', unsafe_allow_html=True)
    event = st.plotly_chart(build_overview(day_blocks), use_container_width=True,
                            key="overview", on_select="rerun",
                            selection_mode=("points",),
                            config={"displayModeBar": False})
    leg = " &nbsp;&nbsp; ".join(
        f"<span style='color:{m['color']};font-size:1.15em'>●</span> {m['icon']} {m['label']}"
        for m in CLASS_META.values())
    leg += (" &nbsp;&nbsp; <span style='color:#6E6E73;font-size:1.15em'>●</span> "
            "keine Aufzeichnung")
    st.markdown(f"<div class='legend'>{leg}</div>", unsafe_allow_html=True)

    sess_order = [s for s in bundle.get("day_plan", {}) if s in per_sess] or list(per_sess.keys())
    if event and event.get("selection", {}).get("points"):
        clicked = event["selection"]["points"][0].get("customdata")
        if clicked:
            st.session_state["sel_session"] = clicked[0]
    if st.session_state.get("sel_session") not in sess_order:
        st.session_state["sel_session"] = sess_order[0]

    st.markdown('<div class="section-title">Tagesbilanz</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, cls in zip(cols, ["stress", "aerobic", "sprint", "rest"]):
        m = CLASS_META[cls]
        col.markdown(
            f"""<div class="balance-card" style="background:{m['color']}22;">
              <div class="ic">{m['icon']}</div>
              <div class="lab" style="color:{m['color']}">{m['label']}</div>
              <div class="val">{minutes[cls]:.0f}<span class="unit"> min</span></div>
            </div>""", unsafe_allow_html=True)

    st.write("")
    stress_min = minutes["stress"]
    exercise_min = minutes["aerobic"] + minutes["sprint"]
    if stress_min > exercise_min * 1.1:
        st.markdown(
            f"""<div class="reco" style="background:rgba(255,159,10,.15);border-left:4px solid #FF9F0A;">
            🏃 <b>Viel Anspannung, wenig Bewegung.</b> Heute {stress_min:.0f} min Stress
            gegenüber nur {exercise_min:.0f} min Sport. Eine lockere Laufrunde oder ein
            Spaziergang würde guttun und den Kopf freimachen.</div>""",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"""<div class="reco" style="background:rgba(48,209,88,.15);border-left:4px solid #30D158;">
            👍 <b>Schöne Balance heute.</b> {exercise_min:.0f} min Bewegung haben die
            {stress_min:.0f} min Anspannung gut ausgeglichen. Weiter so!</div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="section-title">Aktivität im Detail</div>', unsafe_allow_html=True)
    day_plan = bundle.get("day_plan", {})
    labels = {s: f"{day_plan[s][0]} Uhr" if s in day_plan else s for s in sess_order}
    choice = st.selectbox("Session (oder oben auf einen Balken klicken)", sess_order,
                          index=sess_order.index(st.session_state["sel_session"]),
                          format_func=lambda s: labels.get(s, s))
    st.session_state["sel_session"] = choice

    sig_sess = signals[signals["session"] == choice].sort_values("clock_min")
    blocks = per_sess[choice]
    hr = sig_sess["hr"].to_numpy()
    dom = Counter(windows[windows["session"] == choice]["pred"]).most_common(1)[0][0]
    dur = sig_sess["clock_min"].max() - sig_sess["clock_min"].min()
    chips = [("Zeitraum", f"{min_to_clock(blocks[0]['start'])}–{min_to_clock(blocks[-1]['end'])}"),
             ("Ø Puls", f"{np.nanmean(hr):.0f} bpm"),
             ("Max Puls", f"{np.nanmax(hr):.0f} bpm"),
             ("Dauer", f"{dur:.0f} min"),
             ("Hauptzustand", f"{CLASS_META[dom]['icon']} {CLASS_META[dom]['label']}")]
    st.markdown("".join(
        f"<span class='stat-chip'><div class='k'>{v}</div><div class='l'>{k}</div></span>"
        for k, v in chips), unsafe_allow_html=True)

    st.pyplot(plot_session_big(blocks))
    parts = " &nbsp; ".join(
        f"<span style='color:{CLASS_META[b['cls']]['color']}'>●</span> "
        f"{min_to_clock(b['start'])}–{min_to_clock(b['end'])} "
        f"{CLASS_META[b['cls']]['label']} ({b['end']-b['start']:.0f} min)"
        for b in blocks)
    st.markdown(f"<div class='legend' style='text-align:left'>{parts}</div>",
                unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.pyplot(plot_session_detail(sig_sess, blocks))

    st.markdown(
        f"<div class='foot'>Person {bundle['demo_subject']} war nicht im Training "
        f"(das Modell sieht sie zum ersten Mal). Gleitende 60-s-Fenster; die drei "
        f"realen Aufzeichnungen sind zu einem fiktiven Tag angeordnet.</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
