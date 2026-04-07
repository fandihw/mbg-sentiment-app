import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io

# PAGE CONFIG
st.set_page_config(
    page_title="MBG Sentiment & Topic Analysis",
    page_icon="MBG",
    layout="wide",
    initial_sidebar_state="expanded",
)

SENTIMENT_LABELS  = {"negatif": 0, "positif": 1}
TOPIC_LABELS      = {"program": 0, "anggaran": 1, "gizi": 2, "distribusi": 3}
IDX_TO_SENT       = {0: "negatif", 1: "positif"}
IDX_TO_TOPIC      = {0: "program", 1: "anggaran", 2: "gizi", 3: "distribusi"}
MAX_LEN           = 128

# ── GANTI dengan username dan nama repo Hugging Face kamu ──
HF_SENT_MODEL  = "fandihw/mbg-indobert-sentimen/indobert_sentimen"
HF_TOPIC_MODEL = "fandihw/mbg-indobert-topik/indobert_topik"
# ───────────────────────────────────────────────────────────

PALETTE = {
    "negatif"   : "#E74C3C",
    "positif"   : "#2ECC71",
    "program"   : "#3498DB",
    "anggaran"  : "#E67E22",
    "gizi"      : "#9B59B6",
    "distribusi": "#1ABC9C",
}

THESIS_STATS = {
    "total_tweets"    : 5004,
    "total_negatif"   : 4273,
    "total_positif"   : 731,
    "pct_negatif"     : 85.4,
    "pct_positif"     : 14.6,
    "topic_dist"      : {
        "program"   : 2088,
        "anggaran"  : 1197,
        "gizi"      : 1080,
        "distribusi": 639,
    },
    "sent_per_topic"  : {
        "program"   : {"total": 2088, "negatif": 1663, "positif": 425},
        "anggaran"  : {"total": 1197, "negatif": 1156, "positif": 41},
        "gizi"      : {"total": 1080, "negatif": 983,  "positif": 97},
        "distribusi": {"total": 639,  "negatif": 471,  "positif": 168},
    },
    "eval_sentiment"  : {"Accuracy": 0.9101, "Precision": 0.9061, "Recall": 0.9101, "F1-Score": 0.9076},
    "eval_topic"      : {"Accuracy": 0.6404, "Precision": 0.6363, "Recall": 0.6404, "F1-Score": 0.6366},
    "eval_topic_class": {
        "Program"   : {"Precision": 0.6577, "Recall": 0.6435, "F1-Score": 0.6505, "Support": 418},
        "Anggaran"  : {"Precision": 0.6592, "Recall": 0.7303, "F1-Score": 0.6929, "Support": 241},
        "Gizi"      : {"Precision": 0.6422, "Recall": 0.6897, "F1-Score": 0.6651, "Support": 203},
        "Distribusi": {"Precision": 0.5234, "Recall": 0.4029, "F1-Score": 0.4553, "Support": 139},
    },
    "cm_sent" : np.array([[819, 36], [54, 92]]),
    "cm_topic": np.array([
        [289, 64, 53, 32],
        [ 48,176, 10,  7],
        [ 43,  8,140, 12],
        [ 49, 19, 15, 56],
    ]),
    "coherence_score": 0.3596,
    "lda_keywords"   : {
        "Topik 1 (Distribusi)": "catering, sekolah, dapur, kualitas, korupsi, keracunan, porsi, vendor",
        "Topik 2 (Program)"   : "pendidikan, kualitas, guru, negara, keracunan, kebijakan, dana",
        "Topik 3 (Gizi)"      : "anak, gizi, keracunan, kualitas, pelaksanaan, siswa, sekolah, sehat",
        "Topik 4 (Anggaran)"  : "rakyat, anggaran, korupsi, negara, efisiensi, apbn, pejabat, pajak",
    },
}

if "model_sent"   not in st.session_state: st.session_state.model_sent   = None
if "model_top"    not in st.session_state: st.session_state.model_top    = None
if "tok_sent"     not in st.session_state: st.session_state.tok_sent     = None
if "tok_top"      not in st.session_state: st.session_state.tok_top      = None
if "device"       not in st.session_state: st.session_state.device       = None
if "model_loaded" not in st.session_state: st.session_state.model_loaded = False


@st.cache_resource(show_spinner=False)
def load_models():
    """Load fine-tuned IndoBERT models dari Hugging Face Hub."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from huggingface_hub import snapshot_download

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download folder model dari HF
    sent_dir = snapshot_download(
        repo_id="fandihw/mbg-indobert-sentimen",
        allow_patterns="indobert_sentimen/*",
    )
    topik_dir = snapshot_download(
        repo_id="fandihw/mbg-indobert-topik",
        allow_patterns="indobert_topik/*",
    )

    import os
    sent_path  = os.path.join(sent_dir,  "indobert_sentimen")
    topik_path = os.path.join(topik_dir, "indobert_topik")

    # Tokenizer dari base model
    BASE_MODEL = "indobenchmark/indobert-base-p1"
    tok_sent = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok_top  = AutoTokenizer.from_pretrained(BASE_MODEL)

    model_sent = AutoModelForSequenceClassification.from_pretrained(sent_path).to(device)
    model_sent.eval()

    model_top = AutoModelForSequenceClassification.from_pretrained(topik_path).to(device)
    model_top.eval()

    return tok_sent, model_sent, tok_top, model_top, device


def predict_single(text: str, model_sent, tok_sent, model_top, tok_top, device):
    """Prediksi sentimen + topik untuk satu teks."""
    import torch

    def infer(model, tokenizer, text):
        enc = tokenizer(
            text, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
        probs      = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        return pred_idx, confidence, probs

    si, sc, sp = infer(model_sent, tok_sent, text)
    ti, tc, tp = infer(model_top,  tok_top,  text)

    return {
        "sentimen"       : IDX_TO_SENT[si],
        "sentimen_conf"  : f"{sc*100:.2f}%",
        "topik"          : IDX_TO_TOPIC[ti],
        "topik_conf"     : f"{tc*100:.2f}%",
        "detail_sentimen": {IDX_TO_SENT[i]:  f"{p*100:.2f}%" for i, p in enumerate(sp)},
        "detail_topik"   : {IDX_TO_TOPIC[i]: f"{p*100:.2f}%" for i, p in enumerate(tp)},
    }


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return buf


def plot_sentiment_dist():
    stats  = THESIS_STATS
    labels = ["Negatif", "Positif"]
    values = [stats["total_negatif"], stats["total_positif"]]
    colors = [PALETTE["negatif"], PALETTE["positif"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Distribusi Sentimen Publik terhadap Program MBG\n"
                 f"Total: {stats['total_tweets']:,} tweet (setelah buzzer filtering)",
                 fontsize=12, fontweight="bold")

    wedges, texts, ats = ax1.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2.5},
    )
    for at in ats:
        at.set_fontweight("bold")
    ax1.set_title("Proporsi Sentimen", fontsize=11, pad=8)

    bars = ax2.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5, width=0.45)
    for b, v in zip(bars, values):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                 f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    ax2.set_title("Jumlah Tweet per Kelas Sentimen", fontsize=11, pad=8)
    ax2.set_ylabel("Jumlah Tweet", fontsize=10)
    ax2.set_ylim(0, max(values) * 1.15)
    ax2.grid(axis="y", alpha=0.3)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig


def plot_topic_dist():
    td     = THESIS_STATS["topic_dist"]
    labels = [t.capitalize() for t in td.keys()]
    values = list(td.values())
    colors = [PALETTE[t] for t in td.keys()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Distribusi Topik (LDA) - Program MBG", fontsize=12, fontweight="bold")

    ax1.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140,
            textprops={"fontsize": 12}, wedgeprops={"edgecolor": "white", "linewidth": 2.5})
    ax1.set_title("Proporsi Topik", fontsize=11, pad=8)

    bars = ax2.barh(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, values):
        ax2.text(v + 20, b.get_y() + b.get_height() / 2,
                 f"{v:,}", va="center", fontsize=10, fontweight="bold")
    ax2.set_title("Jumlah Tweet per Topik", fontsize=11, pad=8)
    ax2.set_xlabel("Jumlah Tweet", fontsize=10)
    ax2.set_xlim(0, max(values) * 1.18)
    ax2.grid(axis="x", alpha=0.3)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig


def plot_sent_per_topic():
    spt      = THESIS_STATS["sent_per_topic"]
    topics   = [t.capitalize() for t in spt.keys()]
    neg_vals = [v["negatif"] for v in spt.values()]
    pos_vals = [v["positif"] for v in spt.values()]

    x     = np.arange(len(topics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, neg_vals, width, label="Negatif",
                color=PALETTE["negatif"], edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + width/2, pos_vals, width, label="Positif",
                color=PALETTE["positif"], edgecolor="white", linewidth=1.2)

    for b, v in [(b, v) for bars, vals in [(b1, neg_vals), (b2, pos_vals)]
                         for b, v in zip(bars, vals)]:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 10,
                str(v), ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(topics, fontsize=11)
    ax.set_ylabel("Jumlah Tweet", fontsize=10)
    ax.set_title("Distribusi Sentimen per Topik Program MBG",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig


def plot_metrics():
    es      = THESIS_STATS["eval_sentiment"]
    et      = THESIS_STATS["eval_topic"]
    metrics = list(es.keys())
    x       = np.arange(len(metrics))
    width   = 0.34

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, list(es.values()), width, label="Sentimen",
                color="#E74C3C", edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + width/2, list(et.values()), width, label="Topik",
                color="#3498DB", edgecolor="white", linewidth=1.2)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Perbandingan Metrik Evaluasi IndoBERT\nSentimen vs Topik",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(max(5, len(labels)*1.8), max(4, len(labels)*1.6)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[l.capitalize() for l in labels],
                yticklabels=[l.capitalize() for l in labels],
                ax=ax, linewidths=0.5, annot_kws={"size": 12})
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Aktual", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    return fig


# CUSTOM CSS
st.markdown("""
<style>
.mbg-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2980b9 100%);
    padding: 1.6rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.mbg-header h1 { margin: 0; font-size: 1.65rem; font-weight: 700; }
.mbg-header p  { margin: 0.3rem 0 0; font-size: 0.9rem; opacity: 0.88; }

.metric-card {
    background: #f8f9fa;
    border-left: 4px solid #2980b9;
    padding: 0.9rem 1.1rem;
    border-radius: 8px;
    margin-bottom: 0.6rem;
}
.metric-card .val { font-size: 1.8rem; font-weight: 700; color: #2c3e50; }
.metric-card .lbl { font-size: 0.82rem; color: #7f8c8d; margin-top: 2px; }

.section-title {
    font-size: 1.05rem; font-weight: 700; color: #2c3e50;
    border-bottom: 2px solid #2980b9;
    padding-bottom: 4px; margin: 1rem 0 0.7rem;
}

.info-box {
    background: #eaf4fb; border-left: 4px solid #2980b9;
    padding: 0.7rem 1rem; border-radius: 6px; font-size: 0.88rem;
    margin-bottom: 0.8rem; color: #1a3a5c !important;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("## MBG Analyzer")
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        [" Dashboard", " Prediksi Tweet", " Evaluasi Model", "Info Tentang"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown("### Load Model")
    st.markdown(
        "<div class='info-box'>Model IndoBERT akan diunduh otomatis dari "
        "Hugging Face Hub saat pertama kali digunakan.</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.model_loaded:
        if st.button("Load Model dari Hugging Face", use_container_width=True):
            with st.spinner("Mengunduh dan memuat model IndoBERT... (pertama kali bisa 5-15 menit)"):
                try:
                    tok_sent, model_sent, tok_top, model_top, device = load_models()
                    st.session_state.update({
                        "tok_sent"    : tok_sent,
                        "model_sent"  : model_sent,
                        "tok_top"     : tok_top,
                        "model_top"   : model_top,
                        "device"      : device,
                        "model_loaded": True,
                    })
                    st.success(f"Model berhasil dimuat! (device: {device})")
                except Exception as e:
                    st.error(f"Gagal memuat model:\n{e}")

    if st.session_state.model_loaded:
        st.markdown("[v] **Model siap digunakan**")
    else:
        st.markdown("[!] Model belum dimuat - fitur prediksi nonaktif")

    st.markdown("---")
    st.caption("Tugas Akhir - Universitas Telkom - 2025")

# PAGE HEADER
st.markdown("""
<div class="mbg-header">
  <h1>MBG Analisis Sentimen & Topik - Kebijakan Makanan Bergizi Gratis (MBG)</h1>
  <p>Fine-tuned IndoBERT + Latent Dirichlet Allocation (LDA) . Twitter/X Public Opinion Analysis</p>
</div>
""", unsafe_allow_html=True)

# DASHBOARD
if page == " Dashboard":
    st.markdown("### Ringkasan Dataset & Hasil Penelitian")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{THESIS_STATS['total_tweets']:,}</div>
            <div class="lbl">Total Tweet Valid</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="val" style="color:#E74C3C">{THESIS_STATS['pct_negatif']}%</div>
            <div class="lbl">Sentimen Negatif</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="val" style="color:#2ECC71">{THESIS_STATS['pct_positif']}%</div>
            <div class="lbl">Sentimen Positif</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="val" style="color:#E74C3C">91.01%</div>
            <div class="lbl">Akurasi Sentimen</div></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-card">
            <div class="val" style="color:#3498DB">64.04%</div>
            <div class="lbl">Akurasi Topik</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)
        fig = plot_sentiment_dist()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.markdown('<div class="section-title">Distribusi Topik LDA</div>', unsafe_allow_html=True)
        fig = plot_topic_dist()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")

    st.markdown('<div class="section-title">Distribusi Sentimen per Topik</div>', unsafe_allow_html=True)
    fig = plot_sent_per_topic()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown('<div class="section-title">Tabel Distribusi Sentimen per Topik</div>', unsafe_allow_html=True)
    rows = []
    for topik, v in THESIS_STATS["sent_per_topic"].items():
        rows.append({
            "Topik"    : topik.capitalize(),
            "Total"    : v["total"],
            "Negatif"  : v["negatif"],
            "Positif"  : v["positif"],
            "% Negatif": f"{v['negatif']/v['total']*100:.1f}%",
            "% Positif": f"{v['positif']/v['total']*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown('<div class="section-title">Kata Kunci LDA per Topik</div>', unsafe_allow_html=True)
    col_lda = st.columns(2)
    for idx, (label, keywords) in enumerate(THESIS_STATS["lda_keywords"].items()):
        with col_lda[idx % 2]:
            color = list(PALETTE.values())[idx + 2]
            st.markdown(f"""
            <div style='border-left:4px solid {color};padding:8px 12px;
                        background:#f8f9fa;border-radius:6px;margin-bottom:8px;'>
                <b style='color:{color}'>{label}</b><br>
                <span style='font-size:0.87rem;color:#555'>{keywords}</span>
            </div>""", unsafe_allow_html=True)

# PREDIKSI
elif page == " Prediksi Tweet":
    st.markdown("### Prediksi Sentimen & Topik")

    if not st.session_state.model_loaded:
        st.warning("[!] Model belum dimuat. Klik **Load Model dari Hugging Face** di sidebar.")
        st.stop()

    mode = st.tabs(["Mode 1 . Satu Tweet", "Mode 2 . Daftar Tweet", "Mode 3 . Upload CSV"])

    with mode[0]:
        st.markdown("#### Mode 1 - Prediksi Satu Tweet")
        tweet_input = st.text_area(
            "Masukkan teks tweet:",
            placeholder='Contoh: "Terlalu banyak anggaran yang digunakan untuk MBG yang tidak berguna ini"',
            height=110,
        )
        if st.button("Prediksi", key="btn_mode1"):
            if not tweet_input.strip():
                st.error("Tweet tidak boleh kosong.")
            else:
                with st.spinner("Memprediksi..."):
                    hasil = predict_single(
                        tweet_input.strip(),
                        st.session_state.model_sent, st.session_state.tok_sent,
                        st.session_state.model_top,  st.session_state.tok_top,
                        st.session_state.device,
                    )

                sent_color = "#E74C3C" if hasil["sentimen"] == "negatif" else "#2ECC71"
                top_color  = PALETTE[hasil["topik"]]

                st.markdown("---")
                st.markdown("##### Hasil Prediksi")
                col_r1, col_r2 = st.columns(2)

                with col_r1:
                    st.markdown(f"""
                    <div style='background:#f8f9fa;padding:14px 18px;border-radius:10px;
                                border-left:5px solid {sent_color};'>
                        <div style='font-size:0.82rem;color:#888;margin-bottom:4px'>SENTIMEN</div>
                        <div style='font-size:1.7rem;font-weight:700;color:{sent_color}'>
                            {hasil['sentimen'].upper()}
                        </div>
                        <div style='font-size:0.85rem;color:#555;margin-top:4px'>
                            Confidence: <b>{hasil['sentimen_conf']}</b>
                        </div>
                    </div>""", unsafe_allow_html=True)

                with col_r2:
                    st.markdown(f"""
                    <div style='background:#f8f9fa;padding:14px 18px;border-radius:10px;
                                border-left:5px solid {top_color};'>
                        <div style='font-size:0.82rem;color:#888;margin-bottom:4px'>TOPIK</div>
                        <div style='font-size:1.7rem;font-weight:700;color:{top_color}'>
                            {hasil['topik'].upper()}
                        </div>
                        <div style='font-size:0.85rem;color:#555;margin-top:4px'>
                            Confidence: <b>{hasil['topik_conf']}</b>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")
                col_d1, col_d2 = st.columns(2)

                with col_d1:
                    st.markdown("**Detail probabilitas sentimen:**")
                    for lbl, prob in hasil["detail_sentimen"].items():
                        bar_val = float(prob.replace("%", "")) / 100
                        c       = PALETTE[lbl]
                        st.markdown(f"""
                        <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
                            <span style='width:65px;font-size:0.85rem'>{lbl.capitalize()}</span>
                            <div style='flex:1;background:#eee;border-radius:4px;height:14px;'>
                                <div style='width:{bar_val*100:.1f}%;background:{c};
                                            height:14px;border-radius:4px;'></div>
                            </div>
                            <span style='width:55px;font-size:0.85rem;font-weight:600'>{prob}</span>
                        </div>""", unsafe_allow_html=True)

                with col_d2:
                    st.markdown("**Detail probabilitas topik:**")
                    for lbl, prob in hasil["detail_topik"].items():
                        bar_val = float(prob.replace("%", "")) / 100
                        c       = PALETTE[lbl]
                        st.markdown(f"""
                        <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
                            <span style='width:75px;font-size:0.85rem'>{lbl.capitalize()}</span>
                            <div style='flex:1;background:#eee;border-radius:4px;height:14px;'>
                                <div style='width:{bar_val*100:.1f}%;background:{c};
                                            height:14px;border-radius:4px;'></div>
                            </div>
                            <span style='width:55px;font-size:0.85rem;font-weight:600'>{prob}</span>
                        </div>""", unsafe_allow_html=True)

    with mode[1]:
        st.markdown("#### Mode 2 - Prediksi Daftar Tweet (satu tweet per baris)")
        tweets_raw = st.text_area(
            "Masukkan tweet (satu tweet per baris):",
            height=200,
            placeholder=(
                "Program makan bergizi gratis sangat membantu anak-anak yang kurang mampu\n"
                "Anggaran MBG terlalu besar dan tidak transparan, rawan korupsi\n"
                "Banyak kasus keracunan karena mbg\n"
                "Makanan Beracun Gratis"
            ),
        )

        if st.button("Prediksi Semua", key="btn_mode2"):
            lines = [l.strip() for l in tweets_raw.splitlines() if l.strip()]
            if not lines:
                st.error("Masukkan minimal satu tweet.")
            else:
                results  = []
                progress = st.progress(0, text="Memprediksi tweet...")
                for i, tweet in enumerate(lines):
                    h = predict_single(
                        tweet,
                        st.session_state.model_sent, st.session_state.tok_sent,
                        st.session_state.model_top,  st.session_state.tok_top,
                        st.session_state.device,
                    )
                    results.append({
                        "Tweet"         : tweet[:80] + ("..." if len(tweet) > 80 else ""),
                        "Sentimen"      : h["sentimen"].upper(),
                        "Conf. Sentimen": h["sentimen_conf"],
                        "Topik"         : h["topik"].upper(),
                        "Conf. Topik"   : h["topik_conf"],
                    })
                    progress.progress((i + 1) / len(lines),
                                      text=f"Memprediksi tweet {i+1}/{len(lines)}...")
                progress.empty()

                df_res = pd.DataFrame(results)
                st.dataframe(df_res, use_container_width=True, hide_index=True)

                sent_counts = df_res["Sentimen"].value_counts()
                top_counts  = df_res["Topik"].value_counts()
                st.markdown("---")
                col_s, col_t = st.columns(2)
                with col_s:
                    st.markdown("**Ringkasan Sentimen:**")
                    for lbl, cnt in sent_counts.items():
                        c = PALETTE.get(lbl.lower(), "#888")
                        st.markdown(f"<span style='color:{c};font-weight:600'>- {lbl}</span>: {cnt} tweet",
                                    unsafe_allow_html=True)
                with col_t:
                    st.markdown("**Ringkasan Topik:**")
                    for lbl, cnt in top_counts.items():
                        c = PALETTE.get(lbl.lower(), "#888")
                        st.markdown(f"<span style='color:{c};font-weight:600'>- {lbl}</span>: {cnt} tweet",
                                    unsafe_allow_html=True)

                csv_bytes = df_res.to_csv(index=False).encode("utf-8")
                st.download_button("Download Hasil (.csv)", csv_bytes,
                                   "hasil_prediksi_mode2.csv", "text/csv")

    with mode[2]:
        st.markdown("#### Mode 3 - Prediksi dari File CSV")
        st.markdown("""
        <div class="info-box">
        File CSV harus memiliki kolom bernama <b><code>tweet</code></b>
        yang berisi teks tweet yang ingin diprediksi.
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload file CSV:", type=["csv"])
        if uploaded:
            try:
                df_input = pd.read_csv(uploaded)
                st.success(f"File berhasil dibaca: **{len(df_input):,} baris**, kolom: {list(df_input.columns)}")

                if "tweet" not in df_input.columns:
                    st.error(f"Kolom 'tweet' tidak ditemukan. Kolom yang tersedia: {list(df_input.columns)}")
                else:
                    st.dataframe(df_input.head(3), use_container_width=True, hide_index=True)

                    if st.button("Jalankan Prediksi CSV", key="btn_mode3"):
                        results    = []
                        progress   = st.progress(0, text="Memproses tweet...")
                        total_rows = len(df_input)

                        for i, row in df_input.iterrows():
                            h = predict_single(
                                str(row["tweet"]),
                                st.session_state.model_sent, st.session_state.tok_sent,
                                st.session_state.model_top,  st.session_state.tok_top,
                                st.session_state.device,
                            )
                            results.append({
                                "tweet"        : row["tweet"],
                                "sentimen"     : h["sentimen"],
                                "sentimen_conf": h["sentimen_conf"],
                                "topik"        : h["topik"],
                                "topik_conf"   : h["topik_conf"],
                            })
                            progress.progress((i + 1) / total_rows,
                                              text=f"Tweet {i+1}/{total_rows}...")
                        progress.empty()

                        df_out = pd.DataFrame(results)
                        st.success(f"[v] Prediksi selesai untuk {len(df_out):,} tweet!")
                        st.dataframe(df_out, use_container_width=True, hide_index=True)

                        st.markdown("---")
                        st.markdown("##### Ringkasan Distribusi Hasil Prediksi")
                        col_rs, col_rt = st.columns(2)
                        with col_rs:
                            sc = df_out["sentimen"].value_counts()
                            fig_s, ax_s = plt.subplots(figsize=(5, 3.5))
                            ax_s.bar(sc.index, sc.values,
                                     color=[PALETTE.get(k, "#888") for k in sc.index],
                                     edgecolor="white", linewidth=1.2)
                            for idx_b, (lbl, val) in enumerate(sc.items()):
                                ax_s.text(idx_b, val + 0.3, str(val), ha="center",
                                          fontweight="bold", fontsize=10)
                            ax_s.set_title("Distribusi Sentimen", fontsize=11)
                            ax_s.set_ylabel("Jumlah Tweet")
                            ax_s.grid(axis="y", alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig_s, use_container_width=True)
                            plt.close(fig_s)
                        with col_rt:
                            tc = df_out["topik"].value_counts()
                            fig_t, ax_t = plt.subplots(figsize=(5, 3.5))
                            ax_t.barh(tc.index, tc.values,
                                      color=[PALETTE.get(k, "#888") for k in tc.index],
                                      edgecolor="white", linewidth=1.2)
                            for idx_b, (lbl, val) in enumerate(tc.items()):
                                ax_t.text(val + 0.1, idx_b, str(val), va="center",
                                          fontweight="bold", fontsize=10)
                            ax_t.set_title("Distribusi Topik", fontsize=11)
                            ax_t.set_xlabel("Jumlah Tweet")
                            ax_t.grid(axis="x", alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig_t, use_container_width=True)
                            plt.close(fig_t)

                        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                        st.download_button("Download Hasil (.csv)", csv_bytes,
                                           "hasil_prediksi_csv.csv", "text/csv")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

# EVALUASI
elif page == " Evaluasi Model":
    st.markdown("### Hasil Evaluasi Model IndoBERT")

    st.markdown('<div class="section-title">Ringkasan Metrik Evaluasi</div>', unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**Model Sentimen (Negatif / Positif)**")
        df_ms = pd.DataFrame([THESIS_STATS["eval_sentiment"]])
        df_ms.index = ["IndoBERT Sentimen"]
        st.dataframe(df_ms.style.format("{:.4f}"), use_container_width=True)

    with col_m2:
        st.markdown("**Model Topik (Program / Anggaran / Gizi / Distribusi)**")
        df_mt = pd.DataFrame([THESIS_STATS["eval_topic"]])
        df_mt.index = ["IndoBERT Topik"]
        st.dataframe(df_mt.style.format("{:.4f}"), use_container_width=True)

    st.markdown('<div class="section-title">Metrik per Kelas - Model Topik</div>', unsafe_allow_html=True)
    df_cls = pd.DataFrame(THESIS_STATS["eval_topic_class"]).T
    st.dataframe(df_cls.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}",
                                      "F1-Score": "{:.4f}", "Support": "{:.0f}"}),
                 use_container_width=True)

    st.markdown("---")

    st.markdown('<div class="section-title">Visualisasi Metrik Evaluasi</div>', unsafe_allow_html=True)
    fig = plot_metrics()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    col_cm1, col_cm2 = st.columns([1, 1.5])

    with col_cm1:
        st.markdown("**Model Sentimen**")
        fig_cm = plot_confusion_matrix(
            THESIS_STATS["cm_sent"],
            ["negatif", "positif"],
            "Confusion Matrix Sentimen",
        )
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    with col_cm2:
        st.markdown("**Model Topik**")
        fig_cm2 = plot_confusion_matrix(
            THESIS_STATS["cm_topic"],
            ["program", "anggaran", "gizi", "distribusi"],
            "Confusion Matrix Topik",
        )
        st.pyplot(fig_cm2, use_container_width=True)
        plt.close(fig_cm2)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <b>Catatan Interpretasi:</b><br>
    - Model sentimen mencapai akurasi <b>91,01%</b> dengan F1-Score <b>0.9076</b> - sangat baik.<br>
    - Model topik mencapai akurasi <b>64,04%</b> dengan F1-Score <b>0.6366</b>.<br>
    - Kelas <b>Distribusi</b> mendapat F1 terendah (0.4553) karena data paling sedikit (139 sampel test).<br>
    - Class weighting diterapkan untuk mengatasi imbalance negatif:positif = 5.8:1.
    </div>""", unsafe_allow_html=True)

# TENTANG
elif page == "Info Tentang":
    st.markdown("### Info Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini merupakan implementasi hasil **Tugas Akhir** yang mengembangkan model
    klasifikasi sentimen dan topik untuk menganalisis opini publik terhadap kebijakan
    **Makanan Bergizi Gratis (MBG)** Pemerintah Indonesia.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Metodologi:**
        - Pengumpulan data via Twitter scraping (`twikit`)
        - Filtering buzzer & akun tidak autentik
        - Preprocessing: stopwords removal tanpa stemming (Skenario A)
        - Topic modeling: **LDA** (4 topik: Program, Anggaran, Gizi, Distribusi)
        - Sentiment & topic classification: **IndoBERT** fine-tuned
        - Evaluasi: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
        """)
    with col_b:
        st.markdown("""
        **Konfigurasi Model:**
        - Base model: `indobenchmark/indobert-base-p1`
        - Max token length: 128
        - Batch size: 16
        - Epochs: 5 (sentimen & topik)
        - Learning rate: 2e-5
        - Class weighting untuk mengatasi imbalance
        - Optimizer: AdamW + warmup scheduler
        """)

    st.markdown("---")
    st.markdown("""
    **Label Topik:**

    | Topik | Kata Kunci Representatif |
    |-------|--------------------------|
    | Program | pendidikan, kualitas, guru, negara, keracunan, kebijakan, dana |
    | Anggaran | rakyat, anggaran, korupsi, negara, efisiensi, APBN, pejabat, pajak |
    | Gizi | anak, gizi, keracunan, kualitas, pelaksanaan, siswa, sekolah, sehat |
    | Distribusi | catering, sekolah, dapur, kualitas, korupsi, keracunan, porsi, vendor |
    """)
    st.caption("Universitas Telkom - Tugas Akhir 2025")
