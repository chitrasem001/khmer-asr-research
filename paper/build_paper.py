"""Build the paper draft (.docx) from the experiment outputs.

Every number in the paper is read from data/manifests/*.json and experiments/*/metrics.json, so
rerunning this script after new experiments keeps the text and tables consistent with the results.

    python paper/build_paper.py   ->   paper/khmer_asr_paper_draft.docx
"""
import json
from datetime import date
from pathlib import Path

import jiwer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from khmer_asr.preprocessing.text import normalize_khmer_text, remove_spaces

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
MANIFESTS = ROOT / "data" / "manifests"
M0_DIR = ROOT / "experiments" / "m0_baseline"

AUTHORS = "Sem Chitra, Dr. Mao Makara"
AFFILIATION = "Data Science in Computer Science, Royal University of Phnom Penh, Cambodia"

# Validated categorical slots (dataviz reference palette, light surface).
BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#eb6834", "#9a9a94", "#1a9e77"

MODEL_LABELS = {
    "whisper-small": ("Whisper small", "Encoder-decoder", "Apache-2.0"),
    "whisper-medium": ("Whisper medium", "Encoder-decoder", "Apache-2.0"),
    "whisper-large-v3-turbo": ("Whisper large-v3-turbo", "Encoder-decoder", "MIT"),
    "mms-1b-all": ("MMS-1B-all (khm adapter)", "CTC", "CC BY-NC 4.0"),
}

REFERENCES = [
    ("A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech "
    "recognition via large-scale weak supervision,” arXiv:2212.04356, 2022."),
    ("A. Conneau, M. Ma, S. Khanuja, et al., “FLEURS: Few-shot learning evaluation of universal "
    "representations of speech,” arXiv:2205.12446, 2022."),
    ("V. Pratap, A. Tjandra, B. Shi, et al., “Scaling speech technology to 1,000+ languages,” "
    "arXiv:2305.13516, 2023."),
    ("S. Seng, S. Sam, L. Besacier, B. Bigi, and E. Castelli, “First broadcast news transcription "
    "system for Khmer language,” in Proc. LREC, 2008."),
    ("S. Singh, A. Yadavalli, C. Arnett, and A. Warstadt, “BuzzASR: A swarm of 100+ monolingual "
    "speech recognition models,” in Findings of EMNLP, 2026. arXiv:2609.09554."),
    ("J. Liao, W. Zhang, Z. Liu, et al., “SEA-SpeechBench: A large-scale multitask benchmark for "
    "speech understanding across Southeast Asia,” in Proc. EMNLP, 2026. arXiv:2609.09672."),
    ("G. Das, D. Laskar, H. A. Ahmed, et al., “Robust Assamese speech recognition through controlled "
    "fine-tuning of Whisper models,” arXiv:2607.17164, 2026."),
    ("A. Rouditchenko, S. Khurana, S. Thomas, et al., “Comparison of multilingual self-supervised "
    "and weakly-supervised speech pre-training for adaptation to unseen languages,” in Proc. "
    "Interspeech, 2023."),
    ("H. Lovenia et al., “SEACrowd: A multilingual multimodal data hub and benchmark suite for "
    "Southeast Asian languages,” arXiv:2406.10118, 2024."),
    ("Digital Divide Data, “Khmer ASR Cultural Dataset,” CC BY-SA 4.0, Hugging Face: "
    "Digital-Divide-Data/khmer-speech-dataset, revision e872b0c."),
    "sengtha, “whisper-base-khmer,” model card, Hugging Face, accessed Oct. 2026.",
    "Darayut, “Omnilingual-ASR-Khm,” model card, Hugging Face, accessed Oct. 2026.",
    ("M. Bisani and H. Ney, “Bootstrap estimates for confidence intervals in ASR performance "
     "evaluation,” in Proc. ICASSP, 2004, vol. 1, pp. 409–412."),
]


# ---------------------------------------------------------------- data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clip_errors(pred: pd.DataFrame) -> pd.DataFrame:
    """Per-clip character edit counts on normalized, space-free text."""
    rows = []
    for ref, hyp in zip(pred["reference"], pred["hypothesis"].fillna(""), strict=True):
        r = remove_spaces(normalize_khmer_text(ref))
        h = remove_spaces(normalize_khmer_text(hyp))
        o = jiwer.process_characters([r], [h])
        rows.append((o.substitutions + o.deletions + o.insertions, o.hits + o.substitutions + o.deletions))
    return pd.DataFrame(rows, columns=["errors", "ref_chars"])


def bootstrap_ci(errs: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% bootstrap interval for corpus CER, resampling clips."""
    rng = np.random.default_rng(seed)
    e, n = errs["errors"].to_numpy(), errs["ref_chars"].to_numpy()
    idx = rng.integers(0, len(e), size=(n_boot, len(e)))
    cers = e[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(np.percentile(cers, 2.5)), float(np.percentile(cers, 97.5))


def paired_diff_ci(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000,
                   seed: int = 42) -> tuple[float, float, float]:
    """CER(a) - CER(b) on the same clips, with a 95% paired bootstrap interval."""
    if len(a) != len(b):
        raise ValueError("Paired comparison needs the same clips")
    rng = np.random.default_rng(seed)
    ea, eb, n = a["errors"].to_numpy(), b["errors"].to_numpy(), a["ref_chars"].to_numpy()
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    diffs = (ea[idx].sum(axis=1) - eb[idx].sum(axis=1)) / n[idx].sum(axis=1)
    return (float((ea.sum() - eb.sum()) / n.sum()),
            float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def load_results() -> list[dict]:
    results = []
    for slug, (label, family, license_) in MODEL_LABELS.items():
        d = M0_DIR / slug
        if not (d / "metrics.json").exists():
            continue
        m = load_json(d / "metrics.json")
        info = load_json(d / "run_info.json")
        pred = pd.read_csv(d / "predictions.csv")
        lo, hi = bootstrap_ci(clip_errors(pred))
        results.append(
            {"slug": slug, "label": label, "family": family, "license": license_,
             "metrics": m, "info": info, "pred": pred, "ci": (lo, hi)}
        )
    return results


# Fine-tuned runs: (family, label, training hours, run folder). Test/FLEURS/seen-speaker results
# live in experiments/m1_eval[_fleurs|_seen]/<run>, training records in experiments/<run>.
FINETUNED = [
    ("whisper", "Whisper small", 10, "m1_whisper_small"),
    ("whisper", "Whisper small", 50, "s_whisper_small_50h"),
    ("whisper", "Whisper small", 100, "s_whisper_small_100h"),
    ("mms", "MMS-1B", 10, "m1_mms"),
    ("mms", "MMS-1B", 50, "s_mms_50h"),
    ("mms", "MMS-1B", 100, "s_mms_100h"),
]
# M2: MMS adapter started from the released khm adapter; same subsets and steps as the S runs.
M2_RUNS = [
    ("mms_khm", "MMS-1B, khm init (M2)", 10, "m2_mms_khm_init_10h"),
    ("mms_khm", "MMS-1B, khm init (M2)", 100, "m2_mms_khm_init_100h"),
]
# Second training seed (same data subset) for the 10 h MMS runs: seed-42 run -> seed-43 run.
SEED_PAIRS = {"m1_mms": "m1_mms_seed43", "m2_mms_khm_init_10h": "m2_mms_khm_init_10h_seed43"}
FULL_RUN = ("whisper_full", "Whisper small, full data", 872, "f_whisper_small_872h")
EXPERIMENTS = ROOT / "experiments"
EVAL_DIRS = {"test": "m1_eval", "fleurs": "m1_eval_fleurs", "seen": "m1_eval_seen"}
ZERO_SHOT = "mms-1b-all"


def eval_dir(run: str, split: str) -> Path:
    """Result folder of a run on one evaluation set; zero-shot MMS lives in the M0 folders."""
    if run == ZERO_SHOT and split != "seen":
        return M0_DIR / run if split == "test" else EXPERIMENTS / "m0_fleurs" / run
    return EXPERIMENTS / EVAL_DIRS[split] / run


def metric(run: str, split: str) -> dict | None:
    path = eval_dir(run, split) / "metrics.json"
    return load_json(path) if path.exists() else None


def paired(run_a: str, run_b: str, split: str) -> tuple[float, float, float]:
    """CER(a) - CER(b) on the same clips of one evaluation set, with a paired 95% interval."""
    a = pd.read_csv(eval_dir(run_a, split) / "predictions.csv").set_index("utt_id").sort_index()
    b = pd.read_csv(eval_dir(run_b, split) / "predictions.csv").set_index("utt_id").loc[a.index]
    return paired_diff_ci(clip_errors(a.reset_index()), clip_errors(b.reset_index()))


def load_finetuned(runs: list[tuple] = FINETUNED) -> list[dict]:
    """Test, FLEURS and seen-speaker results plus training records for every run that exists."""
    rows = []
    for family, label, hours, run in runs:
        if metric(run, "test") is None:
            continue
        rows.append({
            "family": family, "label": label, "hours": hours, "run": run,
            "test": metric(run, "test"), "fleurs": metric(run, "fleurs"), "seen": metric(run, "seen"),
            "test_ci": bootstrap_ci(clip_errors(pd.read_csv(eval_dir(run, "test") / "predictions.csv"))),
            "train": load_json(EXPERIMENTS / run / "run_info.json"),
        })
    return rows


# ---------------------------------------------------------------- figure
def scaling_figure(rows: list[dict], zero_shot: dict, path: Path) -> None:
    """CER against training hours on the test set and FLEURS: the scaling runs as lines, M2 and the
    one-pass full-data Whisper run as separate markers."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), dpi=200, sharey=False)
    series = (  # family, label, colour, marker, line
        ("whisper", "Whisper small", ORANGE, "o", "-"),
        ("whisper_full", "Whisper small, 872 h (1 pass)", ORANGE, "D", ""),
        ("mms", "MMS-1B, new adapter", BLUE, "o", "-"),
        ("mms_khm", "MMS-1B, khm init (M2)", GREEN, "s", "--"),
    )
    for ax, key, title in ((axes[0], "test", "Test set (unseen speakers + sentences)"),
                           (axes[1], "fleurs", "FLEURS (out of domain)")):
        ends = []
        for family, label, color, marker, line in series:
            pts = [(r["hours"], 100 * r[key]["cer"]) for r in rows if r["family"] == family and r[key]]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=2 if line == "-" else 1.4, linestyle=line or "none",
                    marker=marker, markersize=5, markerfacecolor="white" if marker == "D" else color,
                    label=label)
            ends.append((xs[-1], ys[-1]))
        # Label each series' last point; points sharing an x are spread apart in CER order.
        for x in {e[0] for e in ends}:
            here = sorted(y for ex, y in ends if ex == x)
            for i, y in enumerate(here):
                dy = 0 if len(here) == 1 else (i - (len(here) - 1) / 2) * 9
                ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(6, dy),
                            va="center", fontsize=6.5, color="#3d3d3a")
        if zero_shot.get(key) is not None:
            zs = 100 * zero_shot[key]
            ax.axhline(zs, color=BLUE, linewidth=1, linestyle=":")
            # Put the label on the side of the line away from the fine-tuned MMS points.
            mms_above = any(100 * r[key]["cer"] > zs for r in rows if r["family"] == "mms" and r[key])
            ax.annotate("MMS zero-shot", (10, zs), textcoords="offset points",
                        xytext=(0, -8 if mms_above else 3), fontsize=6, color=BLUE)
        ax.set_xscale("log")
        ax.set_xticks([10, 50, 100, 872], ["10 h", "50 h", "100 h", "872 h"])
        ax.minorticks_off()
        ax.set_xlim(8, 1400)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=7.5)
        ax.set_ylabel("CER (%)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="y", color="#e4e3dd", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def speaker_figure(audit: dict, split: dict, path: Path) -> None:
    speakers = pd.DataFrame(audit["speakers"]).sort_values("hours")
    test = set(split["config"]["split"]["test_speakers"])
    train = set(split["splits"]["train"]["hours_by_speaker"])
    role = speakers["speaker_id"].map(
        lambda s: "Test" if s in test else ("Train" if s in train else "Not used")
    )
    colors = role.map({"Train": BLUE, "Test": ORANGE, "Not used": GREY})

    fig, ax = plt.subplots(figsize=(6.3, 4.4), dpi=200)
    y = np.arange(len(speakers))
    ax.barh(y, speakers["hours"], color=colors, height=0.72, edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y, speakers["speaker_id"], fontsize=7)
    ax.set_xlabel("Hours of audio (log scale)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    for yi, h in zip(y, speakers["hours"], strict=True):
        ax.text(h * 1.08, yi, f"{h:g} h", va="center", fontsize=6.5, color="#3d3d3a")
    ax.grid(axis="x", color="#e4e3dd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GREY)]
    ax.legend(handles, ["Train speaker", "Test speaker", "Not used"], fontsize=7, frameon=False,
              loc="lower right")
    ax.set_xlim(right=speakers["hours"].max() * 4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- docx helpers
def set_fonts(style, latin: str = "Times New Roman", khmer: str = "Khmer UI") -> None:
    """Latin text in `latin`; Khmer (a complex script) in `khmer`. Drops theme-font overrides."""
    style.font.name = latin
    rfonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rfonts.attrib.pop(qn(attr), None)
    rfonts.set(qn("w:eastAsia"), latin)
    rfonts.set(qn("w:cs"), khmer)


def set_cell_shading(cell, hex_fill: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(shd)


def add_table(doc: Document, caption: str, header: list[str], rows: list[list[str]],
              widths_cm: list[float]) -> None:
    cap = doc.add_paragraph(style="Caption")
    cap.add_run(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_with_next = True
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(header):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        set_cell_shading(cell, "E8EEF7")
    for row in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = text
    last = len(table.rows) - 1
    for n, row in enumerate(table.rows):
        # Keep the whole table (and its caption) on one page.
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for i, w in enumerate(widths_cm):
            row.cells[i].width = Cm(w)
            for p in row.cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.keep_with_next = n < last
                for r in p.runs:
                    r.font.size = Pt(8.5)
    doc.add_paragraph()


def para(doc: Document, text: str, bold_lead: str | None = None) -> None:
    p = doc.add_paragraph()
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def h(x: float) -> str:
    return f"{x:,.1f} h"


# ---------------------------------------------------------------- document
def build() -> Path:
    audit = load_json(MANIFESTS / "audit_report.json")
    split = load_json(MANIFESTS / "split_report.json")
    s = audit["summary"]
    sp = split["splits"]
    issues = audit["text_issues_in_sentences"]
    results = load_results()
    ft = load_finetuned()
    m2 = load_finetuned(M2_RUNS)
    full = load_finetuned([FULL_RUN])
    zero_shot = {"test": load_json(M0_DIR / "mms-1b-all" / "metrics.json")["cer"],
                 "fleurs": load_json(EXPERIMENTS / "m0_fleurs" / "mms-1b-all" / "metrics.json")["cer"]}
    diag = load_json(M0_DIR / "whisper_diagnostics.json")
    mms = next((r for r in results if r["slug"] == "mms-1b-all"), None)
    whisper = [r for r in results if r["family"] == "Encoder-decoder"]

    odd = {b: v for b, v in audit["sample_rates_by_batch"].items() if set(v["sample_rates"]) != {"16000"}}
    if len(odd) != 1:
        raise ValueError(f"Text assumes one non-16 kHz batch, found {sorted(odd)}")
    odd_batch, odd_info = next(iter(odd.items()))
    odd_rate = next(iter(odd_info["sample_rates"]))
    odd_batch_text = f"{odd_info['files']} files named *-{odd_batch}"
    rate_text = f"16 kHz mono, except {odd_info['files']} files at {int(odd_rate) // 1000} kHz"

    speakers = pd.DataFrame(audit["speakers"])
    used =set(sp["train"]["hours_by_speaker"]) | set(split["config"]["split"]["test_speakers"])
    excluded = speakers[~speakers["speaker_id"].isin(used)]
    if len(excluded) != 1:
        raise ValueError(f"Text assumes one excluded speaker, found {len(excluded)}")
    excluded_label = f"speaker {excluded['speaker_id'].iloc[0]}"
    excluded_hours = float(excluded["hours"].iloc[0])
    # Speaker IDs encode the age group; adt4 is the 60+ group in speaker_metadata.xlsx.
    n_over_60 = int((speakers["group_code"] == "adt4").sum())

    fig_path = OUT / "fig_speakers.png"
    speaker_figure(audit, split, fig_path)
    scaling_path = OUT / "fig_scaling.png"
    scaling_figure(ft + m2 + full, zero_shot, scaling_path)

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(2.2))
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    normal = doc.styles["Normal"]
    set_fonts(normal)
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(5)
    for name, size in (("Heading 1", 12), ("Heading 2", 11), ("Caption", 9), ("Title", 15)):
        st = doc.styles[name]
        set_fonts(st)
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)

    draft = doc.add_paragraph()
    draft.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = draft.add_run(f"DRAFT FOR ADVISOR REVIEW — {date.today():%d %b %Y}")
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(0xB0, 0x3A, 0x2E)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = title.add_run("Fine-Tuning Pretrained Models for Khmer Speech Recognition: "
                       "Leakage-Free Evaluation, Data Scaling and Adapter Initialisation")
    tr.bold = True
    tr.font.size = Pt(15)
    for text, size in ((AUTHORS, 11), (AFFILIATION, 10)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(text).font.size = Pt(size)

    # ---- Abstract
    doc.add_heading("Abstract", level=1)
    if mms and whisper:
        w_lo = min(r["metrics"]["cer"] for r in whisper)
        result_sentence = (
            f"Zero-shot, all three Whisper models fail on Khmer (CER ≥ {pct(w_lo, 0)}), "
            f"looping on a repeated character or misidentifying the language, whereas the MMS-1B "
            f"CTC model with its Khmer adapter reaches {pct(mms['metrics']['cer'])} CER on the "
            "held-out test set."
        )
    else:
        result_sentence = "Zero-shot baseline results are being computed."

    def ftr(family: str, hours: int) -> dict | None:
        return next((r for r in ft if r["family"] == family and r["hours"] == hours), None)

    w10, w100, m100 = ftr("whisper", 10), ftr("whisper", 100), ftr("mms", 100)
    if w10 and w100 and m100:
        scaling_sentence = (
            "Fine-tuning on nested 10, 50 and 100 h subsets lowers test CER for both families; "
            f"Whisper small gains most, from {pct(w10['test']['cer'])} at 10 h to "
            f"{pct(w100['test']['cer'])} at 100 h, overtaking MMS ({pct(m100['test']['cer'])}). "
            "On the out-of-domain FLEURS benchmark the ranking reverses "
            f"(MMS {pct(m100['fleurs']['cer'])}, Whisper {pct(w100['fleurs']['cer'])}), showing that "
            "a same-corpus test set overstates real-world quality."
        )
    else:
        scaling_sentence = "Fine-tuning experiments on data quantity are in progress."
    m2_100 = next((r for r in m2 if r["hours"] == 100), None)
    if m2_100 and full and full[0]["seen"]:
        f0 = full[0]
        seen_gap = 1 - f0["seen"]["cer"] / f0["test"]["cer"]
        m2_sentence = (
            "Starting the MMS adapter from Meta's released Khmer adapter instead of from scratch "
            "lowers CER at both 10 and 100 h and gives the most robust model "
            f"({pct(m2_100['test']['cer'], 2)} test, {pct(m2_100['fleurs']['cer'], 2)} FLEURS, below "
            f"zero-shot MMS). Whisper small trained on all {f0['hours']} h reaches "
            f"{pct(f0['test']['cer'], 2)} on the test set but {pct(f0['fleurs']['cer'], 2)} on FLEURS, "
            f"and is {pct(seen_gap, 0)} more accurate on speakers seen in training than on new ones, "
            "so held-out speakers are essential for a fair estimate."
        )
    else:
        m2_sentence = ""
    para(doc, (
        "Khmer remains under-resourced for automatic speech recognition (ASR), and published Khmer "
        "results are hard to compare because they use different test sets, text normalization and "
        f"training data. We audit the full Digital Divide Data Khmer corpus ({s['rows']:,} utterances, "
        f"{h(s['total_hours'])}, {s['speakers']} speakers) and find that every sentence is read by "
        "four different speakers, so a random split leaks test sentences into training. We define a "
        "split that holds out both speakers and every sentence they read, a documented Khmer text "
        "normalization protocol, and a space-insensitive character error rate (CER). "
        f"{result_sentence} {scaling_sentence} {m2_sentence}"
    ))

    # ---- 1 Introduction
    doc.add_heading("1. Introduction", level=1)
    para(doc, (
        "Large multilingual speech models such as Whisper [1] and MMS [3] cover Khmer, but their "
        "out-of-the-box accuracy on Khmer is poor. A 2026 study reports 103.19% CER for zero-shot "
        "Whisper-large-v3 on FLEURS Khmer and 25.99% after full fine-tuning [5]; on SEA-SpeechBench, "
        "even commercial speech models reach 13–30% CER [6]. Khmer is written without spaces "
        "between words, so word error rate depends on a segmentation choice, and CER is the more "
        "reliable metric."
    ))
    para(doc, (
        "A large Khmer corpus is now openly available [10], and community models have already been "
        "fine-tuned on it [11]. What is missing is a controlled, reproducible study: reported numbers "
        "use different test sets and normalization, and none that we found guards against the "
        "sentence leakage this corpus makes easy. This paper contributes:"
    ))
    bullets(doc, [
        ("A full-corpus audit of the Digital Divide Data Khmer dataset that corrects several figures "
        "on its dataset card (Section 3)."),
        ("A split protocol that holds out both test speakers and every sentence they read, with "
        "automatic leakage checks (Section 4.1)."),
        ("A Khmer text normalization protocol derived from the character inventory of all "
        f"{s['duplicates']['unique_sentences']:,} sentences (Section 4.2)."),
        ("Zero-shot baselines for two model families, Whisper (encoder-decoder) and MMS (CTC), "
        "including a diagnosis of Whisper's failure mode (Section 5)."),
        ("A controlled data-scaling study (10, 50, 100 h) for both families, evaluated in and out "
         "of domain (Section 6)."),
        ("A Khmer-specific strategy, initialising the MMS adapter from the released Khmer adapter, "
         "tested at two data sizes and two seeds with paired significance tests (Section 7)."),
        ("A full-data model and a seen- versus unseen-speaker comparison showing how much a random "
         "split would overstate accuracy (Section 8)."),
        ("An error analysis showing that the most frequent errors are spelling variants the corpus "
         "itself mixes, and that digits, Latin words and long transcripts are the weakest points "
         "(Section 9)."),
    ])

    # ---- 2 Related work
    doc.add_heading("2. Related Work", level=1)
    para(doc, (
        "Early Khmer ASR built HMM-based broadcast-news recognizers and compared word and sub-word "
        "units because of the missing word boundaries [4]. Current work starts from multilingual "
        "pretrained models. Whisper is trained with weak supervision on about 680,000 hours of audio "
        "[1]; MMS covers 1,107 languages with language-specific adapters [3]; FLEURS provides about "
        "12 hours per language for evaluation [2]. How many hours of a language a model saw in "
        "pre-training predicts whether self-supervised or weakly-supervised models adapt better [8]."
    ))
    para(doc, (
        "For Khmer specifically, BuzzASR fine-tunes Whisper-large-v3 per language [5], and community "
        "model cards report 22.8% FLEURS CER for a Whisper-base model trained partly on the corpus we "
        "use [11] and 12.03% for an Omnilingual CTC model [12]. These numbers come from different "
        "test sets and normalization and are not directly comparable. A controlled Whisper "
        "fine-tuning study for Assamese [7] is the closest methodological template. We found no "
        "controlled study of modern pretrained models fine-tuned on Khmer."
    ))

    # ---- 3 Dataset
    doc.add_heading("3. Dataset and Audit", level=1)
    para(doc, (
        f"We use the Khmer ASR Cultural Dataset [10] (CC BY-SA 4.0) at a pinned revision. Because its "
        "dataset card is internally inconsistent, we audited all "
        f"{audit['files_total']:,} parquet files by reading only their metadata columns, without "
        "downloading the audio. Table 1 compares the card with the audit."
    ))
    add_table(doc, "Table 1. Dataset card versus our full-corpus audit.",
              ["Item", "Dataset card", "Our audit"],
              [["Utterances", "450,396", f"{s['rows']:,}"],
               ["Total audio", "727.94 h", h(s["total_hours"])],
               ["Speakers with audio", "12", f"{s['speakers']} (9 female, 11 male, aged 18–66)"],
               ["Unique sentences", "not stated", f"{s['duplicates']['unique_sentences']:,}"],
               ["Topics / subtopics", "61 / 1,749", f"{s['topics']} / {s['subtopics']}"],
               ["Sample rate", "16 kHz (feature metadata)", rate_text]],
              [4.2, 3.6, 7.6])
    reads = s["duplicates"]["speakers_per_sentence"]
    para(doc, (
        f"Clips have a median length of {s['duration_s']['median']:.2f} s (5th–95th percentile "
        f"{s['duration_s']['p05']:.1f}–{s['duration_s']['p95']:.1f} s). The key finding is that "
        f"{reads.get('4', 0):,} of {s['duplicates']['unique_sentences']:,} sentences are read by "
        "exactly four different speakers, in groups of two female and two male readers. Speaker "
        "hours are also highly unbalanced (Figure 1): the three largest speakers hold 61% of the "
        "audio, while three speakers have only 16 clips each. "
        f"Finally, one upload batch ({odd_batch_text}) is stored at {odd_rate} Hz although the "
        "dataset declares 16 kHz. Pipelines that decode the raw audio bytes without resampling "
        "feed these clips to a model at the wrong speed; we found this through near-total "
        "recognition failure on exactly these clips and resample all audio to 16 kHz."
    ))
    doc.add_picture(str(fig_path), width=Cm(14.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(style="Caption")
    cap.add_run("Figure 1. Hours of audio per speaker (log scale) and each speaker's role in the "
                f"split. {excluded_label} is not used because every sentence this speaker read was "
                "also read by a test speaker.")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---- 4 Method
    doc.add_heading("4. Methodology", level=1)
    doc.add_heading("4.1 Leakage-free split", level=2)
    para(doc, (
        "Because each sentence is read four times, holding out speakers alone would still put test "
        "sentences into training. Our rule: test speakers never appear in training, and no sentence "
        "read by a test speaker appears in training. Training speakers' readings of those sentences "
        "form a second evaluation set of unseen sentences from known speakers. Validation holds out "
        f"whole sentences for checkpoint selection. We compared four choices of test speakers and "
        f"chose option {split['split_option']} (two female and two male speakers aged 23–60), "
        f"which keeps {h(sp['train']['hours'])} for training. We drop "
        f"{split['dropped']['total_dropped']} clips with a missing duration, under "
        f"{split['config']['data']['min_duration_s']} s, or over "
        f"{split['config']['data']['max_duration_s']:.0f} s. Leakage checks run on every build. "
        f"As a side effect, {excluded_label} ({excluded_hours:.2f} h) drops out of training, because "
        "every sentence this speaker read was shared with a test speaker."
    ))
    add_table(doc, "Table 2. Splits (option D, seed 42).",
              ["Split", "Clips", "Hours", "Speakers", "Purpose"],
              [["Train", f"{sp['train']['utterances']:,}", f"{sp['train']['hours']:.1f}",
                str(sp["train"]["speakers"]), "Fine-tuning"],
               ["Validation", f"{sp['validation']['utterances']:,}",
                f"{sp['validation']['hours']:.1f}", str(sp["validation"]["speakers"]),
                "Checkpoint selection"],
               ["Test", f"{sp['test']['utterances']:,}", f"{sp['test']['hours']:.1f}",
                str(sp["test"]["speakers"]), "Unseen speakers and sentences"],
               ["Test (seen speakers)", f"{sp['test_seen_speakers']['utterances']:,}",
                f"{sp['test_seen_speakers']['hours']:.1f}", str(sp["test_seen_speakers"]["speakers"]),
                "Unseen sentences, known speakers"]],
              [3.4, 2.0, 1.7, 1.9, 6.4])

    doc.add_heading("4.2 Text normalization", level=2)
    para(doc, (
        f"We derived the rules from the inventory of all {audit['distinct_chars']} distinct characters "
        "in the transcripts (Table 3). The same function normalizes training targets and both sides "
        "of every score. Primary CER also removes all spaces, because Khmer word spacing is optional "
        "and inconsistent; we report a space-sensitive CER as a secondary metric."
    ))
    add_table(doc, "Table 3. Normalization rules and the sentences they affect.",
              ["Found in transcripts", "Sentences", "Rule"],
              [["Zero-width characters", f"{issues['zero_width_chars']:,}", "Remove"],
               ["Punctuation (\u17d4 \u17d6 \" « » ( ) , -)", "—",
                "Replace with a space (not spoken)"],
               ["% & # @, symbols, repetition sign \u17d7", "—", "Keep (read aloud)"],
               ["Arabic / Khmer digits", f"{issues['arabic_digits']:,} / {issues['khmer_digits']:,}",
                "Map Arabic to Khmer digits"],
               ["Latin words", f"{issues['latin_letters']:,}", "Keep, lowercased"],
               ["Legacy order: subscript ro first / vowel before subscript / nikahit before u",
                (f"{issues['coeng_ro_before_other_subscript']:,} / {issues['vowel_before_subscript']:,}"
                f" / {issues['nikahit_before_u']:,}"),
                "Reorder to the standard order"],
               ["Newlines / double spaces", f"{issues['newlines']:,} / {issues['double_spaces']:,}",
                "Collapse whitespace"]],
              [6.0, 2.6, 6.8])

    doc.add_heading("4.3 Models and setup", level=2)
    gpu = results[0]["info"]["gpu"] if results else "an RTX 4060 Laptop GPU"
    para(doc, (
        "We evaluate two families without fine-tuning (M0): Whisper small, medium and large-v3-turbo, "
        "forced to transcribe Khmer, and MMS-1B-all with its Khmer adapter. All model revisions are "
        f"pinned. Inference uses greedy decoding in fp16 on {gpu} (8 GB). Whisper's zero-shot "
        "decoding repeats until the 440-token limit, which would take about 16 GPU-hours on the full "
        "test set; we therefore evaluate Whisper on a seeded sample of 100 test clips per speaker "
        "(400 clips) and MMS on the full test set. We report 95% bootstrap confidence intervals "
        "(1,000 resamples of clips)."
    ))

    doc.add_heading("4.4 Fine-tuning setup", level=2)
    para(doc, (
        "Whisper small is fully fine-tuned (learning rate 1e-5). For MMS-1B the base model is frozen "
        "and a new Khmer adapter and output layer are trained (learning rate 1e-3) on a character "
        "vocabulary built from the training text, which restores the digits the released adapter "
        "lacks. Training subsets of 10, 50 and 100 h are drawn by whole files with a fixed seed and are "
        "nested (10 h ⊂ 50 h ⊂ 100 h). Every run trains for 3.5 passes over its subset with batch size "
        "16 and 100 warm-up steps, so only the amount of data changes. Checkpoints are selected on "
        "validation (unseen sentences); Whisper transcripts longer than its 448-token decoder limit "
        "are excluded from training (about 1%). Training ran on one NVIDIA A40 (48 GB) at RunPod; "
        "test-set and FLEURS scores use the same evaluation code as the zero-shot baselines."
    ))
    para(doc, (
        "M2 changes only the initialisation of the MMS adapter: its 288 adapter tensors are copied "
        "from the released Khmer adapter, and each output-layer row for a character the released "
        "vocabulary also contains is copied too (105 of 111 at 10 h, 110 of 116 at 100 h); the "
        "remaining six (the word delimiter, è and the digits \u17e4 \u17e6 \u17e8 \u17e9) start new. Before any training "
        "this model already scores 10.07% CER on 40 validation clips, against 249.71% for a new "
        "head. The 10 h runs of both MMS variants are repeated with a second training seed on the "
        "same data. The final Whisper model trains on the whole training split for one pass "
        "(24,000 steps); it is reported separately from the 3.5-pass scaling runs. Besides the "
        "test set and FLEURS, every model is scored on the seen-speaker set (Table 2): new "
        "sentences read by training speakers."
    ))

    # ---- 5 Results
    doc.add_heading("5. Zero-Shot Results (M0)", level=1)
    if results:
        rows = []
        for r in sorted(results, key=lambda r: r["metrics"]["cer"]):
            m = r["metrics"]
            lo, hi = r["ci"]
            rows.append([
                r["label"], r["family"], f"{m['parameters_millions']:,.0f}",
                f"{m['clips']:,}", f"{pct(m['cer'])} [{pct(lo)}–{pct(hi)}]",
                pct(m["cer_with_spaces"]), f"{m['rtf']:.3f}",
            ])
        add_table(doc, "Table 4. Zero-shot results on the test set (unseen speakers and sentences). "
                       "CER ignores spaces; brackets give the 95% bootstrap interval.",
                  ["Model", "Family", "Params (M)", "Clips", "CER [95% CI]", "CER w/ spaces", "RTF"],
                  rows, [3.4, 2.5, 1.4, 1.2, 4.1, 1.7, 1.2])
    if mms:
        m = mms["metrics"]
        para(doc, (
            f"MMS-1B-all reaches {pct(m['cer'])} CER on the full test set "
            f"({m['clips']:,} clips, {m['audio_hours']:.1f} h). Its errors are mostly substitutions "
            f"({pct(m['substitution_rate'])}) and deletions ({pct(m['deletion_rate'])}), with "
            f"{pct(m['insertion_rate'])} insertions. It runs at a real-time factor of {m['rtf']:.3f}. "
            "Per test speaker, CER ranges from "
            f"{pct(min(m['cer_by_speaker'].values()))} to {pct(max(m['cer_by_speaker'].values()))}. "
            "Part of its error is structural: the released Khmer adapter's vocabulary has no symbol "
            "for the Khmer digits \u17e4, \u17e6, \u17e8 and \u17e9, so it can never output them."
        ))
    if whisper:
        loops = sum(r["metrics"]["clips_with_cer_over_100pct"] for r in whisper)
        total = sum(r["metrics"]["clips"] for r in whisper)
        para(doc, (
            f"All Whisper models exceed 100% CER, with {loops:,} of {total:,} evaluated clips above "
            "100%: decoding locks onto a repeated character until the token limit. This agrees with "
            "the 103.19% zero-shot CER reported for Whisper-large-v3 on FLEURS Khmer [5]. To find "
            f"the cause, we ran {diag['clips']} test clips ({diag['clips'] // 4} per speaker) through "
            "each Whisper model twice more (Table 5)."
        ))
        diag_rows = []
        for r in whisper:
            d = diag["models"][r["info"]["model"]["name_or_path"]]
            langs = d["detected_language_without_forcing"]
            diag_rows.append([
                r["label"],
                f"{langs.get('km', 0)} / {diag['clips']}",
                ", ".join(f"{k} {v}" for k, v in langs.items() if k != "km") or "—",
                f"{d['fp16_equals_fp32_outputs']} / {diag['clips']}",
            ])
        add_table(doc, "Table 5. Whisper diagnostics on 40 test clips: language detected when the "
                       "language is not forced, and whether fp16 and fp32 decoding agree.",
                  ["Model", "Detected as Khmer", "Detected as other", "fp16 = fp32"],
                  diag_rows, [4.4, 3.2, 4.4, 3.2])
        km_ok = [r["label"] for r in whisper
                 if diag["models"][r["info"]["model"]["name_or_path"]]
                 ["detected_language_without_forcing"].get("km", 0) >= 0.9 * diag["clips"]]
        km_bad = [r["label"] for r in whisper if r["label"] not in km_ok]
        para(doc, (
            "Numerical precision is not the cause: fp16 and fp32 decoding agree on almost every clip. "
            f"Language identification separates the models: {' and '.join(km_ok)} recognize the "
            f"audio as Khmer yet still collapse into repetition when transcribing it, while "
            f"{' and '.join(km_bad)} mostly labels it as Vietnamese or Thai. Zero-shot Whisper "
            "therefore knows it hears Khmer but cannot write it, which is the gap fine-tuning has "
            "to close."
        ))
    if not results:
        para(doc, "Results pending.")

    # ---- 6 Fine-tuning and scaling
    doc.add_heading("6. Fine-Tuning and Data Scaling (M1, S)", level=1)
    if ft:
        ft_rows = []
        for r in ft:
            t, f = r["test"], r["fleurs"]
            lo, hi = r["test_ci"]
            ft_rows.append([
                r["label"], f"{r['hours']} h", f"{r['train']['best_eval_cer'] * 100:.2f}%",
                f"{pct(t['cer'], 2)} [{pct(lo, 2)}–{pct(hi, 2)}]",
                pct(f["cer"], 2) if f else "—",
                f"{r['train']['train_seconds'] / 60:.0f}",
            ])
        add_table(doc, "Table 6. Fine-tuned models: validation CER of the selected checkpoint, test CER "
                       "with 95% bootstrap interval (all 3,712 clips), FLEURS CER (771 clips) and "
                       "training time on one A40. Validation uses 1,288 clips for MMS and 300 for "
                       "Whisper (decoding is slower).",
                  ["Model", "Data", "Validation", "Test CER [95% CI]", "FLEURS", "Train (min)"],
                  ft_rows, [3.0, 1.4, 2.0, 4.4, 1.9, 2.0])
        doc.add_picture(str(scaling_path), width=Cm(15.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph(style="Caption")
        cap.add_run("Figure 2. CER against training hours (log scale). Solid lines: the scaling runs; "
                    "dashed: MMS with the adapter started from the released Khmer adapter (M2, "
                    "Section 7); open diamond: Whisper small trained once on all 872 h (Section 8); "
                    "dotted: zero-shot MMS. Zero-shot Whisper (> 100%) is off the chart.")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    w50, m10, m50 = ftr("whisper", 50), ftr("mms", 10), ftr("mms", 50)
    if all((w10, w50, w100, m10, m50, m100)):
        para(doc, (
            "Fine-tuning helps both families in domain. Whisper small goes from unusable zero-shot "
            f"to {pct(w10['test']['cer'])} test CER with 10 h and {pct(w100['test']['cer'])} with "
            f"100 h. MMS starts from a strong zero-shot model ({pct(zero_shot['test'])}) and improves "
            f"steadily to {pct(m100['test']['cer'])}. At 10 h MMS is clearly better; at 50 h the two "
            f"are tied (Whisper {pct(w50['test']['cer'], 2)}, MMS {pct(m50['test']['cer'], 2)}, "
            "overlapping intervals); at 100 h Whisper is significantly better (non-overlapping "
            "intervals). Neither curve has flattened at 100 h."
        ), bold_lead="In domain. ")
        para(doc, (
            "On FLEURS the picture changes. Fine-tuned MMS stays close to its zero-shot score "
            f"({pct(m10['fleurs']['cer'])}, {pct(m50['fleurs']['cer'])}, "
            f"{pct(m100['fleurs']['cer'])} vs {pct(zero_shot['fleurs'])}): its new adapter, trained "
            "only on this corpus, first loses some of the generality of Meta's adapter and needs "
            "100 h to recover it. Whisper improves most with data "
            f"({pct(w10['fleurs']['cer'])} → {pct(w100['fleurs']['cer'])}) but stays far behind MMS. "
            "A model that wins on a same-corpus test set can therefore lose badly on other speech; "
            "out-of-domain evaluation is essential for Khmer."
        ), bold_lead="Out of domain. ")
        para(doc, (
            f"MMS trains only {m100['train']['setup_notes']['trainable_parameters'] / 1e6:.1f}M "
            "adapter parameters but runs a 1B-parameter encoder, so each step is slower: training "
            f"100 h took {m100['train']['train_seconds'] / 60:.0f} min for MMS versus "
            f"{w100['train']['train_seconds'] / 60:.0f} min for Whisper small on the same GPU. "
            f"Training all {len(ft)} runs cost {sum(r['train']['estimated_gpu_cost_usd'] for r in ft):.2f} "
            "USD of GPU time at 0.59 USD per hour."
        ), bold_lead="Cost. ")

    # ---- 7 M2
    doc.add_heading("7. Khmer-Specific Strategy: Adapter Initialisation (M2)", level=1)
    base = {10: "m1_mms", 100: "s_mms_100h"}
    if m2:
        rows7 = []
        for r in m2:
            b_run = base[r["hours"]]
            for run, init in ((b_run, "New adapter"), (r["run"], "Released khm adapter")):
                cells = [f"{r['hours']} h", init]
                for split in ("test", "seen", "fleurs"):
                    m = metric(run, split)
                    cells.append(pct(m["cer"], 2) if m else "—")
                rows7.append(cells)
            diffs = []
            for split in ("test", "seen", "fleurs"):
                if metric(b_run, split) and metric(r["run"], split):
                    d, lo, hi = paired(r["run"], b_run, split)
                    diffs.append(f"{d * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]")
                else:
                    diffs.append("—")
            rows7.append(["", "Difference (paired 95% CI)", *diffs])
        add_table(doc, "Table 7. MMS-1B with a new adapter versus an adapter started from the released "
                       "Khmer adapter, same data and steps. CER on the test set (unseen speakers), the "
                       "seen-speaker set and FLEURS; differences in CER points with a paired bootstrap "
                       "interval over the same clips.",
                  ["Data", "Adapter start", "Test", "Seen speakers", "FLEURS"],
                  rows7, [1.4, 4.6, 3.4, 3.4, 3.4])
    m2_10 = next((r for r in m2 if r["hours"] == 10), None)
    if m2_10 and m2_100:
        seeds = {k: v for k, v in SEED_PAIRS.items() if metric(v, "test")}
        seed_moves = [abs(metric(a_, sp)["cer"] - metric(b_, sp)["cer"])
                      for a_, b_ in seeds.items() for sp in ("test", "fleurs")]
        zs_d, zs_lo, zs_hi = paired(m2_100["run"], ZERO_SHOT, "fleurs")
        seed_text = (
            f"A second training seed changes the 10 h results by at most {max(seed_moves) * 100:.2f} "
            f"points (M2: {pct(m2_10['test']['cer'], 2)} and "
            f"{pct(metric(SEED_PAIRS['m2_mms_khm_init_10h'], 'test')['cer'], 2)} on the test set), far "
            "less than the M2 gain."
        ) if seeds else ""
        para(doc, (
            "Starting from the released adapter is better at both sizes and on all three sets, at "
            "the same training cost. The gain shrinks with data (about one point at 10 h, 0.6 at "
            "100 h) but does not vanish. Out of domain it removes the loss seen with a new adapter: "
            f"at 100 h the M2 model scores {pct(m2_100['fleurs']['cer'], 2)} on FLEURS, "
            f"{abs(zs_d) * 100:.2f} points [{abs(zs_hi) * 100:.2f}, {abs(zs_lo) * 100:.2f}] below "
            "zero-shot MMS, the only fine-tuned model in this study that improves on it. "
            f"{seed_text}"
        ))

    # ---- 8 Full data and speakers
    doc.add_heading("8. Full-Data Model and Speaker Familiarity", level=1)
    picks = [(ZERO_SHOT, "MMS-1B zero-shot", None), ("s_mms_100h", "MMS-1B, new adapter", 100),
             ("m2_mms_khm_init_100h", "MMS-1B, khm init (M2)", 100),
             ("s_whisper_small_100h", "Whisper small", 100),
             (FULL_RUN[3], "Whisper small (1 pass)", FULL_RUN[2])]
    rows8 = []
    for run, label, hours in picks:
        t, se, fl = metric(run, "test"), metric(run, "seen"), metric(run, "fleurs")
        if not (t and se):
            continue
        rows8.append([label, f"{hours} h" if hours else "—", pct(t["cer"], 2), pct(se["cer"], 2),
                      f"{(se['cer'] / t['cer'] - 1) * 100:+.0f}%", pct(fl["cer"], 2) if fl else "—"])
    if rows8:
        add_table(doc, "Table 8. Unseen versus seen speakers: CER on the test set (four held-out "
                       "speakers) and on new sentences read by training speakers (1,891 clips), the "
                       "relative change, and FLEURS. 73 seen-speaker clips are read by the speaker "
                       "excluded from training; removing them changes no value by more than 0.05 "
                       "points.",
                  ["Model", "Data", "Unseen speakers", "Seen speakers", "Change", "FLEURS"],
                  rows8, [4.4, 1.4, 2.6, 2.6, 1.7, 2.0])
    if full and m2_100:
        f0 = full[0]
        d_t = paired(f0["run"], m2_100["run"], "test")
        d_f = paired(f0["run"], m2_100["run"], "fleurs")
        para(doc, (
            f"Whisper small trained once over all {f0['hours']} h reaches "
            f"{pct(f0['test']['cer'], 2)} on the test set, {abs(d_t[0]) * 100:.2f} points below the "
            f"best MMS model, but {pct(f0['fleurs']['cer'], 2)} on FLEURS, {d_f[0] * 100:.2f} points "
            "above it, despite almost nine times more training data. Its validation CER was still "
            f"falling when the pass ended ({f0['train']['best_eval_cer'] * 100:.2f}%)."
        ), bold_lead="Full data. ")
        def seen_gain(run: str) -> float:
            return 1 - metric(run, "seen")["cer"] / metric(run, "test")["cer"]

        mms_gains = [seen_gain(r) for r in ("s_mms_100h", "m2_mms_khm_init_100h")]
        para(doc, (
            "Zero-shot MMS scores the same on both speaker sets, as it should. At 100 h, fine-tuned "
            f"MMS, which trains only a small adapter, is {pct(min(mms_gains), 0)}–"
            f"{pct(max(mms_gains), 0)} better on speakers it has heard; fully fine-tuned Whisper is "
            f"{pct(seen_gain('s_whisper_small_100h'), 0)} better, and {pct(seen_gain(f0['run']), 0)} "
            "with the full corpus. A split that let test speakers into training would therefore "
            "flatter Whisper by about a third and change the comparison between the two families."
        ), bold_lead="Speaker familiarity. ")

    # ---- 9 Error analysis
    doc.add_heading("9. Error Analysis", level=1)
    ea_dir = EXPERIMENTS / "error_analysis"
    if (ea_dir / "subsets.json").exists():
        classes = pd.read_csv(ea_dir / "class_rates.csv")
        subsets = load_json(ea_dir / "subsets.json")
        ea_models = [("MMS-1B zero-shot", "mms-1b-all"), ("MMS-1B khm init (M2) 100 h", "m2_mms_khm_init_100h"),
                     ("Whisper small 872 h", FULL_RUN[3])]
        test_cls = classes[classes["set"] == "test"]
        order = ["consonant", "dependent vowel", "coeng (subscript)", "diacritic / sign",
                 "independent vowel", "digit", "latin"]
        rows9 = []
        for cls in order:
            part = test_cls[test_cls["class"] == cls]
            if part.empty:
                continue
            chars = int(part["chars"].iloc[0])
            rows9.append([cls, f"{chars:,}"] + [
                pct(float(part[part["model"] == label]["error_rate"].iloc[0]), 1)
                for label, _ in ea_models])
        add_table(doc, "Table 9. Test-set error rate by character class ((S + D + I) / N, counting "
                       "substitutions and deletions of the class's reference characters and insertions "
                       "of characters of that class).",
                  ["Character class", "Characters"] + ["Zero-shot MMS", "M2 100 h", "Whisper 872 h"],
                  rows9, [4.2, 2.4, 2.9, 2.9, 2.9])
        m = subsets["models"]
        best_w, best_m = m[f"{FULL_RUN[3]}|test"], m["m2_mms_khm_init_100h|test"]
        para(doc, (
            "The most frequent error of every model is the subscript da written as subscript ta "
            "(\u17d2\u178a / \u17d2\u178f), and the second is the independent vowel \u17b2 written as "
            "\u17b1, as in \u17b2\u17d2\u1799 / \u17b1\u17d2\u1799. Both pairs sound the same and the corpus "
            "itself uses both spellings, so no model can choose correctly from the audio. Treating "
            "them as equal on both sides lowers test CER by 3–8% relative (Whisper 872 h "
            f"{pct(best_w['cer'], 2)} → {pct(best_w['cer_spelling_variants_merged'], 2)}; M2 100 h "
            f"{pct(best_m['cer'], 2)} → {pct(best_m['cer_spelling_variants_merged'], 2)}) without changing "
            "any ranking; the primary CER keeps protocol v1. The next errors are consonants of the "
            "two series that share a sound (\u1780/\u1782, \u1785/\u1787, \u1794/\u1796), the final "
            "m written as nikahit, and added or dropped subscripts."
        ), bold_lead="Spelling variants. ")
        ratios = [m[f"{run}|test"][key]["cer"] / m[f"{run}|test"]["cer"]
                  for _, run in ea_models for key in ("with_digits", "with_latin")]
        para(doc, (
            "Digits and Latin letters are rare (under 0.3% of test characters) but are the weakest "
            f"classes: clips that contain them have {min(ratios):.1f} to {max(ratios):.1f} times the "
            "model's overall test CER "
            f"(Whisper 872 h: {pct(best_w['with_digits']['cer'], 2)} with digits, "
            f"{pct(best_w['with_latin']['cer'], 2)} with Latin words). Whisper 872 h writes "
            f"{pct(best_w['latin_written_as_other_script'] / best_w['latin_reference_chars'], 0)} of the "
            f"Latin letters in the test references ({best_w['latin_written_as_other_script']} of "
            f"{best_w['latin_reference_chars']}) as Khmer characters, M2 "
            f"{pct(best_m['latin_written_as_other_script'] / best_m['latin_reference_chars'], 0)}: "
            "Whisper transliterates English words into Khmer script more often. Zero-shot MMS misses almost "
            "half of all digits, partly because it cannot write four of them."
        ), bold_lead="Digits and Latin words. ")
        para(doc, (
            f"The {subsets['too_long_test_clips']} test transcripts longer than Whisper's decoder "
            f"limit score {pct(best_w['too_long_for_whisper']['cer'], 1)} for Whisper 872 h against "
            f"{pct(best_m['too_long_for_whisper']['cer'], 1)} for M2, while the rest score "
            f"{pct(best_w['fits_whisper']['cer'], 2)}; long-form input needs segmentation before "
            "Whisper can be used on it. Every model is worse on the two female test speakers "
            f"(Whisper 872 h {pct(best_w['cer_by_gender']['f'], 2)} vs "
            f"{pct(best_w['cer_by_gender']['m'], 2)}; M2 {pct(best_m['cer_by_gender']['f'], 2)} vs "
            f"{pct(best_m['cer_by_gender']['m'], 2)}), mostly because of one speaker, so with four "
            "test speakers this is a speaker effect rather than evidence of a gender gap."
        ), bold_lead="Long transcripts and speakers. ")

    # ---- 10 Planned
    doc.add_heading("10. Planned Work", level=1)
    bullets(doc, [
        ("MMS from the released Khmer adapter on the full training split, the likely best model "
         "out of domain."),
        "A second pass for the full-data Whisper model, whose validation CER was still falling.",
        ("A normalization protocol v2 that unifies the spelling variants found in Section 9, "
         "decided before the test set is locked."),
    ])

    # ---- 11 Limitations
    doc.add_heading("11. Limitations", level=1)
    bullets(doc, [
        (f"The corpus has only {s['speakers']} speakers and contains read cultural-domain speech; "
         "results may not transfer to spontaneous or other-domain speech."),
        "Whisper zero-shot results use a 400-clip sample; MMS uses the full test set.",
        "MMS-1B-all is licensed CC BY-NC 4.0, which restricts it to non-commercial research.",
        ("Only the two 10 h MMS runs were repeated with a second seed; other configurations are "
         "single runs, with bootstrap intervals over clips."),
        ("Bootstrap intervals and paired tests resample clips. Errors from one speaker are "
         "correlated, and resampling sentences gives narrower intervals than resampling speakers "
         "[13], so the intervals may be optimistic; with four test speakers, speaker-level "
         "resampling would be very wide."),
        ("The full-data Whisper model trained for one pass and the scaling runs for 3.5, so it is "
         "not a point on the same curve."),
        ("Validation sizes differ between families (1,288 clips for MMS, 300 for Whisper), so "
         "validation CERs are not directly comparable; test and FLEURS use identical clips."),
        ("24 test clips (0.65%) have transcripts longer than Whisper's 448-token decoder limit and "
         "count against Whisper."),
        ("Inference speed was measured on different GPUs (laptop RTX 4060 for zero-shot and 10 h "
         "models, A40 for the rest), so it is not compared across runs."),
        "Training subsets are drawn by whole files, so small subsets cover fewer topics.",
        (f"The split leaves {sp['train']['speakers']} training speakers and excludes "
         f"{excluded_label}, one of only {n_over_60} speakers aged 60 or over."),
        "The split option is provisional until approved; the test set will then be locked.",
    ])

    # ---- 12 Conclusion
    doc.add_heading("12. Conclusion", level=1)
    para(doc, (
        "A full audit of the largest open Khmer speech corpus shows that its sentences are shared "
        "across speakers, which makes leakage-free evaluation essential. Zero-shot Whisper is "
        "unusable for Khmer while MMS with a Khmer adapter is already a strong starting point. "
        "Fine-tuning lets Whisper small overtake MMS in domain, reaching under 3% CER with the full "
        "corpus, but MMS remains far more robust on other speech. Starting the MMS adapter from the "
        "released Khmer adapter, rather than from scratch, helps at every data size and gives the "
        "best out-of-domain model. Which model to use therefore depends on whether the target "
        "speech resembles the training corpus."
    ))
    para(doc, (
        "Code, configurations and split manifests: github.com/chitrasem001/khmer-asr-research."
    ), bold_lead="Reproducibility. ")

    doc.add_heading("References", level=1)
    for i, ref in enumerate(REFERENCES, 1):
        p = doc.add_paragraph(f"[{i}] {ref}")
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.6)
        for run in p.runs:
            run.font.size = Pt(9)

    out = OUT / "khmer_asr_paper_draft.docx"
    doc.save(out)
    return out


if __name__ == "__main__":
    print(build())
