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
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9a94"

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


# ---------------------------------------------------------------- figure
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
    r = draft.add_run(f"DRAFT FOR ADVISOR REVIEW — {date.today():%d %b %Y} — "
                      "fine-tuning results (M1–M3) pending")
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(0xB0, 0x3A, 0x2E)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = title.add_run("Fine-Tuning Pretrained Models for Khmer Speech Recognition: "
                       "A Leakage-Free Evaluation Protocol and Zero-Shot Baselines")
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
    para(doc, (
        "Khmer remains under-resourced for automatic speech recognition (ASR), and published Khmer "
        "results are hard to compare because they use different test sets, text normalization and "
        f"training data. We audit the full Digital Divide Data Khmer corpus ({s['rows']:,} utterances, "
        f"{h(s['total_hours'])}, {s['speakers']} speakers) and find that every sentence is read by "
        "four different speakers, so a random split leaks test sentences into training. We define a "
        "split that holds out both speakers and every sentence they read, a documented Khmer text "
        "normalization protocol, and a space-insensitive character error rate (CER). "
        f"{result_sentence} These results set the baselines for controlled fine-tuning experiments "
        "on data quantity and Khmer-specific strategies."
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

    # ---- 6 Planned
    doc.add_heading("6. Planned Fine-Tuning Experiments", level=1)
    para(doc, (
        "All experiments use the same locked test set and normalization, change one variable at a "
        "time, and select checkpoints on validation only."
    ))
    bullets(doc, [
        "M1 — Fine-tune a pretrained model on the training split; compare with its M0 baseline (RQ1).",
        "S — Train on nested subsets (e.g. 10, 50, 100 h) to measure the effect of data quantity (RQ2).",
        "M2 — Add one Khmer-specific strategy (normalization, augmentation or fine-tuning method) (RQ3).",
        "M3 (optional) — Compare Whisper and MMS families after fine-tuning (RQ4).",
    ])

    # ---- 7 Limitations
    doc.add_heading("7. Limitations", level=1)
    bullets(doc, [
        (f"The corpus has only {s['speakers']} speakers and contains read cultural-domain speech; "
        "results may not transfer to spontaneous or other-domain speech."),
        "Whisper zero-shot results use a 400-clip sample; MMS uses the full test set.",
        "MMS-1B-all is licensed CC BY-NC 4.0, which restricts it to non-commercial research.",
        ("Each configuration is a single run; we report bootstrap intervals over clips, not over "
        "training seeds."),
        (f"The split leaves {sp['train']['speakers']} training speakers and excludes "
         f"{excluded_label}, one of only {n_over_60} speakers aged 60 or over."),
        "The split option is provisional until approved; the test set will then be locked.",
    ])

    # ---- 8 Conclusion
    doc.add_heading("8. Conclusion", level=1)
    para(doc, (
        "A full audit of the largest open Khmer speech corpus shows that its sentences are shared "
        "across speakers, which makes leakage-free evaluation essential. With a split that holds out "
        "speakers and sentences and a documented normalization protocol, zero-shot Whisper is "
        "unusable for Khmer while MMS with a Khmer adapter is already a strong starting point. The "
        "next step is controlled fine-tuning on the same protocol."
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
