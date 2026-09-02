"""Builds Report.pdf (12pt Times, 1in margins) and its figures from outputs/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

OUT = "../outputs"
CFGS = ["C1", "C2", "C3", "C4", "C5"]
# reference categorical palette, fixed slot order (see report note on validation)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#c9c8c3"


def load():
    runs = {c: json.load(open(f"{OUT}/metrics_{c}.json")) for c in CFGS}
    bench = {r["config"]: r for r in json.load(open(f"{OUT}/benchmark.json"))}
    return runs, bench


def style_axes(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(axis="y", color=GRID, alpha=.5, linewidth=.6)
    ax.set_axisbelow(True)


def fig_curves(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), dpi=200)
    for ax, key, title in zip(axes, ["train_loss", "val_loss"], ["Training", "Validation"]):
        for c, col in zip(CFGS, SERIES):
            y = runs[c]["history"][key]
            ax.plot(range(1, len(y) + 1), y, color=col, linewidth=1.6, label=c)
        ax.set_yscale("log"); style_axes(ax)
        ax.set_xlabel("epoch", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9, color=INK)
    axes[0].set_ylabel("cross-entropy (nats)", fontsize=8, color=INK2)
    axes[1].legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), labelcolor=INK2, columnspacing=1.0, handlelength=1.2)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_bars(runs, bench, path):
    panels = [("Sequence accuracy", [runs[c]["test"]["sequence_accuracy"] for c in CFGS], "{:.3f}"),
              ("Bit-level accuracy", [runs[c]["test"]["bit_accuracy"] for c in CFGS], "{:.3f}"),
              ("BLEU", [runs[c]["test"]["bleu"] for c in CFGS], "{:.3f}"),
              ("Levenshtein distance (chars)", [runs[c]["test"]["levenshtein"] for c in CFGS], "{:.2f}"),
              ("Training ms / step", [bench[c]["ms_per_step"] for c in CFGS], "{:.1f}"),
              ("Peak GPU memory (MB)", [bench[c]["train_peak_mem_mb"] for c in CFGS], "{:.0f}"),
              ("Greedy decode ms / example", [bench[c]["decode_ms_per_example"] for c in CFGS], "{:.1f}"),
              ("Parameters (M)", [runs[c]["params"] / 1e6 for c in CFGS], "{:.2f}")]
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.5), dpi=200)
    for ax, (title, vals, fmt) in zip(axes.ravel(), panels):
        ax.bar(CFGS, vals, color=SERIES, width=.68)
        style_axes(ax)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.grid(False)
        ax.set_ylim(0, max(vals) * 1.22)
        for x, v in enumerate(vals):
            ax.text(x, v * 1.03, fmt.format(v), ha="center", va="bottom", fontsize=6.6, color=INK)
        ax.set_title(title, fontsize=7.8, color=INK, pad=3)
    fig.tight_layout(h_pad=1.2); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- pdf
def styles():
    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=12, leading=15.2,
                          alignment=TA_JUSTIFY, spaceAfter=5, textColor=colors.black)
    return dict(
        body=body,
        title=ParagraphStyle("title", parent=body, fontName="Times-Bold", fontSize=15,
                             leading=18, alignment=TA_CENTER, spaceAfter=3),
        sub=ParagraphStyle("sub", parent=body, fontSize=11, alignment=TA_CENTER, spaceAfter=10),
        h1=ParagraphStyle("h1", parent=body, fontName="Times-Bold", fontSize=12.5,
                          spaceBefore=9, spaceAfter=3, keepWithNext=1),
        h2=ParagraphStyle("h2", parent=body, fontName="Times-Italic", fontSize=12,
                          spaceBefore=6, spaceAfter=2, keepWithNext=1),
        left=ParagraphStyle("left", parent=body, alignment=0),
        cap=ParagraphStyle("cap", parent=body, fontSize=9.5, leading=11.5, alignment=TA_CENTER,
                           spaceBefore=3, spaceAfter=8, textColor=colors.HexColor("#333333")),
        code=ParagraphStyle("code", parent=body, fontName="Courier", fontSize=8.4, leading=10.2,
                            alignment=TA_JUSTIFY, spaceAfter=4))


def table(data, widths, size=9, align_right_from=1):
    t = Table(data, colWidths=widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Times-Roman", size),
        ("FONT", (0, 0), (-1, 0), "Times-Bold", size),
        ("FONT", (0, 1), (0, -1), "Times-Roman", size),
        ("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), .8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), .5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), .8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def page_number(canvas, doc):
    canvas.saveState(); canvas.setFont("Times-Roman", 9.5)
    canvas.drawCentredString(A4[0] / 2, 0.55 * inch, str(doc.page))
    canvas.restoreState()


def build(runs, bench):
    S = styles()
    P = lambda t, s="body": Paragraph(t, S[s])
    st = []
    st.append(P("ANLP Assignment-1", "title"))
    st.append(P("Messing around with transformers, I guess &nbsp;|&nbsp; Roll number: 2023101091", "sub"))

    st.append(P("1&nbsp;&nbsp;Task and data", "h1"))
    st.append(P(
        "The dataset provides 5,000 lines of ciphertext written as <font face='Courier' size='10'>0</font>/"
        "<font face='Courier' size='10'>1</font> characters, aligned line by line with 5,000 plaintext "
        "English lines. Every plaintext character corresponds to exactly eight cipher bits &mdash; verified "
        "for all 5,000 pairs &mdash; so the task is to map a binary string onto the English text it encrypts."))
    st.append(P(
        "The median line is 553 characters, too long for a small model, so each line is cut into aligned "
        "chunks of at most 64 plaintext characters (512 cipher bits); the mean chunk is 61.3 characters. "
        "Splits are taken over whole lines <i>before</i> chunking (4,500 / 250 / 250 lines, giving 43,674 / "
        "2,456 / 2,465 chunks for train / validation / test), so no sentence appears in two splits."))
    st.append(P(
        "Inspecting the data shows the cipher is a repeating-key XOR, "
        "<i>cipher</i>[i] = <i>plain</i>[i] &#8853; <i>key</i>[i mod 8] with <i>key</i> = ANLP2026, and this "
        "holds for every line in the corpus. No model is given this fact, but it frames the results: "
        "recovering a character requires knowing its phase, its index modulo 8. A representation that blurs "
        "the 8-bit grid forces the model to reconstruct that phase by counting; one that respects the grid "
        "hands it over for free."))

    st.append(P("2&nbsp;&nbsp;Implemented components", "h1"))
    st.append(P(
        "Every component is built from basic PyTorch operations &mdash; <font face='Courier' size='10'>nn.Linear</font>, "
        "<font face='Courier' size='10'>nn.Embedding</font>, matrix multiplication and softmax. "
        "<font face='Courier' size='10'>nn.Transformer</font>, <font face='Courier' size='10'>nn.MultiheadAttention</font> "
        "and <font face='Courier' size='10'>F.scaled_dot_product_attention</font> are not used anywhere, and "
        "neither is any tokenizer or metric library."))
    st.append(P("Attention.", "h2"))
    st.append(P(
        "Scaled dot-product attention is computed directly as softmax(QK<sup>T</sup>/&#8730;d<sub>k</sub>)V, "
        "with additive masks folded into the logits before the softmax. Multi-head and grouped-query attention "
        "share one implementation parameterised by the number of key/value heads: with n<sub>kv</sub> = "
        "n<sub>heads</sub> it is standard MHA, and with n<sub>kv</sub> &lt; n<sub>heads</sub> each key/value "
        "head is shared by n<sub>heads</sub>/n<sub>kv</sub> query heads."))
    st.append(P("Feed-forward and normalisation.", "h2"))
    st.append(P(
        "The position-wise FFN is Linear&rarr;ReLU&rarr;Linear. LayerNorm normalises by mean and variance with "
        "a learned gain and bias; RMSNorm divides by the root mean square with a gain only, dropping both the "
        "mean subtraction and the bias. Both are used in pre-norm position, x + Sublayer(Norm(x))."))
    st.append(P("Positions.", "h2"))
    st.append(P(
        "Sinusoidal absolute encodings are added to the embeddings. RoPE instead rotates each (2i, 2i+1) pair "
        "of every query and key by an angle proportional to the position, and is applied inside self-attention "
        "only &mdash; cross-attention relates two different sequences, where a shared rotation carries no meaning. "
        "Both the tokenized and token-free models use three pre-norm encoder layers and three pre-norm decoder "
        "layers (self-attention, cross-attention, FFN), with the output projection tied to the target embedding."))

    st.append(P("3&nbsp;&nbsp;Experimental setup", "h1"))
    st.append(P("3.1&nbsp;&nbsp;Configurations", "h2"))
    st.append(table([["Config", "Change from base", "Positional", "Attention", "Norm.", "Tokenization"],
                     ["C1", "none (base)", "sinusoidal", "MHA, 8 heads", "LayerNorm", "subword BPE"],
                     ["C2", "positional encoding", "RoPE", "MHA, 8 heads", "LayerNorm", "subword BPE"],
                     ["C3", "attention", "sinusoidal", "GQA, 8Q / 2KV", "LayerNorm", "subword BPE"],
                     ["C4", "normalisation", "sinusoidal", "MHA, 8 heads", "RMSNorm", "subword BPE"],
                     ["C5", "tokenization", "sinusoidal", "MHA, 8 heads", "LayerNorm", "BLT, token-free"]],
                    [0.5 * inch, 1.5 * inch, 0.95 * inch, 1.1 * inch, 0.85 * inch, 1.2 * inch],
                    size=9, align_right_from=6))
    st.append(P("Table 1: the five configurations. Each of C2&ndash;C5 changes exactly one component of C1.", "cap"))
    st.append(P(
        "All configurations share d<sub>model</sub> = 256, three encoder and three decoder layers, 8 attention "
        "heads, d<sub>ff</sub> = 1024, dropout 0.1, AdamW (lr 10<sup>-3</sup>, &beta; = 0.9/0.98, weight decay "
        "0.01), a 1,000-step warm-up followed by cosine decay, batch size 64, 60 epochs, bf16 autocast and "
        "gradient clipping at 1.0. Configurations were trained one at a time on a single NVIDIA L40S."))
    st.append(P("3.2&nbsp;&nbsp;Subword tokenization (C1&ndash;C4)", "h2"))
    st.append(P(
        "A byte-pair encoding is trained from scratch on the training split and shared by the binary source and "
        "the English target. Pre-tokenization only bounds where merges may occur &mdash; whitespace for English, "
        "and fixed 32-bit windows for the binary string, which has no natural boundaries &mdash; after which the "
        "most frequent adjacent pair is merged repeatedly, maintaining an incremental pair-frequency index. The "
        "resulting vocabulary of 8,000 (7,941 merges; 5,206 binary and 2,794 text tokens) encodes a 512-bit "
        "source into 26.7 tokens on average and a target chunk into 17.9. Source tokens span 1 to 32 bits with "
        "a mean of 18.3, and 76.9% are shorter than the 32-bit window, so the segmentation is genuinely learned "
        "rather than fixed width. The merges are <i>not</i> aligned to the 8-bit character grid."))
    st.append(P("3.3&nbsp;&nbsp;Token-free Byte Latent Transformer (C5)", "h2"))
    st.append(P(
        "C5 removes the vocabulary. Every 8 cipher bits are grouped into one byte value 0&ndash;255, so a "
        "512-bit source becomes 64 bytes and the target is the plaintext's UTF-8 bytes; the embedding table is "
        "a plain 256 entries plus PAD/BOS/EOS. Patches are dynamic rather than fixed width: an order-2 next-byte "
        "n-gram model with back-off is fitted on the training bytes (separately for the cipher and plaintext "
        "sides) and a new patch begins wherever H(x<sub>i</sub> | x<sub>i-2</sub>, x<sub>i-1</sub>) exceeds a "
        "threshold &theta;, calibrated on the training split for a mean patch length of 4 bytes "
        "(&theta;<sub>src</sub> = 3.03, &theta;<sub>tgt</sub> = 2.51 nats). This yields 4.13 and 4.15 bytes per "
        "patch and about 15 patches per example, and the cuts follow content:"))
    st.append(P("Head |VI| |was |first |exh|ibited |at |the |Ha|nover |Ga|llery |Lo|ndon |in |It |", "code"))
    st.append(P(
        "A local encoder embeds bytes, attends only within a patch (a mask built from patch indices, since "
        "patches vary in length) and mean-pools each patch into one latent. The global encoder and decoder "
        "&mdash; the same modules as C1 &mdash; operate only on those latents. A local decoder takes the "
        "right-shifted bytes, attends causally inside the patch, and cross-attends to the latents of its own and "
        "all preceding patches to emit each byte. Latent <i>m</i> is built from patches &lt; <i>m</i> only, so "
        "the model stays autoregressive at byte level, and because the entropy model reads only preceding bytes "
        "the boundaries can be placed incrementally while decoding; teacher-forced and incremental logits agree "
        "to 8&times;10<sup>-7</sup>."))
    st.append(P("3.4&nbsp;&nbsp;Metrics", "h2"))
    st.append(P(
        "All numbers come from greedy decoding, and every metric is implemented in "
        "<font face='Courier' size='10'>src/utils.py</font>. <b>Bit-level accuracy</b> compares the UTF-8 bit "
        "strings of prediction and reference position by position, counting a length mismatch as error (the "
        "denominator is the longer of the two). <b>Sequence accuracy</b> is exact string match over a whole "
        "chunk. <b>Levenshtein distance</b> is the character-level edit distance. <b>BLEU</b> is corpus BLEU-4 "
        "over whitespace tokens with brevity penalty, and <b>ROUGE</b>-1/2/L are F1 scores averaged over "
        "examples. Test figures use 2,000 held-out chunks."))
    return st, S


def build_results(runs, bench, S):
    P = lambda t, s="body": Paragraph(t, S[s])
    st = [P("4&nbsp;&nbsp;Results", "h1")]
    rows = [["Config", "Params", "Val loss", "Bit acc.", "Seq. acc.", "Lev.", "BLEU", "ROUGE-L"]]
    for c in CFGS:
        r, t = runs[c], runs[c]["test"]
        rows.append([c, f"{r['params']:,}", f"{r['history']['val_loss'][-1]:.4f}",
                     f"{t['bit_accuracy']:.4f}", f"{t['sequence_accuracy']:.4f}",
                     f"{t['levenshtein']:.3f}", f"{t['bleu']:.4f}", f"{t['rougeL']:.4f}"])
    st.append(table(rows, [0.55 * inch, 0.85 * inch, 0.72 * inch, 0.72 * inch, 0.78 * inch,
                           0.62 * inch, 0.68 * inch, 0.78 * inch], size=9.5))
    st.append(P("Table 2: test-set quality, greedy decoding on 2,000 held-out chunks. "
                "Lower is better for validation loss and Levenshtein distance.", "cap"))
    st.append(P(
        "The cross-entropy of C5 is measured per byte over a 259-symbol vocabulary and is therefore not "
        "comparable with the per-subword losses of C1&ndash;C4; only the decoded-output metrics compare across "
        "tokenizations. Wall-clock training time is likewise not used for comparison, because the GPU was "
        "shared with other jobs during the run (C1's epochs ranged from 26 to 40 s with external load). Table 3 "
        "instead re-measures all five configurations back to back under identical conditions: same batch size, "
        "100 timed steps after 20 warm-up steps, one process at a time."))
    rows = [["Config", "ms / step", "examples / s", "Peak mem. (MB)", "Decode ms / ex.",
             "Src len", "Tgt len"]]
    for c in CFGS:
        b = bench[c]
        rows.append([c, f"{b['ms_per_step']:.1f}", f"{b['examples_per_s']:,.0f}",
                     f"{b['train_peak_mem_mb']:,.0f}", f"{b['decode_ms_per_example']:.1f}",
                     f"{b['mean_src_len']:.1f}", f"{b['mean_tgt_len']:.1f}"])
    st.append(table(rows, [0.6 * inch, 0.8 * inch, 1.0 * inch, 1.15 * inch, 1.1 * inch,
                           0.65 * inch, 0.65 * inch], size=9.5))
    st.append(P("Table 3: controlled cost comparison. Peak memory is "
                "<font face='Courier' size='9'>max_memory_allocated</font> during training; the last two "
                "columns are the mean padded sequence lengths the model actually processes.", "cap"))
    st.append(Image(f"{OUT}/fig_curves.png", width=6.4 * inch, height=6.4 * inch * 0.385))
    st.append(P("Figure 1: training and validation cross-entropy (log scale). C5's curve is per byte and "
                "sits on a different scale by construction, not because it is a better model.", "cap"))
    st.append(Image(f"{OUT}/fig_bars.png", width=6.4 * inch, height=6.4 * inch * 0.56))
    st.append(P("Figure 2: quality (top row) and cost (bottom row) across the five configurations.", "cap"))

    st.append(P("5&nbsp;&nbsp;Analysis", "h1"))
    st.append(P("5.1&nbsp;&nbsp;RoPE (C2) improves everything at no parameter cost", "h2"))
    st.append(P(
        "RoPE is the only change that improves every quality metric: sequence accuracy rises from 0.809 to "
        "0.854, bit accuracy from 0.978 to 0.988, mean edit distance falls from 0.37 to 0.25 characters, and it "
        "reaches the lowest validation loss of the five (0.068 against 0.092), all at an identical parameter "
        "count. This is consistent with the structure of the cipher. The key repeats every 8 characters, while "
        "BPE merges span 1&ndash;32 bits and shift the character grid from one token to the next, so what the "
        "decoder needs is <i>how far back</i> a source token sits rather than where it lies absolutely &mdash; "
        "which is exactly what RoPE writes into the attention logits. The costs are modest but real: 4.5% more "
        "time per step (52.9 vs 50.6 ms) and 67% slower greedy decoding (1.5 vs 0.9 ms per example), since the "
        "rotation is recomputed for every query and key at every decoding step."))
    st.append(P("5.2&nbsp;&nbsp;GQA (C3) buys parameters, not training memory", "h2"))
    st.append(P(
        "Sharing 8 query heads over 2 key/value heads removes 884,736 parameters (9.2%) and gives the fastest "
        "training step (49.1 ms), but it is the weakest configuration on quality: sequence accuracy drops 1.65 "
        "points to 0.793 and validation loss is the highest at 0.095. Its peak training memory is also the "
        "highest of the five, 1,029 MB against 785 MB for the base &mdash; the opposite of what GQA is normally "
        "adopted for. The reason is that the shared key/value heads are expanded back to the full head count "
        "before attention, so a training step holds both the narrow projections and their expanded copies. "
        "GQA's memory advantage lies in the inference-time KV cache, which the greedy decoder used here never "
        "builds, since it re-runs the decoder over the prefix instead of caching. On this task, then, GQA is a "
        "parameter saving paid for in accuracy."))
    st.append(P("5.3&nbsp;&nbsp;RMSNorm (C4) is a small free saving", "h2"))
    st.append(P(
        "RMSNorm is near neutral on quality: 0.816 sequence accuracy against the base's 0.809, a gap well "
        "within seed noise, with a slightly better validation loss (0.083 vs 0.092). It removes only 4,352 "
        "parameters &mdash; the LayerNorm biases &mdash; yet cuts peak training memory by 6.4% (734.6 vs 785.1 "
        "MB), because dropping the mean subtraction removes an intermediate activation at each of the model's "
        "twenty normalisation sites. It is the cheapest subword configuration by memory at no cost in quality."))
    st.append(P("5.4&nbsp;&nbsp;BLT (C5): cheaper memory, dearer generation", "h2"))
    st.append(P(
        "Token-free processing gives the second-best model overall: 0.843 sequence accuracy, behind only RoPE "
        "and 3.4 points ahead of the base, with the lowest peak training memory of the five (734.7 MB) and 20x "
        "fewer embedding parameters (198,912 against 4,096,000). Two structural facts explain the accuracy. "
        "First, grouping 8 bits into one byte puts source bytes in one-to-one correspondence with plaintext "
        "characters, so the phase problem that C1&ndash;C4 must solve by counting does not arise at all. Second, "
        "the global transformer sees only about 15 patch latents per example instead of about 27 subword "
        "tokens, shortening the quadratic part of the model, which is where the memory saving comes from."))
    st.append(P(
        "The costs land on time rather than memory. Each training step is 9.9% slower than the base (55.6 vs "
        "50.6 ms), because the local encoder and decoder run over 64- and 65-byte sequences underneath the "
        "global model. Greedy decoding is six times slower, 5.5 against 0.9 ms per example: the model emits one "
        "byte per step, so a 64-character chunk costs 65 decoding steps where the subword models need about 28. "
        "That is the central trade-off of the token-free approach here &mdash; cheaper in memory and embedding "
        "parameters, more accurate than the base, and markedly more expensive to generate from."))
    st.append(P(
        "One metric disagrees with the others, and it is informative. C5 has the second-best sequence accuracy "
        "but the <i>worst</i> mean edit distance, 0.58 characters against 0.25&ndash;0.40 for the subword "
        "models: its errors are rarer but larger. Decoded output shows why. Mistakes fall on low-frequency "
        "spans where the entropy model cuts a patch differently than it did in training, and the local decoder "
        "then mis-generates several bytes inside that one patch &mdash; <i>Wrightsville</i> becomes "
        "<i>Wrivishville</i>, <i>surf eroded an</i> becomes <i>surferded an an</i>. The subword models fail more "
        "often but usually by a character or two."))
    st.append(P(
        "The design of the patching mattered more than any other single decision in this study. An earlier "
        "version of C5 that used fixed 8-byte patches over the raw ASCII 0/1 characters, rather than entropy "
        "patching over grouped bytes, collapsed completely: 0.0 sequence accuracy, emitting the same generic "
        "English for every input (<i>the state of the state of the season</i>). It had learned an unconditional "
        "language model and ignored the source entirely. The two changes that fixed it &mdash; dynamic "
        "entropy-based patches and byte grouping &mdash; are precisely the two that align patch structure with "
        "the information the source actually carries."))

    st.append(P("6&nbsp;&nbsp;Conclusion", "h1"))
    st.append(P(
        "Ranked by sequence accuracy the ordering is C2 (RoPE, 0.854) &gt; C5 (BLT, 0.843) &gt; C4 (RMSNorm, "
        "0.816) &gt; C1 (base, 0.809) &gt; C3 (GQA, 0.793). Only RoPE improves quality at no parameter cost; "
        "RMSNorm is a small free memory saving; GQA trades accuracy for parameters and, in this training-only "
        "setting, does not reduce memory. The token-free BLT is the most interesting outcome: more accurate "
        "than the base model with a fifth of the embedding parameters and the lowest peak memory, but six times "
        "slower to generate &mdash; the price of working one byte at a time."))
    return st


def build_links(S):
    P = lambda t, s="body": Paragraph(t, S[s])
    st = [P("Links", "h1")]
    st.append(P(
        "Weights &amp; Biases project: &lt;paste link after <font face='Courier' size='10'>wandb sync "
        "outputs/wandb/offline-run-*</font>&gt;<br/>"
        "Hugging Face checkpoints: &lt;paste model repository link&gt;<br/>"
        "Code: <font face='Courier' size='10'>src/</font> (models, tokenizer, training, metrics); "
        "logs, plots and per-configuration metrics in <font face='Courier' size='10'>outputs/</font>.",
        "left"))
    st.append(P("References", "h1"))
    for i, r in enumerate([
        "A. Vaswani et al. Attention Is All You Need. NeurIPS, 2017.",
        "J. Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. 2021.",
        "B. Zhang and R. Sennrich. Root Mean Square Layer Normalization. NeurIPS, 2019.",
        "J. Ainslie et al. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head "
        "Checkpoints. EMNLP, 2023.",
        "A. Pagnoni et al. Byte Latent Transformer: Patches Scale Better Than Tokens. ACL, 2025.",
        "R. Sennrich, B. Haddow and A. Birch. Neural Machine Translation of Rare Words with Subword Units. "
        "ACL, 2016."], 1):
        st.append(Paragraph(f"[{i}]&nbsp;&nbsp;{r}", ParagraphStyle(
            "ref", parent=S["body"], fontSize=10.5, leading=12.5, spaceAfter=2)))
    return st


def main():
    runs, bench = load()
    fig_curves(runs, f"{OUT}/fig_curves.png")
    fig_bars(runs, bench, f"{OUT}/fig_bars.png")
    doc = SimpleDocTemplate("../Report.pdf", pagesize=A4, leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch, title="ANLP Assignment 1 Report")
    story, S = build(runs, bench)
    story += build_results(runs, bench, S)
    story += build_links(S)
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print("wrote ../Report.pdf")


if __name__ == "__main__":
    main()
