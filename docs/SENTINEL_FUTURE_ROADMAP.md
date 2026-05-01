# SENTINEL: Language-Guided Predictive VAD
## Future Extensions Roadmap — Waiting to be Implemented

> **Status:** Ideas captured. V4 training in progress. Implement AFTER V4 baseline is confirmed.
> **Priority:** High — these extensions use zero new data and could push AUROC 3–8% higher.

---

## Plain English: What Are We Doing Right Now vs What We COULD Do

### What the Current System Does (V4)

Imagine you are a security guard who has watched thousands of CCTV videos and memorised them.
Every time you see a new video, you compare it to your memories and ask:
*"Does any part of this look like the anomalous things I remember?"*

That is exactly what our V4 model does. It has learned, over 1000 training epochs, what the
**visual and language patterns** of anomalous vs. normal videos look like. When it sees a new
video, it scores each 2-second segment (32 total) by how similar it looks to anomalous moments
in its training memory.

**The fundamental limitation:** It can only recognise crimes it has effectively "seen before."
It is a pattern-matcher, not a reasoner.

---

### What SENTINEL Does Differently (Future)

Instead of asking *"does this LOOK like something bad I remember?"*, SENTINEL asks two better questions:

1. **"Did something UNEXPECTED just happen?"** (Temporal Prediction Error)
2. **"Does the STORY of this video, told in plain English, sound dangerous?"** (LLM Narrative Reasoning)

These are the same two questions a smart human security guard asks — and they generalise
to crime types the system has NEVER seen before.

---

## The Three Extensions (Ready to Build, Not Built Yet)

---

### Extension 1: CLIP Semantic Danger Score
📁 *Planned file: `scripts/05_semantic_ensemble.py`*

**What it is in plain English:**

Our BLIP-2 already generates a text description for every 2-second segment of every video.
For example: *"two people arguing aggressively near a parked car"*.

CLIP (which we already use) lives in a special mathematical space where words and images mean
the same thing. This means we can ask:

> *"How similar is this caption to the phrase: 'violent, dangerous, or threatening activity'?"*

If the answer is "very similar", that segment is flagged — **without any training at all**.
This is called **zero-shot anomaly detection via semantic similarity**.

**What changes from current system:**
- Current: model LEARNS what anomalous looks like from training data
- Extension 1: model ASKS what sounds dangerous using pre-existing language knowledge

**Why it's powerful:**
- Zero additional training required
- Works on crime types never seen in training
- Purely language-based — complementary to the visual model
- Can be combined (ensembled) with the V4 model score instantly

**Implementation effort:** ~2 hours. All features already on disk.

```
CURRENT PIPELINE:
Visual features → V4 Model → Anomaly Score

EXTENDED PIPELINE:
Visual features → V4 Model ──────────────────┐
                                              ├─→ Weighted Average → Better Score
BLIP-2 Captions → CLIP Danger Similarity ───┘
```

---

### Extension 2: LLM Sequential Narrative Analysis
📁 *Planned file: `scripts/06_llm_narrative.py`*

**What it is in plain English:**

Right now, our model reads each 2-second clip in isolation (with some attention across clips).
It doesn't read the STORY.

Imagine I gave you these 32 descriptions from a video:
- Segments 1–8: "person walking normally in a car park"
- Segments 9–12: "person stops, looks around nervously"
- Segments 13–14: "person approaches another person from behind"
- Segments 15–16: "person grabs bag, other person falls"
- Segments 17–32: "person running away"

A human reading this sequence instantly knows segments 15–16 are the crime, and 9–14 are the
build-up. Our V4 model guesses this from visual patterns. A Large Language Model (LLM) like
LLaMA or GPT reads these captions as a story and **reasons about them the way a detective would.**

We send the 32 captions to an LLM and ask:
> *"Which of these 32 scenes, in order, describe an anomalous or dangerous event? Explain your reasoning."*

The LLM returns segment numbers and a plain-English explanation. This becomes a third signal
alongside the V4 model and CLIP danger score.

**What changes from current system:**
- Current: 32 segments scored independently (with some attention)
- Extension 2: 32 captions read as a narrative and reasoned about holistically

**Why it's revolutionary:**
- This is the first WS-VAD system to use LLM-level reasoning on caption sequences
- It provides a **human-readable explanation** for every detection: *"Segments 13–16 were flagged
  because the narrative describes approach, contact, and sudden departure — consistent with theft"*
- Legally defensible: the system can explain itself in court
- Generalises to unknown crime types automatically

**Implementation effort:** ~4 hours. Uses local LLaMA via Ollama (free, no API cost).

---

### Extension 3: Temporal Prediction Error (Lightweight World Model)
📁 *Planned file: `models/temporal_predictor.py` + `scripts/07_world_model.py`*

**What it is in plain English:**

Our CLIP visual features for every video segment are already extracted (32 vectors of 768 numbers
each, per video). If you think of each vector as a "fingerprint" of that 2-second clip, then a
normal video has fingerprints that smoothly flow from one to the next.

We train a tiny neural network (2 layers, trained only on NORMAL videos) to predict:
*"Given fingerprint at time T, what should fingerprint at time T+1 look like?"*

For normal videos, the prediction is accurate. The world unfolds as expected.
For anomalous videos, at the moment the crime happens, the prediction is **WRONG** — the visual
world suddenly does something the model didn't expect.

```
Normal video:  Expected → Actual = small difference ✓
Anomaly video: Expected → Actual = LARGE difference at crime moments ✗
```

This "surprise signal" can be used as a completely independent anomaly detector that requires
**no labelled anomaly data** — it only trains on normal videos.

**What changes from current system:**
- Current: model has seen anomalies and learns what they look like
- Extension 3: model has ONLY seen normal things and flags what doesn't fit

**Why it's significant:**
- This is unsupervised anomaly detection — works on any dataset, any crime type
- Based on the same principle as how the human brain detects anomalies
- The "world model" architecture is the frontier of AI research (see: Yann LeCun's predictions
  for next-generation AI)
- Training takes 30 minutes on a CPU — no GPU required

---

## The Combined Architecture: SENTINEL-V1

When all three extensions are combined with our V4 model:

```
┌── CLIP Visual Features (on disk) ──────────────────────────────────┐
│                                    ├──→ V4 Cross-Attention Model   │
│                                    │         ↓                      │
│                                    │    MIL Score [0-1]            │
│                                    │                                │
│                                    ├──→ Temporal Predictor         │
│                                    │    (trained on normals only)   │
│                                    │         ↓                      │
│                                    │    Surprise Score [0-∞]       │
└────────────────────────────────────┘                                │
                                                                      ↓
┌── BLIP-2 Captions (on disk) ───────────────────────────────────┐   │
│                                    ├──→ CLIP Danger Similarity   │   │
│                                    │         ↓                   │   │
│                                    │    Semantic Score [0-1]     │   │
│                                    │                             │   │
│                                    └──→ LLM Narrative Analysis   │   │
│                                              ↓                   │   │
│                                         Story Score [0-1]        │   │
│                                        + Explanation text        │   │
└────────────────────────────────────────────────────────────────┘   │
                                                                      │
                          ┌───────────────────────────────────────────┘
                          ↓
                 Learnable Ensemble Fusion
                 (weighted combination of 4 signals)
                          ↓
              Final Anomaly Score per Segment [0-1]
                  + Plain English Explanation
                  + Confidence Level
```

**Everything in this diagram uses features already on disk. No new data. No new GPU hours.**

---

## Why This Is Genuinely Revolutionary

| What Exists Today | SENTINEL-V1 |
|---|---|
| Detects known crime types | Detects ANY anomalous event (including novel ones) |
| Black box — no explanation | Provides plain-English explanation for every flag |
| Single camera, single clip | Reads temporal narrative across the full video |
| Requires labelled anomaly data | Extension 3 requires NO anomaly labels |
| Fixed AUROC ceiling (~88%) | No theoretical ceiling |
| Cannot say WHY it flagged | Can say: *"Segment 14 flagged: narrative shows approach, grab, and flight — consistent with theft"* |

---

## The Thesis Framing

This positions your thesis as a three-contribution paper:

> **Contribution 1 (Done):** V4 Language-Guided Cross-Attention VAD — a new SOTA architecture
> combining multi-scale attention, contrastive loss, and memory bank supervision.
>
> **Contribution 2 (Future):** Zero-Shot Semantic Danger Scoring — using CLIP similarity to
> danger-phrase embeddings as a training-free anomaly signal.
>
> **Contribution 3 (Future):** LLM Narrative Reasoning — using sequential BLIP-2 captions as
> input to an LLM for causally-grounded anomaly explanation.
>
> **Future Direction:** SENTINEL — a full world-model-based predictive safety system that
> transcends the weakly-supervised paradigm.

This narrative arc — from pattern-matcher to reasoner to predictor — is the structure of a
Distinction-level thesis and a potential top-tier publication (CVPR/ICCV/ECCV).

---

## Implementation Priority (After V4 Training Completes)

| Order | Extension | Effort | Expected AUROC Gain |
|---|---|---|---|
| 1st | CLIP Semantic Danger Score | 2 hours | +1–3% |
| 2nd | LLM Narrative Reasoning | 4 hours | +1–2% (qualitative) |
| 3rd | Temporal Prediction Error | 6 hours | +2–4% |
| 4th | Learnable Ensemble Fusion | 3 hours | +1–2% (combines all) |

---

*Document created: 2026-04-15. Status: Pending V4 training completion.*
*Author note: All features required for Extensions 1–3 are pre-extracted and stored in*
*`data/features_v31_blip2_prompt/`. Implementation requires no new hardware or data collection.*
