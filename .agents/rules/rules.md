---
trigger: always_on
---

# AI Agent Instructions: MSc Thesis on Weakly Supervised Video Anomaly Detection (WS-VAD)

## 1. Project Context & Scientific Objective
- **Goal:** Develop a Distinction-level MSc thesis project focused on Weakly Supervised Video Anomaly Detection.
- **Dataset:** UCF-Crime and XD-Violence.
- **Novel Contribution:** Shifting from standard Multiple Instance Learning (MIL) and simple feature concatenation to a **Language-Guided Cross-Attention** framework.
- **Pipeline:** We are using pre-trained Vision-Language Models (BLIP-2 for captioning, CLIP for feature extraction) to extract features *offline*. The core trainable model will take pre-extracted visual and textual features (T=32 segments per video), use the text as a Query to guide the visual Keys/Values via Cross-Attention, and output an anomaly score per segment using a Top-K MIL ranking loss.

## 2. Code Quality & Style Rules (Strictly Enforced)
- **Type Hinting:** EVERY function and method must have complete Python type hints (e.g., `def load_features(path: str) -> torch.Tensor:`).
- **Docstrings:** Use Google-style docstrings for every class and function. Explain the mathematical tensor shapes in the docstrings (e.g., `Returns: torch.Tensor: Guided features of shape (Batch, 32, 512)`).
- **Modularity:** No monolithic scripts. Separate data loading, model architecture, loss functions, and training loops into their respective modules (`utils/`, `models/`, `scripts/`).
- **No Hardcoding:** Never hardcode hyperparameters (batch size, learning rate, margins, segment count $T=32$) in the Python scripts. Always load them from `configs/config.yaml`.

## 3. PyTorch & Deep Learning Best Practices
- **Reproducibility is paramount:** Provide utility functions to set seeds for `torch`, `numpy`, and `random` to ensure deterministic behavior.
- **Memory Management:** When doing offline feature extraction, strictly use `with torch.no_grad():` and `.eval()` mode. Move tensors to `.cpu()` before saving to disk to prevent GPU OOM (Out of Memory) errors.
- **Device Agnosticism:** Always dynamically check for hardware: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Never hardcode `.cuda()`.
- **Tensor Operations:** Prefer vectorized PyTorch operations over `for` loops wherever mathematically possible.

## 4. Architectural Specifics for this Project
- **T=32 Paradigm:** Videos are always divided into exactly 32 temporal segments.
- **Features:** Visual and textual features will be extracted offline and saved as `.pt` files. The PyTorch `Dataset` class should load these `.pt` files, NOT raw `.mp4` files.
- **Loss Function:** The loss will be a variation of the Top-K MIL ranking loss, incorporating temporal smoothness and sparsity constraints.

## 5. Agent Behavior
- Before writing a complex function, briefly state your mathematical or logical plan.
- If the user asks for an architecture change, consider the implications on Tensor shapes and warn the user if a dimensional mismatch is imminent.
- Write clean, PEP8-compliant code. Focus on readability and academic rigor.

## 6. Continuous Thesis Documentation Protocol (Research Scribe)
You are acting as a Principal Investigator's Research Scribe. To prepare for the final IEEE-format MSc Thesis, you must maintain a running academic log of our progress.

**Action Required:**
Create and continuously update a file named `docs/THESIS_NOTES.md`. 
Every time we complete a major module, resolve a significant architectural decision, or finish an experimental phase, you must append an entry to `docs/THESIS_NOTES.md` following this strict academic format:

### [Date] - [Task/Module Name]
*   **Target IEEE Section:** (e.g., Methodology III.B, Experimental Setup IV.A)
*   **Objective:** Briefly state the engineering or scientific goal of this step.
*   **Academic Justification:** WHY did we do this? (e.g., "Rather than using raw mp4 decoding, which is computationally bottlenecked, we extract static frames offline to ensure deterministic loading and prevent GPU memory overflow during MIL training.")
*   **Mathematical/Architectural Formulation:** Write out the relevant math using LaTeX formatting. (e.g., Define the tensor shapes, the attention mechanism $\text{Softmax}(QK^T/ \sqrt{d})V$, or the MIL Top-K ranking loss).
*   **Implementation Details:** Specifics that must go into a paper for reproducibility (e.g., "Utilized OpenCV to sample $T=32$ frames uniformly. Employed the AdamW optimizer with a learning rate of $1e-4$.").
*   **Challenges & Resolutions:** What broke? How did we fix it? (e.g., "Encountered dimensional mismatch between textual and visual streams; resolved by ensuring the projection head maps both to $\mathbb{R}^{512}$.")

**Tone & Style:** 
Use formal, objective academic language suitable for a top-tier Computer Vision conference (e.g., CVPR/ICCV). Do not use conversational filler in this document.