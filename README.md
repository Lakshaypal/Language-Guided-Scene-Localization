# Language-Guided Scene Localization

An interactive language-guided image-localization prototype combining GroundingDINO candidate detection, CLIP ranking and SAM segmentation. It returns up to three matching regions and supports an optional negative prompt. Quantitative accuracy has not yet been established on a labelled benchmark.

**🎬 Drive Link: [Demo Vedio](https://drive.google.com/file/d/1hXu-7xfE6Awf8NKWoz-nhikCckRx--Yw/view?usp=share_link)**

---

## ✨ Core Features

- 📌 **State-of-the-Art GroundedSAM Pipeline**: Produces precise, background-removed segmentation masks instead of simple bounding boxes for superior accuracy and cleaner visualizations.
- 🔓 **Open-Vocabulary Detection**: Localizes any object, attribute, or action described by free-form text, not just pre-defined classes.
- 🧠 **Advanced Semantic Refinement**: Incorporates a suite of custom modules for intelligent filtering and ranking:
    - 🥇 **Top-3 Ranked Results**: Acknowledges ambiguity in dense scenes by providing the top three best-matching candidates.
    - 🚫 **Negative Prompting (Contrastive Filtering)**: Allows the user to provide negative queries (e.g., "white shirt") to actively disqualify irrelevant results, significantly improving search precision.
    - 🛡️ **Multi-Stage Quality Gating**: Employs a series of confidence and heuristic checks to filter out low-quality or nonsensical "hallucinated" results.
- 📦 **Checkpoint-based setup**: Uses pretrained checkpoints managed through Git LFS, plus an external GroundingDINO installation and a CLIP model download.
- 🖥️ **Interactive Web UI**: A user-friendly interface built with Gradio for easy demonstration and testing, complete with status feedback for a professional user experience.

---

## 🏗️ Technical Architecture

The system is built on a state-of-the-art **GroundedSAM** pipeline, which intelligently combines three powerful pre-trained models in a multi-stage process. The architecture is heavily influenced by the principle of **contrastive learning**, applying it at inference time to refine results.

1. 🐤 **Candidate Proposal (GroundingDINO):** The input image and text prompt are fed into a **GroundingDINO** model to identify potential bounding boxes. A Non-Maximum Suppression (NMS) step with an IoU threshold of 0.7 is applied to remove redundant, highly overlapping boxes.

2. 🔍 **Semantic Disqualification (CLIP):** If a negative prompt is provided, this custom filter is activated. This module is a practical, inference-time application of the same principle behind training techniques like **Triplet Loss**. For each candidate, its semantic similarity to both the positive and negative prompts is calculated using **CLIP**. A candidate is **completely disqualified** if it matches the negative prompt more strongly than the positive one.

3. ⚖️ **Heuristic Re-ranking (DINO + CLIP):** All surviving candidates are ranked based on a `combined_score` (`0.4 * dino_confidence + 0.6 * clip_score`), balancing raw detection confidence with nuanced semantic understanding.

4. 🛟 **Quality Gating & Final Selection:** The system iterates through the ranked list and applies two final gatekeeper filters:
    - ✅ **Confidence Gate:** A result is only accepted if its `combined_score` is above a minimum threshold (`0.25`) OR its raw `dino_confidence` is exceptionally high (`0.90`). This dual-condition check prevents low-quality guesses while preserving obvious, high-confidence detections.
    - 🧩 **Mask Density Gate:** The candidate box is passed to SAM. If the resulting mask is too sparse (less than 5% of its bounding box area), it is discarded as "pixel dust."
    
    The system collects the **first three candidates** that successfully pass all filters.

5. ✂️ **Precise Segmentation (SAM):** For the final, validated candidates, the **Segment Anything Model (SAM)** is used to generate pixel-level, background-removed segmentation masks.

---

## 🛠️ Setup Instructions

This repository uses Git LFS to manage large model weights.

**1. Install Git LFS:**
Before cloning, ensure you have Git LFS installed. You can download it from [git-lfs.github.com](https://git-lfs.github.com).
```bash
# On macOS with Homebrew
brew install git-lfs
```

**2. Clone the repository:**
When you clone the repository, Git LFS will automatically download the large model weight files, ensuring all necessary components are present.
```bash
git clone https://github.com/Lakshaypal/language-guided-scene-localization.git
cd language-guided-scene-localization
git lfs pull
```
*(Note: If the weights do not download automatically, run `git lfs pull` inside the repository.)*

**3. Create and activate a conda environment:**
A Python 3.10 environment is recommended.
```bash
conda create --name scene_loc python=3.10
conda activate scene_loc
```

**4. Install dependencies and GroundingDINO:**
Install the Python dependencies and the external GroundingDINO implementation. Its config must be available under `GroundingDINO/groundingdino/config/`.
```bash
python -m pip install -r requirements.txt
git clone https://github.com/IDEA-Research/GroundingDINO.git
python -m pip install -e GroundingDINO
```

---

## ▶️ How to Run the Prototype

Launch the interactive Gradio application with the following command from the project's root directory:

```bash
python src/app.py
```
Verify that `weights/groundingdino_swint_ogc.pth` and `weights/sam_vit_b_01ec64.pth` contain the actual checkpoints rather than Git LFS pointer text. CLIP is fetched from Hugging Face on first use. The code selects Apple MPS when available and otherwise CPU.

Then, open your web browser and navigate to the local URL provided (e.g., `http://127.0.0.1:7860`).

---

## 🧭 Development Journey: Blockers and Solutions

This project involved an iterative development process where several challenges were identified and overcome through systematic tuning and architectural improvements.

- 🔮 **Blocker: Model "Hallucinations" on Non-Existent Objects.**
    - 🧩 **Problem:** Initial versions of the pipeline would find low-confidence, nonsensical matches for queries like "a purple elephant" instead of returning nothing.
    - 🛠️ **Solution:** A **Confidence Gate** was implemented. This dual-threshold check (`MIN_FINAL_SCORE` and `HIGH_DINO_CONFIDENCE_THRESHOLD`) acts as a quality filter, rejecting results that are not sufficiently confident and ensuring the system fails gracefully.

- 🧼 **Blocker: Imprecise or "Junk" Segmentations.**
    - 🧾 **Problem:** When SAM was given a vague bounding box, it sometimes produced fragmented, "pixel dust" masks that were not useful.
    - ✅ **Solution:** A **Mask Density Filter** was added. This custom module calculates the ratio of object pixels to the total bounding box area and discards any mask that is too sparse, ensuring only solid, object-like segmentations are shown.

- ⚖️ **Blocker: Ambiguous Results and Poor Precision.**
    - 🔍 **Problem:** The model struggled to differentiate between visually similar concepts, such as a "light blue shirt" versus a "white shirt" in tricky lighting.
    - 🚫 **Solution:** The **Negative Prompting** feature was designed and implemented. Initial attempts using score manipulation (subtraction/multiplication) proved unstable. The final, robust solution was a **disqualification filter**, which completely removes a candidate if it matches the negative query more strongly than the positive one. This provides a powerful and intuitive way for the user to refine search results.

- 🔁 **Blocker: Redundant Detections.**
    - ⚠️ **Problem:** The model would sometimes detect the same object multiple times with slightly different bounding boxes.
    - 🎯 **Solution:** The **NMS (Non-Maximum Suppression) threshold** was tuned from a permissive `0.8` to a more aggressive `0.7`, effectively merging these duplicate detections and ensuring the Top-3 results are visually distinct.

---

## 🚀 Future Aspects & Unimplemented Ideas

While the current prototype is robust and feature-rich, the following advanced phases were planned and explored but not implemented due to the project's timeframe.

- 📊 **Quantitative Evaluation (mAP):** The logical next step is to establish a rigorous, quantitative benchmark for the system. This would involve creating a manually-labeled ground-truth test set and implementing an evaluation script to calculate the **Mean Average Precision (mAP)** score. This would allow for objective measurement of any future model improvements.

- ⚙️ **Model Fine-Tuning (Triplet Loss):** To address the model's core limitations in differentiating nuanced concepts (like colors), a full fine-tuning phase was planned. This would involve using the massive **Visual Genome** dataset to re-train the final layers of the CLIP vision encoder. The training would leverage a **Triplet Loss** function to explicitly teach the model a better semantic understanding of similarity and difference. This advanced step requires significant cloud compute resources (GPU/TPU) and a dedicated data processing pipeline.
