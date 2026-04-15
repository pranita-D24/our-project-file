# SAM2 and CLIP Integration Plan

This plan outlines the creation of two new core modules (`segmentor.py` and `semantic_diff.py`) incorporating SAM2 for segmenting engineering drawings and CLIP for semantic patch comparisons, fulfilling Layers 4 and 5 of your requested architecture.

## Proposed Changes

---

### Machine Learning Integrations

#### [NEW] [segmentor.py](file:///c:/Trivim%20Internship/engineering_comparison_system/segmentor.py)
Creates the `SAM2Segmentor` module. 
- Uses Meta's `SAM2AutomaticMaskGenerator` to segment all components on the engineering drawing.
- **Methods:** 
  - `__init__(model_type, checkpoint_path)`: Loads SAM2 weights onto the best available device (`cuda`, `mps`, or `cpu`).
  - `generate_masks(image)`: Infers all individual components across the drawing.
  - `filter_masks(masks)`: Excludes the whole background and microscopic noise components, leaving only the primary structural items.

#### [NEW] [semantic_diff.py](file:///c:/Trivim%20Internship/engineering_comparison_system/semantic_diff.py)
Creates the `CLIPSemanticAnalyzer` module.
- Uses HuggingFace's `transformers` to load `openai/clip-vit-base-patch32` (or similar).
- **Methods:**
  - `__init__(model_name)`: Loads CLIP Vision Model and Processor.
  - `get_embedding(image_patch)`: Extracts the dense semantic vector from an image region.
  - `compare_patches(patch1, patch2)`: Calculates cosine similarity between embeddings to detect semantic differences (shading, rotation) independent of raw pixel differences.

---

### Dependency Management

#### [MODIFY] [requirements.txt](file:///c:/Trivim%20Internship/engineering_comparison_system/requirements.txt)
To support these models locally, your list of dependencies must be expanded. I will add:
- `torch`, `torchvision` (Base PyTorch bindings)
- `transformers` (HuggingFace, for CLIP)
- Note: Meta's SAM2 usually requires manual installation (`pip install git+https://github.com/facebookresearch/sam2.git`).

---

## User Review Required

> [!WARNING]
> SAM2 and CLIP rely heavily on PyTorch.
> 1. **Hardware:** Do you have a dedicated NVIDIA GPU (CUDA) available locally to hit your target speeds of "1-2s per image", or will this run on a CPU?
> 2. **SAM2 Checkpoints:** SAM2 requires downloading a weights file (such as `sam2_hiera_tiny.pt` or `sam2_hiera_large.pt`). Do you already have the weights downloaded, or should I write an automatic download function inside `segmentor.py`?

## Verification Plan

### Automated Tests
1. Generate dummy patches or load simple engineering drawing fragments using `cv2.imread`.
2. Run `segment_components()` locally to confirm SAM2 successfully returns a list of dictionaries with valid bounding boxes and masks.
3. Run `compare_patches()` through CLIP to ensure cosine similarity correctly distinguishes between identical views and completely mismatched components.
