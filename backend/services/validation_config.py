"""
Validation configuration / tunables.

Centralizes the knobs the scoring pipeline reads so behaviour can be changed
without editing logic, and so the rationale for each default lives in one place.
All values can be overridden per-request (see ValidateRequest.asset_class) or via
environment variables for ops tuning.
"""

import os


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ── Asset class ───────────────────────────────────────────────────────────
# "product"      → multi-part / assembly meshes (furniture, appliances). These
#                  are MANY separate open-shell parts BY DESIGN, so
#                  non-watertightness, high component counts and open boundary
#                  edges are NOT defects and must not cap the geometry score.
# "single_solid" → a single watertight solid (e.g. a 3D scan / printable part),
#                  where those signals ARE meaningful defects (legacy behaviour).
DEFAULT_ASSET_CLASS = os.environ.get("ASSET_CLASS", "product")
VALID_ASSET_CLASSES = ("product", "single_solid")

# Auto-detection of assembly meshes: a model with at least this many connected
# components AND at least this many *substantial* parts is treated as a product
# assembly even if the caller did not specify an asset class.
ASSEMBLY_MIN_COMPONENTS = int(os.environ.get("ASSEMBLY_MIN_COMPONENTS", 8))
ASSEMBLY_MIN_SUBSTANTIAL_PARTS = int(os.environ.get("ASSEMBLY_MIN_SUBSTANTIAL_PARTS", 3))
# A connected component is "substantial" (a real part, not a screw/glide/sliver)
# when it has at least this many faces.
SUBSTANTIAL_PART_MIN_FACES = int(os.environ.get("SUBSTANTIAL_PART_MIN_FACES", 200))

# ── Alignment-confidence gating (Fix 4) ─────────────────────────────────────
# Per-pixel metrics (SSIM / LPIPS) only compare meaningfully when the render and
# photo silhouettes overlap tightly. Below this IoU we distrust them and fall
# back to alignment-robust foreground LAB-histogram comparison, and we LOWER the
# reported confidence.
IOU_TRUST_THRESHOLD = _env_float("IOU_TRUST_THRESHOLD", 0.85)

# ── Material sanity (Fix 3) ─────────────────────────────────────────────────
# A textured fabric/painted product reported as (near-)fully metallic is almost
# always an export bug. We surface it as a MATERIAL warning rather than letting
# the dark metallic render tank the color score.
METALLIC_WARN_THRESHOLD = _env_float("METALLIC_WARN_THRESHOLD", 0.8)
