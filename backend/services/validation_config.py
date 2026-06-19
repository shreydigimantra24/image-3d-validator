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

# ── Alignment-confidence gating (Fix 4 / Fix 6) ─────────────────────────────
# Per-pixel metrics only compare meaningfully when the silhouettes overlap
# enough. Below this IoU we distrust them and fall back to the alignment-robust
# foreground LAB-histogram comparison, at reduced confidence.
#
# The bar used to be 0.85 — the value tight per-pixel SSIM/LPIPS over the FULL
# frame needs. But (a) real multi-part products (table + chairs) rarely exceed
# ~0.5-0.6 IoU even when correct, and (b) the per-pixel metrics are now computed
# FOREGROUND-MASKED (Fix 6), so they no longer need near-perfect global overlap
# to be meaningful. We therefore trust them from a furniture-realistic 0.5, and
# only drop to the histogram fallback when overlap is genuinely poor.
IOU_TRUST_THRESHOLD = _env_float("IOU_TRUST_THRESHOLD", 0.5)

# ── Material sanity (Fix 3) ─────────────────────────────────────────────────
# A textured fabric/painted product reported as (near-)fully metallic is almost
# always an export bug. We surface it as a MATERIAL warning rather than letting
# the dark metallic render tank the color score.
METALLIC_WARN_THRESHOLD = _env_float("METALLIC_WARN_THRESHOLD", 0.8)

# ── Object-match scaling (Fix 5) ────────────────────────────────────────────
# The pose search rotates the model to the photo and reports the BEST achievable
# silhouette IoU. A genuinely matching model+photo overlaps well (IoU climbs to
# IOU_TRUST_THRESHOLD); a low best IoU means the rendered SHAPE never matches the
# photo at any pose — i.e. likely the WRONG model for this image.
#
# Rather than CLAMP every category to one hardcoded constant (which makes the
# three scores collapse to an identical, arbitrary-looking number), we SCALE each
# category's own score by a continuous "object-match factor" derived from IoU:
#
#     factor = clamp(iou / MATCH_FULL_IOU, 0, 1)
#
# IMPORTANT calibration: a GENUINELY matching multi-part product (a table + 4
# chairs photo vs its own model) only reaches a MODERATE silhouette IoU — roughly
# 0.5-0.6 — because thin legs, gaps between chairs, and small pose-registration
# errors prevent the spindly silhouettes from ever overlapping like a solid blob.
# Anchoring "full trust" at IOU_TRUST_THRESHOLD (0.85, the per-pixel SSIM/LPIPS
# bar) therefore over-penalised correct matches. We instead anchor at
# MATCH_FULL_IOU — the IoU a real good match actually achieves — so a correct
# model is left essentially unscaled, while a clear mismatch (which sits below the
# realistic match band) is pulled down.
#
# MISMATCH_IOU_HARD only controls the WARNING banner: below it we additionally
# surface a "probable wrong model" notice. It does not introduce a flat cap.
MISMATCH_IOU_HARD = _env_float("MISMATCH_IOU_HARD", 0.45)
MATCH_FULL_IOU = _env_float("MATCH_FULL_IOU", 0.55)


def object_match_factor(iou) -> float:
    """Continuous 0-1 trust multiplier from best-aligned silhouette IoU.

    1.0 at/above MATCH_FULL_IOU (the IoU a genuine good match realistically
    reaches), linearly to 0.0 at IoU 0. None → 1.0 (legacy callers that don't
    supply alignment are left unscaled).
    """
    if iou is None:
        return 1.0
    try:
        v = float(iou)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, v / MATCH_FULL_IOU))
