"""
Reason Generator Service — uses Groq LLM to produce human-readable explanations.

Both the LLM path and the offline template fallback aim for explanations that a
non-expert can act on: state the finding, say what it means in plain language,
note the practical impact on the model, and suggest a fix when one is obvious.
"""

import os
import json


def generate_reasons(scores: dict, details: dict, alignment: dict = None) -> dict:
    """
    Generate human-readable explanations for validation scores using Groq LLM.

    Args:
        scores: dict with 'geometry', 'texture', 'color' scores (0-100).
        details: dict with 'geometry', 'texture', 'color' detail dicts.
        alignment: optional camera-pose result (azimuth/elevation/iou/confidence,
            plus the 'debug' block produced by the pose estimator).

    Returns:
        dict with 'geometry_reason', 'texture_reason', 'color_reason'.
    """
    try:
        return _generate_with_groq(scores, details, alignment)
    except Exception as e:
        # Fallback: generate template-based reasons
        print(f"Groq LLM unavailable, using template fallback: {e}")
        return _generate_template_reasons(scores, details, alignment)


def _generate_with_groq(scores: dict, details: dict, alignment: dict = None) -> dict:
    """Call Groq API for LLM-based reasoning."""
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    client = Groq(api_key=api_key)

    alignment_block = ""
    if alignment:
        dbg = alignment.get("debug", {}) or {}
        alignment_block = (
            f"\n## Camera Alignment\n"
            f"- Best pose: azimuth {alignment.get('azimuth')}°, "
            f"elevation {alignment.get('elevation')}°\n"
            f"- Silhouette IoU: {alignment.get('iou')}, "
            f"confidence: {alignment.get('confidence')}\n"
            f"- Chamfer similarity: {dbg.get('chamfer_similarity')}, "
            f"scale error: {dbg.get('scale_error')}, "
            f"center error: {dbg.get('center_error')}\n"
            "All scores below were computed from this aligned viewpoint. A high "
            "IoU/chamfer with low scale & center error means the comparison is "
            "trustworthy; a low IoU means the shapes themselves differ.\n"
        )

    prompt = f"""You are an expert 3D model quality analyst writing for a non-technical user
(a designer or store owner) who needs to understand WHY their model scored the way it did
and WHAT to do about it.
{alignment_block}
## Scores (0-100, higher is better)
- Geometry: {scores['geometry']}/100
- Texture: {scores['texture']}/100
- Color: {scores['color']}/100

## Detailed Metrics
{json.dumps(details, indent=2, default=str)}

For EACH category write 2-4 sentences that:
1. State the concrete finding with real numbers (silhouette IoU; mesh holes / disconnected
   components / degenerate faces; watertight status; missing material/UV/texture image;
   SSIM; mean ΔE; dominant color shift).
2. Explain in plain language what that means (e.g. "ΔE above 10 means the difference is
   obvious to the eye"; "disconnected components are stray floating pieces of mesh";
   "SSIM near 1.0 means the texture looks almost identical").
3. Note the practical impact and, when there's an obvious one, a concrete suggestion
   ("re-export with watertight geometry", "bake a texture and UV-unwrap", "adjust the
   base color toward the source").
4. Mention any score gates/caps that were applied and why.

Avoid raw jargon without a short explanation. Be specific, accurate, and encouraging.

Respond with a JSON object containing exactly three keys:
- "geometry_reason"
- "texture_reason"
- "color_reason"
Respond ONLY with the JSON object, no markdown formatting."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a 3D model quality analysis expert who explains "
                "technical results clearly to non-experts. Respond only with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=900,
    )

    content = response.choices[0].message.content.strip()

    # Parse JSON from response
    # Handle potential markdown code fences
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]

    return json.loads(content)


# ──────────────── Plain-language helpers ────────────────


def _num(x) -> str:
    """Format an integer-ish count with thousands separators."""
    try:
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return str(x)


def _iou_phrase(iou) -> str:
    try:
        v = float(iou)
    except (TypeError, ValueError):
        return "could not be measured"
    pct = round(v * 100)
    if v >= 0.9:
        return f"an excellent {pct}% silhouette overlap"
    if v >= 0.75:
        return f"a good {pct}% silhouette overlap"
    if v >= 0.5:
        return f"a moderate {pct}% silhouette overlap"
    return f"a weak {pct}% silhouette overlap"


def _ssim_phrase(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    if v >= 0.9:
        return f"SSIM {v} — the rendered surface looks almost identical to the photo"
    if v >= 0.75:
        return f"SSIM {v} — the surface looks close to the photo with minor differences"
    if v >= 0.5:
        return f"SSIM {v} — the surface is recognizably similar but visibly different"
    return f"SSIM {v} — the surface differs clearly from the photo"


def _delta_e_phrase(de):
    try:
        v = float(de)
    except (TypeError, ValueError):
        return None, "color difference"
    if v < 1:
        return f"mean ΔE {v}", "a difference the eye essentially can't see"
    if v < 2.3:
        return f"mean ΔE {v}", "a barely perceptible difference"
    if v < 5:
        return f"mean ΔE {v}", "a slight, mostly subtle difference"
    if v < 10:
        return f"mean ΔE {v}", "a clearly noticeable difference"
    return f"mean ΔE {v}", "an obvious, large difference"


def _alignment_sentence(alignment) -> str:
    """One friendly sentence describing the viewpoint the scores were measured from."""
    if not alignment or alignment.get("fallback"):
        return (
            " Note: a matching viewpoint could not be found, so these scores compare "
            "a default front view and may be less reliable."
        )
    az = alignment.get("azimuth")
    el = alignment.get("elevation")
    return (
        f" These results were measured after rotating the model to match the photo "
        f"(azimuth {az}°, elevation {el}°), giving {_iou_phrase(alignment.get('iou'))}."
    )


# ──────────────── Template fallback ────────────────


def _generate_template_reasons(scores: dict, details: dict, alignment: dict = None) -> dict:
    """Detailed, user-friendly template reasons when the LLM is unavailable."""
    reasons = {}
    align_note = _alignment_sentence(alignment)

    # ── Geometry ──
    g_score = scores["geometry"]
    g_details = details.get("geometry", {})
    mesh_info = g_details.get("mesh_integrity", {}).get("checks", {})
    sil_info = g_details.get("silhouette_matching", {}).get("metrics", {})
    qc = g_details.get("quality_checks", {})
    gate_g = g_details.get("gating", {})

    holes = int(qc.get("holes", 0) or 0)
    comps = int(qc.get("components", 1) or 1)
    degen = int(qc.get("degenerate_faces", 0) or 0)
    watertight = mesh_info.get("is_watertight")
    iou = sil_info.get("iou", alignment.get("iou") if alignment else None)

    defects = []
    if holes > 0:
        defects.append(
            f"{_num(holes)} holes (gaps in the surface where the mesh isn't sealed)"
        )
    if comps > 1:
        defects.append(
            f"{_num(comps)} disconnected components (separate floating pieces that aren't "
            f"joined to the main body)"
        )
    if degen > 0:
        defects.append(
            f"{_num(degen)} degenerate faces (zero-area triangles that can cause rendering glitches)"
        )

    if g_score >= 80:
        g_reason = "The mesh structure is solid"
        g_reason += " and watertight (fully sealed)." if watertight else "."
        if iou is not None:
            g_reason += f" Its outline matches the photo well, with {_iou_phrase(iou)}."
    elif g_score >= 50:
        g_reason = "The geometry is acceptable but has some issues. "
        if defects:
            g_reason += "We found " + "; ".join(defects) + ". "
        if not watertight:
            g_reason += "The mesh is also not watertight (it has open edges). "
        g_reason += (
            "These won't necessarily break the model but can cause rendering artifacts "
            "or problems in 3D printing and physics."
        )
    else:
        g_reason = "We detected significant structural problems with this mesh. "
        if defects:
            g_reason += "Specifically: " + "; ".join(defects) + ". "
            g_reason += (
                "High counts like these usually mean the model was exported without "
                "cleanup — re-exporting with merged/sealed geometry should improve the score."
            )
        else:
            g_reason += "The mesh shape deviates noticeably from the source silhouette."

    if gate_g.get("gated"):
        g_reason += (
            f" Because of these defects the score was capped by structural safety limits "
            f"({'; '.join(gate_g.get('applied_gates', []))})."
        )
    g_reason += align_note
    reasons["geometry_reason"] = g_reason

    # ── Texture ──
    t_score = scores["texture"]
    t_details = details.get("texture", {})
    tp = t_details.get("texture_presence", {}).get("checks", {})
    perc = t_details.get("perceptual", {})
    pres = t_details.get("presence_checks", {})
    gate_t = t_details.get("gating", {})

    has_texture = pres.get("texture_present", tp.get("has_texture"))
    has_uv = pres.get("uv_present", tp.get("has_uv_coordinates"))
    has_material = pres.get("material_present", tp.get("has_material"))
    ssim = _ssim_phrase(perc.get("ssim_raw"))

    present = []
    missing = []
    for label, flag in (("a material", has_material), ("a texture image", has_texture),
                        ("UV coordinates", has_uv)):
        (present if flag else missing).append(label)

    if t_score >= 80:
        t_reason = "The texturing is high quality"
        t_reason += f" — {ssim}." if ssim else "."
        if present:
            t_reason += f" The model includes {', '.join(present)}."
    elif t_score >= 50:
        t_reason = "Texture quality is moderate"
        t_reason += f" — {ssim}." if ssim else "."
        if missing:
            t_reason += (
                f" It's missing {', '.join(missing)}, which limits how faithfully the "
                f"surface can match the photo."
            )
        else:
            t_reason += " Some perceptual differences from the source remain."
    else:
        t_reason = "Texture quality is poor. "
        if missing:
            t_reason += (
                f"The model is missing {', '.join(missing)}. Without these the renderer "
                f"can't reproduce the photographed surface — UV-unwrapping the mesh and "
                f"baking/assigning a texture image would fix most of this."
            )
        else:
            t_reason += "The rendered texture differs substantially from the source image."
        if ssim:
            t_reason += f" ({ssim}.)"

    if gate_t.get("gated"):
        t_reason += (
            f" The score was capped because required texture data is missing "
            f"({'; '.join(gate_t.get('applied_gates', []))})."
        )
    reasons["texture_reason"] = t_reason

    # ── Color ──
    c_score = scores["color"]
    c_details = details.get("color", {})
    de = c_details.get("delta_e", {}).get("metrics", {})
    hist = c_details.get("histogram", {}).get("metrics", {})
    de_val, de_meaning = _delta_e_phrase(de.get("mean_delta_e"))

    dom = c_details.get("dominant_color", {})
    shift = (dom or {}).get("primary_shift")
    shift_note = ""
    if shift and shift.get("delta_e", 0) >= 10:
        shift_note = (
            f" The dominant color also shifted noticeably (ΔE {shift['delta_e']}), "
            f"from RGB {shift['source_rgb']} in the photo toward {shift['render_rgb']} "
            f"in the render."
        )

    if c_score >= 80:
        c_reason = (
            f"Color reproduction is accurate ({de_val}, {de_meaning})"
        )
        corr = hist.get("avg_correlation")
        if corr is not None:
            c_reason += f", with strong histogram correlation ({corr})."
        else:
            c_reason += "."
    elif c_score >= 50:
        c_reason = (
            f"Color accuracy is moderate. The {de_val} indicates {de_meaning} between the "
            f"render and the photo — recognizable, but viewers will spot it on close comparison."
        )
    else:
        c_reason = (
            f"Color differs substantially from the source ({de_val}, {de_meaning}). "
            f"Adjusting the model's base/material colors toward the photographed colors "
            f"would close most of this gap."
        )
    c_reason += shift_note
    reasons["color_reason"] = c_reason

    return reasons
