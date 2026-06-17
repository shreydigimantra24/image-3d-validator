"""
Reason Generator Service — uses Groq LLM to produce human-readable explanations.
"""

import os
import json


def generate_reasons(scores: dict, details: dict, alignment: dict = None) -> dict:
    """
    Generate human-readable explanations for validation scores using Groq LLM.

    Args:
        scores: dict with 'geometry', 'texture', 'color' scores (0-100).
        details: dict with 'geometry', 'texture', 'color' detail dicts.
        alignment: optional camera-pose result (azimuth/elevation/iou/confidence).

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
        alignment_block = (
            f"\n## Camera Alignment\n"
            f"- Best pose: azimuth {alignment.get('azimuth')}°, "
            f"elevation {alignment.get('elevation')}°\n"
            f"- Silhouette IoU: {alignment.get('iou')}, "
            f"confidence: {alignment.get('confidence')}\n"
            "All scores below were computed from this aligned viewpoint.\n"
        )

    prompt = f"""You are an expert 3D model quality analyst. Given the following validation scores and detailed metrics,
generate concise, human-readable explanations for each score category.
{alignment_block}
## Scores
- Geometry: {scores['geometry']}/100
- Texture: {scores['texture']}/100
- Color: {scores['color']}/100

## Detailed Metrics
{json.dumps(details, indent=2, default=str)}

When relevant, cite concrete evidence: silhouette IoU, mesh holes / disconnected
components / degenerate faces, missing material/UV/texture, and any dominant
color shift (e.g. red → orange). Mention any applied score gates.

Respond with a JSON object containing exactly three keys:
- "geometry_reason": 1-2 sentence explanation of the geometry score
- "texture_reason": 1-2 sentence explanation of the texture score
- "color_reason": 1-2 sentence explanation of the color score

Be specific about what metrics contributed to the score. Mention specific issues if any.
Respond ONLY with the JSON object, no markdown formatting."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a 3D model quality analysis expert. Respond only with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )

    content = response.choices[0].message.content.strip()

    # Parse JSON from response
    # Handle potential markdown code fences
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]

    return json.loads(content)


def _generate_template_reasons(scores: dict, details: dict, alignment: dict = None) -> dict:
    """Generate template-based reasons as fallback when LLM is unavailable."""
    reasons = {}

    align_note = ""
    if alignment and not alignment.get("fallback"):
        align_note = (
            f" View aligned at azimuth {alignment.get('azimuth')}°/"
            f"elevation {alignment.get('elevation')}° (IoU {alignment.get('iou')})."
        )

    # Geometry reason
    g_score = scores["geometry"]
    g_details = details.get("geometry", {})
    mesh_info = g_details.get("mesh_integrity", {}).get("checks", {})
    sil_info = g_details.get("silhouette_matching", {}).get("metrics", {})
    qc = g_details.get("quality_checks", {})
    gate_g = g_details.get("gating", {})

    defects = []
    if qc.get("holes", 0) and qc["holes"] > 0:
        defects.append(f"{qc['holes']} holes")
    if qc.get("components", 1) > 1:
        defects.append(f"{qc['components']} disconnected components")
    if qc.get("degenerate_faces", 0):
        defects.append(f"{qc['degenerate_faces']} degenerate faces")
    defect_str = ", ".join(defects)

    if g_score >= 80:
        g_reason = "Mesh structure is solid"
        if mesh_info.get("is_watertight"):
            g_reason += ", watertight"
        g_reason += f" with good silhouette alignment (IoU: {sil_info.get('iou', 'N/A')})."
    elif g_score >= 50:
        issues = []
        if defect_str:
            issues.append(defect_str)
        if not mesh_info.get("is_watertight"):
            issues.append("mesh is not watertight")
        g_reason = f"Moderate geometry quality. Issues: {', '.join(issues) if issues else 'minor silhouette mismatch'}."
    else:
        g_reason = "Significant geometry issues detected."
        if defect_str:
            g_reason += f" Mesh contains {defect_str}."
        else:
            g_reason += " Mesh structure deviates from source silhouette."

    if gate_g.get("gated"):
        g_reason += f" Score capped by structural gates ({'; '.join(gate_g.get('applied_gates', []))})."
    g_reason += align_note

    reasons["geometry_reason"] = g_reason

    # Texture reason
    t_score = scores["texture"]
    t_details = details.get("texture", {})
    tp = t_details.get("texture_presence", {}).get("checks", {})
    perc = t_details.get("perceptual", {})
    pres = t_details.get("presence_checks", {})
    gate_t = t_details.get("gating", {})

    # Prefer the dedicated presence check (Enhancement 4) when available.
    has_texture = pres.get("texture_present", tp.get("has_texture"))
    has_uv = pres.get("uv_present", tp.get("has_uv_coordinates"))
    has_material = pres.get("material_present", tp.get("has_material"))

    if t_score >= 80:
        t_reason = f"Texture quality is high (SSIM: {perc.get('ssim_raw', 'N/A')})"
        if has_texture:
            t_reason += " with proper material and UV mapping."
        else:
            t_reason += "."
    elif t_score >= 50:
        issues = []
        if not has_texture:
            issues.append("texture image missing")
        if not has_uv:
            issues.append("UV coordinates missing")
        t_reason = f"Moderate texture quality (SSIM: {perc.get('ssim_raw', 'N/A')}). {', '.join(issues).capitalize() if issues else 'Some perceptual differences observed'}."
    else:
        missing = []
        if not has_material:
            missing.append("material")
        if not has_texture:
            missing.append("texture image")
        if not has_uv:
            missing.append("UV coordinates")
        t_reason = "Texture quality is poor."
        if missing:
            t_reason += f" Missing {', '.join(missing)}."
        else:
            t_reason += " Rendered texture differs substantially from source."

    if gate_t.get("gated"):
        t_reason += f" Presence gate applied ({'; '.join(gate_t.get('applied_gates', []))})."

    reasons["texture_reason"] = t_reason

    # Color reason
    c_score = scores["color"]
    c_details = details.get("color", {})
    de = c_details.get("delta_e", {}).get("metrics", {})
    hist = c_details.get("histogram", {}).get("metrics", {})

    dom = c_details.get("dominant_color", {})
    shift = (dom or {}).get("primary_shift")
    shift_note = ""
    if shift and shift.get("delta_e", 0) >= 10:
        shift_note = (
            f" Primary color shifted (ΔE {shift['delta_e']}) from RGB"
            f" {shift['source_rgb']} toward {shift['render_rgb']}."
        )

    if c_score >= 80:
        c_reason = f"Color reproduction is accurate (mean ΔE: {de.get('mean_delta_e', 'N/A')}) with strong histogram correlation ({hist.get('avg_correlation', 'N/A')})."
    elif c_score >= 50:
        c_reason = f"Moderate color accuracy. Mean ΔE of {de.get('mean_delta_e', 'N/A')} indicates noticeable color differences."
    else:
        c_reason = f"Significant color deviation detected (mean ΔE: {de.get('mean_delta_e', 'N/A')}). Colors differ substantially from source."

    c_reason += shift_note
    reasons["color_reason"] = c_reason

    return reasons
