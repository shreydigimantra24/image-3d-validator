"""
Reason Generator Service — uses Groq LLM to produce human-readable explanations.
"""

import os
import json


def generate_reasons(scores: dict, details: dict) -> dict:
    """
    Generate human-readable explanations for validation scores using Groq LLM.

    Args:
        scores: dict with 'geometry', 'texture', 'color' scores (0-100).
        details: dict with 'geometry', 'texture', 'color' detail dicts.

    Returns:
        dict with 'geometry_reason', 'texture_reason', 'color_reason'.
    """
    try:
        return _generate_with_groq(scores, details)
    except Exception as e:
        # Fallback: generate template-based reasons
        print(f"Groq LLM unavailable, using template fallback: {e}")
        return _generate_template_reasons(scores, details)


def _generate_with_groq(scores: dict, details: dict) -> dict:
    """Call Groq API for LLM-based reasoning."""
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    client = Groq(api_key=api_key)

    prompt = f"""You are an expert 3D model quality analyst. Given the following validation scores and detailed metrics, 
generate concise, human-readable explanations for each score category.

## Scores
- Geometry: {scores['geometry']}/100
- Texture: {scores['texture']}/100
- Color: {scores['color']}/100

## Detailed Metrics
{json.dumps(details, indent=2, default=str)}

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


def _generate_template_reasons(scores: dict, details: dict) -> dict:
    """Generate template-based reasons as fallback when LLM is unavailable."""
    reasons = {}

    # Geometry reason
    g_score = scores["geometry"]
    g_details = details.get("geometry", {})
    mesh_info = g_details.get("mesh_integrity", {}).get("checks", {})
    sil_info = g_details.get("silhouette_matching", {}).get("metrics", {})

    if g_score >= 80:
        g_reason = "Mesh structure is solid"
        if mesh_info.get("is_watertight"):
            g_reason += ", watertight"
        g_reason += f" with good silhouette alignment (IoU: {sil_info.get('iou', 'N/A')})."
    elif g_score >= 50:
        issues = []
        if not mesh_info.get("is_watertight"):
            issues.append("mesh is not watertight")
        if mesh_info.get("has_floating_components"):
            issues.append(f"{mesh_info.get('num_components', 0)} disconnected components detected")
        g_reason = f"Moderate geometry quality. Issues: {', '.join(issues) if issues else 'minor silhouette mismatch'}."
    else:
        g_reason = "Significant geometry issues detected. Mesh structure needs improvement and silhouette deviates from source."

    reasons["geometry_reason"] = g_reason

    # Texture reason
    t_score = scores["texture"]
    t_details = details.get("texture", {})
    tp = t_details.get("texture_presence", {}).get("checks", {})
    perc = t_details.get("perceptual", {})

    if t_score >= 80:
        t_reason = f"Texture quality is high (SSIM: {perc.get('ssim_raw', 'N/A')})"
        if tp.get("has_texture"):
            t_reason += " with proper material and UV mapping."
        else:
            t_reason += "."
    elif t_score >= 50:
        issues = []
        if not tp.get("has_texture"):
            issues.append("texture image missing")
        if not tp.get("has_uv_coordinates"):
            issues.append("UV coordinates missing")
        t_reason = f"Moderate texture quality (SSIM: {perc.get('ssim_raw', 'N/A')}). {', '.join(issues).capitalize() if issues else 'Some perceptual differences observed'}."
    else:
        t_reason = "Texture quality is poor. Missing or damaged textures significantly impact visual fidelity."

    reasons["texture_reason"] = t_reason

    # Color reason
    c_score = scores["color"]
    c_details = details.get("color", {})
    de = c_details.get("delta_e", {}).get("metrics", {})
    hist = c_details.get("histogram", {}).get("metrics", {})

    if c_score >= 80:
        c_reason = f"Color reproduction is accurate (mean ΔE: {de.get('mean_delta_e', 'N/A')}) with strong histogram correlation ({hist.get('avg_correlation', 'N/A')})."
    elif c_score >= 50:
        c_reason = f"Moderate color accuracy. Mean ΔE of {de.get('mean_delta_e', 'N/A')} indicates noticeable color differences."
    else:
        c_reason = f"Significant color deviation detected (mean ΔE: {de.get('mean_delta_e', 'N/A')}). Colors differ substantially from source."

    reasons["color_reason"] = c_reason

    return reasons
