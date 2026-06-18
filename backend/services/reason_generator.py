"""
Reason Generator Service — uses Groq LLM to produce human-readable explanations.

Both the LLM path and the offline template fallback aim for explanations that a
non-expert can act on: state the finding, say what it means in plain language,
note the practical impact on the model, and suggest a fix when one is obvious.
"""

import os
import json


def generate_reasons(
    scores: dict,
    details: dict,
    alignment: dict = None,
    confidences: dict = None,
    material: dict = None,
) -> dict:
    """
    Generate human-readable explanations for validation scores using Groq LLM.

    Args:
        scores: dict with 'geometry', 'texture', 'color' scores (0-100).
        details: dict with 'geometry', 'texture', 'color' detail dicts.
        alignment: optional camera-pose result (azimuth/elevation/iou/confidence,
            plus the 'debug' block produced by the pose estimator).
        confidences: optional per-score confidence + alignment trust flags (Fix 4).
        material: optional material inspection result with warnings (Fix 3).

    Returns:
        dict with 'geometry_reason', 'texture_reason', 'color_reason'.
    """
    confidences = confidences or {}
    material = material or {}
    try:
        return _generate_with_groq(scores, details, alignment, confidences, material)
    except Exception as e:
        # Fallback: generate template-based reasons
        print(f"Groq LLM unavailable, using template fallback: {e}")
        return _generate_template_reasons(scores, details, alignment, confidences, material)


def _generate_with_groq(
    scores: dict, details: dict, alignment: dict = None,
    confidences: dict = None, material: dict = None,
) -> dict:
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

    asset_class = (details.get("geometry", {}) or {}).get("asset_class", "product")
    conf_block = ""
    if confidences:
        conf_block = (
            f"\n## Confidence\n"
            f"- Alignment IoU: {confidences.get('alignment_iou')} "
            f"(trusted: {confidences.get('alignment_trusted')})\n"
            f"- Per-score confidence — geometry {confidences.get('geometry')}, "
            f"texture {confidences.get('texture')}, color {confidences.get('color')}\n"
            "If alignment is NOT trusted, say plainly that per-pixel texture metrics "
            "(SSIM/LPIPS) were down-weighted and an alignment-robust comparison was used, "
            "so confidence is reduced.\n"
        )
    material_block = ""
    if material:
        warns = material.get("warnings") or []
        material_block = (
            f"\n## Material\n"
            f"- metallicFactor: {material.get('metallic_factor')}, "
            f"roughnessFactor: {material.get('roughness_factor')}, "
            f"has baseColor texture: {material.get('has_base_color_texture')}\n"
            + ("- WARNINGS: " + "; ".join(warns) + "\n" if warns else "")
        )

    guidance = (
        f"\n## IMPORTANT scoring rules ({asset_class} asset)\n"
        "- This is a multi-part PRODUCT/ASSEMBLY mesh. Non-watertightness, large "
        "connected-component counts, and open boundary edges are BY DESIGN and are "
        "NOT defects — do NOT describe them as problems or cite their counts as faults.\n"
        "- Geometry is judged on structural soundness (NaN/inf, normals, degenerate "
        "faces, true floaters) plus how well the rendered silhouette matched the photo.\n"
        "- Color is judged on the asset's baseColor ALBEDO texture vs the photo, not on "
        "an unlit render. If metallicFactor is high, explain that the albedo may MATCH "
        "the photo while the model still LOOKS dark under lighting (a material issue, not "
        "a color mismatch). Do not assert a large color mismatch when albedo matches.\n"
    ) if asset_class == "product" else ""

    prompt = f"""You are an expert 3D model quality analyst writing for a non-technical user
(a designer or store owner) who needs to understand WHY their model scored the way it did
and WHAT to do about it.
{alignment_block}{conf_block}{material_block}{guidance}
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
5. NEVER use an adjective that contradicts its number. Know each metric's scale:
   histogram correlation and SSIM are 0–1 (1 = identical, so 0.16 is WEAK, not
   strong); ΔE2000 is a distance (0 = identical, lower is better). The texture
   surface-appearance metrics are computed on the SAME exposure-normalized image
   as color, so do not claim the surface "looks different" while color "matches".
6. TEXTURE: if the material, texture image and UVs are present/valid, do NOT call
   the texture a defect. Cite texture presence, its resolution, and valid UVs;
   when alignment IoU is low, say per-pixel SSIM/LPIPS is limited by alignment and
   report REDUCED CONFIDENCE — do not say "visibly different". The texture verdict
   must agree with color that the surface/albedo is essentially correct.

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


def _corr_phrase(corr):
    """
    Map an OpenCV HISTCMP_CORREL value (1 = identical distributions, 0 = none,
    negative = anti-correlated) to a descriptor. The adjective is DERIVED from the
    value via these thresholds so the word can never contradict the number (Fix A).
    """
    try:
        v = float(corr)
    except (TypeError, ValueError):
        return None
    if v >= 0.8:
        word = "strong"
    elif v >= 0.6:
        word = "moderate"
    elif v >= 0.4:
        word = "fair"
    elif v >= 0.2:
        word = "weak"
    else:
        word = "very weak"
    return f"{word} foreground histogram correlation ({round(v, 4)}, where 1.0 = identical)"


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


def _generate_template_reasons(
    scores: dict, details: dict, alignment: dict = None,
    confidences: dict = None, material: dict = None,
) -> dict:
    """Detailed, user-friendly template reasons when the LLM is unavailable.

    Every figure below is pulled from its OWN metric variable (Fix 2): `holes`
    from quality_checks.holes, `comps` from quality_checks.components, etc. — they
    are never interchanged.
    """
    reasons = {}
    confidences = confidences or {}
    material = material or {}
    align_note = _alignment_sentence(alignment)

    # ── Geometry (asset-class aware — Fix 1) ──
    g_score = scores["geometry"]
    g_details = details.get("geometry", {})
    asset_class = g_details.get("asset_class", "product")
    sil_info = g_details.get("silhouette_matching", {}).get("metrics", {})
    qc = g_details.get("quality_checks", {})
    gate_g = g_details.get("gating", {})

    # Each number from its own variable — no swapping (Fix 2).
    comps = int(qc.get("components", 1) or 1)
    substantial = int(qc.get("substantial_components", 0) or 0)
    degen = int(qc.get("degenerate_faces", 0) or 0)
    floaters = int(qc.get("far_floaters", 0) or 0)
    slivers = int(qc.get("isolated_slivers", 0) or 0)
    normals_ok = qc.get("normals_consistent", True)
    iou = sil_info.get("iou", alignment.get("iou") if alignment else None)

    if asset_class == "product":
        # Genuine defects ONLY — never holes/components/watertightness.
        defects = []
        if not normals_ok:
            defects.append("inconsistent face normals (some faces wound inside-out)")
        if floaters > 0:
            defects.append(f"{_num(floaters)} detached part(s) floating far from the body")
        if slivers > 0:
            defects.append(f"{_num(slivers)} stray sliver triangle(s)")
        if degen > 0:
            defects.append(f"{_num(degen)} degenerate (zero-area) face(s)")

        if g_score >= 80:
            g_reason = (
                f"The mesh is structurally sound. It is a multi-part assembly "
                f"({_num(comps)} separate pieces, {_num(substantial)} of them substantial "
                f"parts), which is expected for this kind of product — the open shells and "
                f"non-watertight surfaces are by design, not defects."
            )
        elif g_score >= 50:
            g_reason = "The mesh is mostly sound, with a few genuine issues. "
            if defects:
                g_reason += "We found " + "; ".join(defects) + ". "
            g_reason += "Its many open-shell parts are normal for an assembly and were not penalised."
        else:
            g_reason = "We detected genuine structural problems. "
            if defects:
                g_reason += "Specifically: " + "; ".join(defects) + ". "
            else:
                g_reason += "The rendered shape deviated noticeably from the photo silhouette."
    else:
        # single_solid: legacy topology-aware messaging.
        mesh_info = g_details.get("mesh_integrity", {}).get("checks", {})
        watertight = mesh_info.get("is_watertight")
        holes = int(qc.get("holes", 0) or 0)
        defects = []
        if holes > 0:
            defects.append(f"{_num(holes)} open boundary loops (unsealed holes)")
        if comps > 1:
            defects.append(f"{_num(comps)} disconnected components")
        if degen > 0:
            defects.append(f"{_num(degen)} degenerate faces")
        if g_score >= 80:
            g_reason = "The mesh structure is solid"
            g_reason += " and watertight." if watertight else "."
            if iou is not None:
                g_reason += f" Its outline matches the photo, with {_iou_phrase(iou)}."
        elif g_score >= 50:
            g_reason = "The geometry is acceptable but has issues. "
            if defects:
                g_reason += "We found " + "; ".join(defects) + ". "
            if not watertight:
                g_reason += "The mesh is not watertight. "
        else:
            g_reason = "We detected significant structural problems. "
            if defects:
                g_reason += "Specifically: " + "; ".join(defects) + ". "

    if gate_g.get("gated"):
        g_reason += (
            f" The score was capped by structural safety limits "
            f"({'; '.join(gate_g.get('applied_gates', []))})."
        )
    g_reason += align_note
    reasons["geometry_reason"] = g_reason

    # ── Texture ──
    #
    # Texture similarity is measured on the SAME non-dark image basis as color
    # (color_validator.prepare_comparison). When the data is present/valid, a low
    # per-pixel SSIM/LPIPS is attributed to weak alignment or the metallic/lighting
    # appearance — NOT a texture defect — so this panel agrees with color.
    t_score = scores["texture"]
    t_details = details.get("texture", {})
    tp = t_details.get("texture_presence", {}).get("checks", {})
    perc = t_details.get("perceptual", {})
    pres = t_details.get("presence_checks", {})
    gate_t = t_details.get("gating", {})

    has_texture = pres.get("texture_present", tp.get("has_texture"))
    has_uv = pres.get("uv_present", tp.get("has_uv_coordinates"))
    has_material = pres.get("material_present", tp.get("has_material"))
    resolution = pres.get("texture_resolution")
    trusted = perc.get("trusted")
    t_conf = confidences.get("texture")
    mat_warnings = material.get("warnings") or []

    missing = []
    for label, flag in (("a material", has_material), ("a texture image", has_texture),
                        ("UV coordinates", has_uv)):
        if not flag:
            missing.append(label)

    # Describe the texture data that IS present (presence + resolution + UVs).
    if resolution and len(resolution) == 2:
        tex_desc = f"a {int(resolution[0])}×{int(resolution[1])} baseColor texture"
    else:
        tex_desc = "a baseColor texture"
    present_facts = []
    if has_material:
        present_facts.append("a material")
    if has_texture:
        present_facts.append(tex_desc)
    if has_uv:
        present_facts.append("valid UV coordinates")
    facts = ", ".join(present_facts) if present_facts else "no texture data"

    if missing or gate_t.get("gated"):
        # GENUINE defect: required texture data is actually missing.
        t_reason = "Texture quality is limited by missing data. "
        if missing:
            t_reason += (
                f"The model is missing {', '.join(missing)}. Without these the renderer "
                f"can't reproduce the photographed surface — UV-unwrap the mesh and "
                f"bake/assign a texture image to fix most of this."
            )
        if gate_t.get("gated"):
            t_reason += (
                f" The score was capped because required texture data is missing "
                f"({'; '.join(gate_t.get('applied_gates', []))})."
            )
    elif trusted is False:
        # Data is complete; the only limit is weak alignment / lighting. NOT a defect.
        t_reason = (
            f"The texture data is complete and valid — the model has {facts}. "
            f"Its surface color matches the photo (see the color analysis). "
            f"Per-pixel similarity (SSIM/LPIPS) could only be measured at "
            f"{_iou_phrase(perc.get('alignment_iou'))}, which limits those metrics, so the "
            f"texture result is reported at reduced confidence ({t_conf}) rather than as a "
            f"texture defect."
        )
        if mat_warnings:
            t_reason += (
                " The darker rendered appearance comes from the material's metallic setting, "
                "not the texture itself."
            )
    else:
        # Trusted alignment: per-pixel similarity is meaningful here.
        ssim = _ssim_phrase(perc.get("ssim_raw"))
        if t_score >= 80:
            t_reason = "The texturing is high quality"
            t_reason += f" — {ssim}." if ssim else "."
            t_reason += f" The model includes {facts}."
        else:
            t_reason = f"The model has {facts}"
            t_reason += f", and {ssim}." if ssim else "."
            t_reason += (
                " With alignment trustworthy, the remaining per-pixel difference reflects a "
                "genuine surface/detail gap."
            )
        if t_conf is not None:
            t_reason += f" (Confidence {t_conf}.)"

    reasons["texture_reason"] = t_reason

    # ── Color ──
    c_score = scores["color"]
    c_details = details.get("color", {})
    de = c_details.get("delta_e", {}).get("metrics", {})
    hist = c_details.get("histogram", {}).get("metrics", {})
    de_val, de_meaning = _delta_e_phrase(de.get("mean_delta_e"))

    reference = c_details.get("reference", "")
    on_albedo = reference == "albedo_texture"
    ref_word = "albedo texture" if on_albedo else "lighting-normalized render"

    dom = c_details.get("dominant_color", {})
    shift = (dom or {}).get("primary_shift")
    shift_note = ""
    if shift and shift.get("delta_e", 0) >= 10:
        shift_note = (
            f" The dominant color also shifts (ΔE2000 {shift['delta_e']}), "
            f"from RGB {shift['source_rgb']} in the photo toward {shift['render_rgb']} "
            f"in the {'albedo' if on_albedo else 'render'}."
        )

    if c_score >= 80:
        c_reason = (
            f"Color is accurate: comparing the asset's {ref_word} against the photo "
            f"foreground gives {de_val} (CIEDE2000), {de_meaning}."
        )
        # Histogram correlation is a secondary distribution cue (the ΔE2000 above
        # is the verdict). Only mention it when it actually corroborates — never
        # headline a weak value next to an "accurate" finding.
        corr = hist.get("avg_correlation")
        if corr is not None and float(corr) >= 0.4:
            c_reason += f" This is corroborated by {_corr_phrase(corr)}."
    elif c_score >= 50:
        c_reason = (
            f"Color accuracy is moderate. Comparing the {ref_word} with the photo gives "
            f"{de_val} (CIEDE2000) — {de_meaning}, recognizable but visible on close comparison."
        )
    else:
        c_reason = (
            f"Color differs substantially from the photo ({de_val}, CIEDE2000 — {de_meaning}). "
            f"Adjusting the base/albedo colors toward the photographed colors would close the gap."
        )
    c_reason += shift_note

    # Material caveat (Fix 3): albedo can MATCH while the lit render looks dark.
    mat_warnings = material.get("warnings") or []
    if mat_warnings:
        c_reason += (
            f" Material note: {mat_warnings[0]} The albedo color above is the asset's true "
            f"surface color; the dark look under lighting is a material setting, not a color "
            f"mismatch — set metallicFactor toward 0 (or add an environment map) to fix the look."
        )
    reasons["color_reason"] = c_reason

    return reasons
