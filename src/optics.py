"""Pixels to millimetres - slide 05 asks for disc and cup area in mm2.

A fundus photograph has no intrinsic scale: the same disc fills more pixels in a
longer eye, and a disc measured in pixels cannot be compared between patients.
The standard correction is Bennett-Littmann:

    q  = 0.01306 x (axial length - 1.82)      millimetres of retina per degree
    mm_per_pixel = (camera field of view / image width in pixels) x q

Two assumptions are involved, and both are reported alongside the result rather
than buried:

  * the camera's field of view. PAPILA was captured on a non-mydriatic fundus
    camera; FOV_DEGREES below is the assumed value and can be overridden per call.
  * the axial length. When it is not measured, EMMETROPIC_AXIAL_LENGTH is used and
    the output is flagged `assumed_axial_length: true`. A 2 mm error in axial
    length moves the area by roughly 18%, so an assumed value is an estimate, not
    a measurement, and the app labels it that way.

Areas scale with the square of the linear factor.
"""

FOV_DEGREES = 45.0            # assumed camera field of view
EMMETROPIC_AXIAL_LENGTH = 24.0  # mm, used when the eye was not biometry-measured
BENNETT_A, BENNETT_B = 0.01306, 1.82


def mm_per_degree(axial_length_mm):
    return BENNETT_A * (float(axial_length_mm) - BENNETT_B)


def mm_per_pixel(image_width_px, axial_length_mm=None, fov_degrees=FOV_DEGREES):
    if not image_width_px:
        return None
    al = EMMETROPIC_AXIAL_LENGTH if axial_length_mm in (None, "") else float(axial_length_mm)
    return (float(fov_degrees) / float(image_width_px)) * mm_per_degree(al)


def physical_sizes(cdr, image_width_px, axial_length_mm=None, fov_degrees=FOV_DEGREES):
    """Add mm / mm2 versions of the pixel measurements in a CDR dict."""
    scale = mm_per_pixel(image_width_px, axial_length_mm, fov_degrees)
    if not scale or not cdr:
        return None
    assumed = axial_length_mm in (None, "")
    al = EMMETROPIC_AXIAL_LENGTH if assumed else float(axial_length_mm)
    a = scale ** 2
    out = {
        "mm_per_pixel": round(scale, 6),
        "axial_length_mm": round(al, 2),
        "assumed_axial_length": bool(assumed),
        "fov_degrees": float(fov_degrees),
        "disc_diameter_mm": round((cdr.get("disc_height") or 0) * scale, 3),
        "disc_width_mm": round((cdr.get("disc_width") or 0) * scale, 3),
        "cup_diameter_mm": round((cdr.get("cup_height") or 0) * scale, 3),
        "disc_area_mm2": round((cdr.get("disc_area") or 0) * a, 3),
        "cup_area_mm2": round((cdr.get("cup_area") or 0) * a, 3),
    }
    out["rim_area_mm2"] = round(max(0.0, out["disc_area_mm2"] - out["cup_area_mm2"]), 3)
    d = out["disc_area_mm2"]
    # A disc far outside 1.2-3.8 mm2 usually means the assumptions are wrong for
    # this camera, not that the eye is unusual - say so instead of printing it flat.
    out["plausible"] = bool(1.0 <= d <= 4.5) if d else False
    out["note"] = ("Estimated from an assumed "
                   f"{fov_degrees:g} degree field of view"
                   + (" and an assumed emmetropic axial length" if assumed
                      else f" and a measured axial length of {al:.2f} mm") + ".")
    return out


__all__ = ["mm_per_pixel", "mm_per_degree", "physical_sizes", "FOV_DEGREES",
           "EMMETROPIC_AXIAL_LENGTH"]
