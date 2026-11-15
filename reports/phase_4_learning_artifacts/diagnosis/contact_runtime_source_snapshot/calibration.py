"""Fit a declared depth variance model to paired, registered reference depths."""
from .sensors import _numpy
try:
    import numpy as np
except ImportError:
    np = None


def fit_depth_noise(measured, reference, materials, *, source="synthetic"):
    """Fit var(error|z) = base² + quadratic²*z⁴; fail on insufficient coverage.

    NaN measured returns count toward holes, never residuals. Reference must be
    registered independently; this estimator cannot distinguish misregistration,
    reference error, multipath, or scene motion from sensor error.
    """
    _numpy()
    if source not in ("synthetic", "real"):
        raise ValueError("source must be 'synthetic' or 'real'")
    measured, reference = np.asarray(measured, dtype=float), np.asarray(reference, dtype=float)
    materials = np.asarray(materials)
    if measured.shape != reference.shape or materials.shape != reference.shape or measured.ndim not in (1, 2, 3):
        raise ValueError("measured/reference/materials must have matching 1D, 2D, or 3D shapes")
    if not np.all(np.isin(materials, [-1, 0, 1, 2, 3, 4])):
        raise ValueError("material IDs must be -1 or 0..4")
    eligible = np.isfinite(reference) & (reference > 0) & (materials >= 0)
    returned = eligible & np.isfinite(measured) & (measured > 0)
    z = reference[returned]
    error = measured[returned] - z
    if len(z) < 200 or np.ptp(z) < .5 or np.max(z) < 1.5 * np.min(z):
        raise ValueError("insufficient distance coverage: need 200 returns and >=0.5 m span with max/min >=1.5")
    boundaries = np.linspace(float(np.min(z)), float(np.max(z)), 13)
    bin_index = np.minimum(np.searchsorted(boundaries, z, side="right") - 1, 11)
    bins, rows, variances, weights = [], [], [], []
    for index in range(12):
        selected = bin_index == index
        count = int(selected.sum())
        if count < 20:
            continue
        e, distance = error[selected], z[selected]
        variance = float(np.var(e, ddof=1))
        bins.append({"lower_m": float(boundaries[index]), "upper_m": float(boundaries[index + 1]),
                     "count": count, "mean_depth_m": float(np.mean(distance)),
                     "bias_m": float(np.mean(e)), "std_m": float(np.sqrt(variance))})
        rows.append([1., float(np.mean(distance ** 4))])
        variances.append(variance)
        weights.append(count - 1)
    if len(bins) < 4:
        raise ValueError("insufficient populated distance bins: need four bins of >=20 returns")
    x, y, weight = np.asarray(rows), np.asarray(variances), np.sqrt(weights)
    a, b = x * weight[:, None], y * weight
    solution = np.linalg.lstsq(a, b, rcond=None)[0]
    # Exact two-variable nonnegative least squares, including boundary minima.
    candidates = [np.zeros(2), np.array([max(0., np.dot(a[:, 0], b) / np.dot(a[:, 0], a[:, 0])), 0.]),
                  np.array([0., max(0., np.dot(a[:, 1], b) / np.dot(a[:, 1], a[:, 1]))])]
    if np.all(solution >= 0):
        candidates.append(solution)
    coefficients = min(candidates, key=lambda c: np.sum((a @ c - b) ** 2))
    dropout = []
    material_counts = []
    for material_id in range(5):
        mask = eligible & (materials == material_id)
        count = int(mask.sum())
        missing = int((mask & ~returned).sum())
        # Missing coverage is explicit; zero is not asserted as a measured rate.
        dropout.append(float(missing / count) if count else None)
        material_counts.append({"material_id": material_id, "reference_count": count, "missing_count": missing})
    return {"schema_version": "ares.depth_noise_fit.v1", "source": source,
            "model": "std(z)=sqrt(base_std_m^2+(quadratic_std_per_m2*z^2)^2)",
            "base_std_m": float(np.sqrt(coefficients[0])),
            "quadratic_std_per_m2": float(np.sqrt(coefficients[1])),
            "material_hole_probability": dropout, "material_counts": material_counts,
            "reference_count": int(eligible.sum()), "return_count": int(returned.sum()),
            "missing_count": int((eligible & ~returned).sum()),
            "bias_m": float(np.mean(error)), "residual_rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "centered_std_m": float(np.std(error)), "distance_bins": bins,
            "assumptions": ["paired registered depths in optical-axis metres", "independent accurate reference",
                            "stationary independent zero-mean errors after bin centering",
                            "distance bins use count-weighted nonnegative variance least squares",
                            "material missing fractions include all causes; range/edge effects are not separated"],
            "provenance": "synthetic fixture fit; not real sensor calibration" if source == "synthetic" else
                          "user-declared real pairs; registration and reference accuracy not verified"}
