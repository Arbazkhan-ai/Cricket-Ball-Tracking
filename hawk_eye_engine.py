import numpy as np

# -------------------------
# SPEED ESTIMATION
# -------------------------
def estimate_speed(track, fps=30):
    if len(track) < 2:
        return 0

    dist = 0
    for i in range(1, len(track)):
        p1 = track[i - 1]
        p2 = track[i]
        dist += np.linalg.norm(np.array(p2) - np.array(p1))

    time_sec = len(track) / fps

    speed_px_s = dist / time_sec if time_sec > 0 else 0

    # rough conversion (tunable)
    speed_kmh = speed_px_s * 0.05

    return round(speed_kmh, 2)


# -------------------------
# SWING MODEL
# -------------------------
def swing_amount(track):
    if len(track) < 5:
        return 0

    x = [p[0] for p in track]
    return round(x[-1] - x[0], 2)


# -------------------------
# SPIN / CURVE INTENSITY
# -------------------------
def spin_intensity(track):
    if len(track) < 5:
        return 0

    y = [p[1] for p in track]

    curvature = 0
    for i in range(2, len(y)):
        curvature += abs(y[i] - 2*y[i-1] + y[i-2])

    return round(curvature, 2)