import cv2
import numpy as np
from hawk_eye_engine import estimate_speed, swing_amount, spin_intensity


PITCH_W, PITCH_H = 900, 500
STUMP_X = 750
STUMP_Y1, STUMP_Y2 = 180, 320


# ---------------------------
# ROBUST PITCH DETECTION
# ---------------------------
def detect_pitch(points):
    for i in range(2, len(points)-2):
        prev_y = points[i-1][1]
        curr_y = points[i][1]
        next_y = points[i+1][1]

        # bounce = local MAX y (because image y increases downward)
        if curr_y > prev_y and curr_y > next_y:
            return points[i]
    return None


def run_lbw(ball_track):

    canvas = np.zeros((PITCH_H, PITCH_W, 3), dtype=np.uint8)
    canvas[:] = (30, 120, 30)

    pts = np.array(ball_track, dtype=np.int32)

    if len(pts) < 6:
        return

    x = pts[:, 0]
    y = pts[:, 1]

    # -------------------------
    # STABLE CURVE FITTING
    # -------------------------
    try:
        poly = np.polyfit(x, y, 2)
        f = np.poly1d(poly)
    except:
        return

    future_x = np.linspace(x[0], STUMP_X + 120, 100)
    future_y = f(future_x)

    # -------------------------
    # PHYSICS FEATURES
    # -------------------------
    speed = estimate_speed(ball_track)
    swing = swing_amount(ball_track)
    spin = spin_intensity(ball_track)

    # -------------------------
    # PITCH DETECTION (FIXED)
    # -------------------------
    pitch_point = detect_pitch(ball_track)

    if pitch_point is not None:
        cv2.circle(canvas, tuple(pitch_point), 10, (255, 255, 0), -1)
        cv2.putText(canvas, "PITCH", (pitch_point[0]-40, pitch_point[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # -------------------------
    # HIT WICKET CHECK
    # -------------------------
    hit_wicket = False

    for i in range(1, len(future_x)):
        p1 = (int(future_x[i-1]), int(future_y[i-1]))
        p2 = (int(future_x[i]), int(future_y[i]))

        cv2.line(canvas, p1, p2, (0, 0, 255), 2)

        if STUMP_X - 8 < p2[0] < STUMP_X + 18 and STUMP_Y1 < p2[1] < STUMP_Y2:
            hit_wicket = True

    # -------------------------
    # DRAW BALL PATH
    # -------------------------
    for i in range(1, len(pts)):
        cv2.line(canvas,
                 tuple(pts[i-1]),
                 tuple(pts[i]),
                 (0, 255, 255), 2)

    # -------------------------
    # DRAW STUMPS
    # -------------------------
    for i in range(3):
        x = STUMP_X + i * 6
        cv2.line(canvas, (x, STUMP_Y1), (x, STUMP_Y2), (255, 255, 255), 3)

    # -------------------------
    # 3D FAKE DEPTH EFFECT
    # -------------------------
    for i, p in enumerate(pts):
        depth = int(i * 0.4)
        cv2.circle(canvas, (p[0], p[1] - depth), 3, (200, 200, 200), -1)

    # -------------------------
    # DECISION ENGINE
    # -------------------------
    if pitch_point is not None and pitch_point[0] < 300:
        decision = "NOT OUT (Outside Leg)"
        color = (0, 255, 0)

    elif hit_wicket:
        decision = "OUT (LBW)"
        color = (0, 0, 255)

    else:
        decision = "NOT OUT"
        color = (0, 255, 0)

    # -------------------------
    # IPL UI PANEL
    # -------------------------
    cv2.rectangle(canvas, (20, 20), (450, 190), (0, 0, 0), -1)

    cv2.putText(canvas, f"Speed: {speed} km/h", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(canvas, f"Swing: {swing}px", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(canvas, f"Spin: {spin}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(canvas, decision, (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    cv2.imshow("Hawk-Eye LBW FIXED SYSTEM", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()