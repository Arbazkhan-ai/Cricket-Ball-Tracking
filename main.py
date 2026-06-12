from ultralytics import YOLO
import cv2
import numpy as np

ball_model = YOLO('detect/train/weights/best.pt')
pitch_model = YOLO('detect/train/weights/pitch.pt')

ball_track = []
frames_without_ball = 0
MAX_MISSING_FRAMES = 10

cap = cv2.VideoCapture("ball_4.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))


# ---------------------------
# PHYSICS ANALYSIS ENGINE
# ---------------------------
def analyze_ball(track):
    if len(track) < 6:
        return "INSUFFICIENT DATA"

    pts = np.array(track)

    x = pts[:, 0]
    y = pts[:, 1]

    # ---------------- BOUNCE ----------------
    bounce = False
    for i in range(2, len(y)-2):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            bounce = True
            break

    # ---------------- SWING ----------------
    swing_amount = x[-1] - x[0]
    swing_strength = abs(swing_amount)

    # ---------------- SPIN (CURVATURE) ----------------
    spin = 0
    for i in range(2, len(x)):
        spin += abs(x[i] - 2*x[i-1] + x[i-2])

    # ---------------- SPEED ----------------
    dist = 0
    for i in range(1, len(track)):
        dist += np.linalg.norm(np.array(track[i]) - np.array(track[i-1]))

    speed = dist / len(track)

    # ---------------- CLASSIFICATION ----------------
    if bounce and spin > 200:
        return "SPIN BOUNCER"
    elif bounce:
        return "BOUNCING BALL"
    elif swing_strength > 50:
        return "SWING BALL"
    elif speed > 15:
        return "FAST BALL"
    else:
        return "NORMAL BALL"


# ---------------------------
# TV TRAIL FUNCTION
# ---------------------------
def draw_trail(frame, track):
    if len(track) < 2:
        return frame

    overlay = frame.copy()

    for i in range(1, len(track)):
        alpha = i / len(track)

        color = (
            int(0 + 255 * alpha),
            int(255 * (1 - alpha)),
            255
        )

        thickness = int(8 * alpha + 2)

        cv2.line(overlay, track[i-1], track[i], color, thickness)

    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i, pt in enumerate(track):
        r = int(5 + 5 * (i / len(track)))
        cv2.circle(frame, pt, r, (0, 255, 255), -1)

    return frame


# ---------------------------
# MAIN LOOP
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results_ball = ball_model(frame)
    annotated_frame = results_ball[0].plot()

    # -------- PITCH DETECTION --------
    results_pitch = pitch_model.predict(frame, conf=0.5, verbose=False)

    pitch_boxes = []

    for res in results_pitch:
        for box in res.boxes:
            if int(box.cls[0]) == 1:
                px1, py1, px2, py2 = box.xyxy[0].tolist()
                pitch_boxes.append((px1, py1, px2, py2))

                cv2.rectangle(annotated_frame,
                              (int(px1), int(py1)),
                              (int(px2), int(py2)),
                              (0, 255, 0), 2)

                cv2.putText(annotated_frame, "PITCH",
                            (int(px1), int(py1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # -------- BALL TRACKING --------
    boxes = results_ball[0].boxes
    ball_detected_this_frame = False

    for box in boxes:
        cls_name = ball_model.names[int(box.cls[0])]

        if cls_name.lower() == 'ball':
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # only if inside pitch
            inside = any(px1 <= cx <= px2 and py1 <= cy <= py2
                         for (px1, py1, px2, py2) in pitch_boxes)

            if inside:
                ball_track.append((cx, cy))
                ball_detected_this_frame = True
                frames_without_ball = 0
                break

    if not ball_detected_this_frame:
        frames_without_ball += 1
        if frames_without_ball > MAX_MISSING_FRAMES:
            ball_track = []

    # -------- DRAW TRAIL --------
    annotated_frame = draw_trail(annotated_frame, ball_track)

    # -------- ANALYZE BALL TYPE --------
    ball_type = analyze_ball(ball_track)

    # -------- DISPLAY BALL TYPE --------
    cv2.rectangle(annotated_frame, (20, 20), (400, 120), (0, 0, 0), -1)

    cv2.putText(annotated_frame, f"BALL TYPE: {ball_type}",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    cv2.putText(annotated_frame, f"POINTS: {len(ball_track)}",
                (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    # -------- SAVE + SHOW --------
    out.write(annotated_frame)
    cv2.imshow("AI Cricket Physics System", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("l") and len(ball_track) > 6:
        import lbw_analysis
        lbw_analysis.run_lbw(ball_track)

    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()