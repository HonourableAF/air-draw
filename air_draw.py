"""
Air Draw - draw in the air with your index finger using your webcam.

Gestures
  Index finger up only          -> DRAW
  Index + middle fingers up     -> HOVER (move without drawing; pick colors from the toolbar)
  Pinch (thumb + index) on a
  drawing, then move your hand  -> GRAB and DRAG that stroke; open fingers to drop it
  Colors: click a button with the mouse, hover over it with two fingers,
          or pinch while your hand is over it

Keys
  s = save PNG    c = clear all    u = UNDO (stroke, erase, drag or clear)
  + / - = brush size    q or ESC = quit
"""

import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

WIDTH, HEIGHT = 1920, 1080  # try 1280, 720 if your camera or PC struggles
SCALE = HEIGHT / 720        # sizes below scale with the resolution
TOOLBAR_H = int(90 * SCALE)
SMOOTHING = 0.5            # 0 = very smooth/laggy, 1 = raw/jittery
CLEAR_HOLD_SECONDS = 1.0
GRAB_RADIUS = int(50 * SCALE)  # how close (px) a pinch must be to a stroke to grab it
PINCH_ON = 0.28            # pinch distance / hand size below this -> pinching
PINCH_OFF = 0.40           # above this -> released (hysteresis avoids flicker)

# name, BGR color (None color = eraser)
TOOLS = [
    ("Red",    (0, 0, 255)),
    ("Green",  (0, 200, 0)),
    ("Blue",   (255, 80, 0)),
    ("Yellow", (0, 230, 255)),
    ("White",  (255, 255, 255)),
]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def fingers_up(lm):
    """Return [index, middle, ring, pinky] booleans (tip above the PIP joint)."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return [lm[t].y < lm[p].y for t, p in zip(tips, pips)]


def dist_to_stroke(p, pts):
    """Shortest distance from point p to a polyline (Nx2 array)."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 1:
        return float(np.linalg.norm(p - pts[0]))
    a, b = pts[:-1], pts[1:]
    ab = b - a
    denom = np.maximum((ab ** 2).sum(axis=1), 1e-9)
    t = np.clip(((p - a) * ab).sum(axis=1) / denom, 0, 1)
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(proj - p, axis=1).min())


STEP_PX = 4 * SCALE   # max gap between stored points, so strokes can be cut cleanly


def add_point(stroke, x, y):
    """Append a point, filling in any gap from fast movement with evenly spaced points."""
    pts = stroke["pts"]
    if pts:
        lx, ly = pts[-1]
        d = float(np.hypot(x - lx, y - ly))
        n = int(d // STEP_PX)
        for i in range(1, n + 1):
            t = i / (n + 1)
            pts.append([lx + (x - lx) * t, ly + (y - ly) * t])
    pts.append([float(x), float(y)])


def erase_along(strokes, p0, p1, r):
    """Remove only the points within r of the eraser's path (p0 -> p1).
    A stroke that gets cut in the middle becomes two separate strokes."""
    out = []
    ab = p1 - p0
    denom = max(float(ab @ ab), 1e-9)
    for s in strokes:
        pts = np.array(s["pts"], dtype=float)
        t = np.clip(((pts - p0) @ ab) / denom, 0, 1)
        proj = p0 + t[:, None] * ab
        keep = np.linalg.norm(pts - proj, axis=1) > r
        if keep.all():
            out.append(s)
            continue
        run = []
        for pt, k in zip(pts, keep):
            if k:
                run.append([float(pt[0]), float(pt[1])])
            else:
                if len(run) >= 2:   # drop 1-point leftovers so no stray specks remain
                    out.append({"pts": run, "color": s["color"], "size": s["size"]})
                run = []
        if len(run) >= 2:
            out.append({"pts": run, "color": s["color"], "size": s["size"]})
    return out


def push_history(history, strokes):
    """Save a snapshot of all strokes so the next action can be undone with 'u'."""
    history.append([{"pts": [p[:] for p in s["pts"]], "color": s["color"], "size": s["size"]}
                    for s in strokes])
    if len(history) > 50:
        history.pop(0)


def render_strokes(strokes, grabbed=None):
    """Draw all strokes onto a black layer and return it."""
    layer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    if grabbed is not None:  # highlight halo under the grabbed stroke
        draw_one(layer, grabbed, (200, 200, 200), extra=8)
    for s in strokes:
        draw_one(layer, s, s["color"])
    return layer


def draw_one(layer, s, color, extra=0):
    pts = np.array(s["pts"], dtype=np.int32)
    size = s["size"] + extra
    if len(pts) == 1:
        cv2.circle(layer, tuple(pts[0]), max(size // 2, 1), color, -1, cv2.LINE_AA)
    else:
        cv2.polylines(layer, [pts], False, color, size, cv2.LINE_AA)


def draw_toolbar(frame, selected):
    n = len(TOOLS)
    w = WIDTH // n
    for i, (name, color) in enumerate(TOOLS):
        x0, x1 = i * w, (i + 1) * w
        cv2.rectangle(frame, (x0, 0), (x1, TOOLBAR_H), (40, 40, 40), -1)
        s = SCALE
        cx = x0 + w // 2
        if color is not None:
            cv2.circle(frame, (cx, int(35 * s)), int(22 * s), color, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (cx - int(22 * s), int(13 * s)), (cx + int(22 * s), int(57 * s)),
                          (200, 200, 200), 2)
        cv2.putText(frame, name, (cx - int(30 * s), int(80 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (230, 230, 230), 1, cv2.LINE_AA)
        if i == selected:
            cv2.rectangle(frame, (x0 + 3, 3), (x1 - 3, TOOLBAR_H - 3), (255, 255, 255), 3)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)      # V4L2 is the native Linux backend
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera.")
    # MJPG must be set BEFORE the size: most webcams only offer 720p/1080p at 30fps in MJPG,
    # and fall back to blurry low-res / low-fps raw formats otherwise.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # less lag
    print("Camera running at",
          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
          "@", cap.get(cv2.CAP_PROP_FPS), "fps")
    # resizable window; WINDOW_GUI_NORMAL hides OpenCV's built-in Qt toolbar and status bar
    cv2.namedWindow("Air Draw", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    strokes = []             # list of {"pts": [[x, y], ...], "color": BGR, "size": int}
    current = None           # stroke currently being drawn
    erase_prev = None        # previous eraser position (so fast swipes erase the whole path)
    history = []             # snapshots for undo
    grabbed = None           # stroke currently being dragged
    last_pos = None          # last pinch position while dragging
    pinching = False
    selected = 0

    def on_mouse(event, mx, my, flags, param):
        """Click a color button in the toolbar with the mouse."""
        nonlocal selected
        if event == cv2.EVENT_LBUTTONDOWN and my < TOOLBAR_H:
            selected = min(mx // (WIDTH // len(TOOLS)), len(TOOLS) - 1)

    cv2.setMouseCallback("Air Draw", on_mouse)
    brush = int(8 * SCALE)
    smooth = None
    clear_started = None

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            # track on a small copy (faster, landmarks are normalized so it doesn't matter);
            # the full-resolution frame is what you see on screen
            small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            mode = "NO HAND"
            erasing_now = False
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                lm = hand.landmark
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                # --- pinch detection (thumb tip 4 vs index tip 8, scaled by hand size) ---
                p4 = np.array([lm[4].x * WIDTH, lm[4].y * HEIGHT])
                p8 = np.array([lm[8].x * WIDTH, lm[8].y * HEIGHT])
                wrist = np.array([lm[0].x * WIDTH, lm[0].y * HEIGHT])
                mcp9 = np.array([lm[9].x * WIDTH, lm[9].y * HEIGHT])
                hand_size = max(np.linalg.norm(wrist - mcp9), 1.0)
                ratio = np.linalg.norm(p4 - p8) / hand_size
                was_pinching = pinching
                pinching = ratio < (PINCH_OFF if was_pinching else PINCH_ON)

                # target point: pinch midpoint while pinching, else index fingertip
                raw = (p4 + p8) / 2 if pinching else p8
                if smooth is None or pinching != was_pinching:
                    smooth = raw
                else:
                    smooth = SMOOTHING * raw + (1 - SMOOTHING) * smooth
                x, y = int(smooth[0]), int(smooth[1])

                idx, mid, ring, pinky = fingers_up(lm)

                if pinching:
                    clear_started = None
                    current = None
                    if not was_pinching and y > TOOLBAR_H:
                        # pinch just started: try to grab the nearest stroke
                        best, best_d = None, GRAB_RADIUS
                        for s in strokes:
                            d = dist_to_stroke(smooth, s["pts"])
                            if d < best_d:
                                best, best_d = s, d
                        grabbed = best
                        if best is not None:
                            push_history(history, strokes)
                        last_pos = smooth.copy()
                    elif not was_pinching:
                        # pinch inside the toolbar = "click" that color button
                        selected = min(x // (WIDTH // len(TOOLS)), len(TOOLS) - 1)
                    if grabbed is not None:
                        mode = "DRAGGING"
                        delta = smooth - last_pos
                        for pt in grabbed["pts"]:
                            pt[0] += delta[0]
                            pt[1] += delta[1]
                        last_pos = smooth.copy()
                        cv2.circle(frame, (x, y), 16, (255, 255, 255), 3)
                    else:
                        mode = "PINCH (nothing grabbed)"
                        cv2.circle(frame, (x, y), 12, (120, 120, 120), 2)
                else:
                    grabbed = None
                    last_pos = None

                    if idx and mid and ring and pinky:
                        mode = "OPEN PALM (does nothing)"
                        current = None
                    elif idx and mid:
                        mode = "HOVER"
                        clear_started = None
                        current = None
                        cv2.circle(frame, (x, y), 14, (255, 255, 255), 2)
                        if y < TOOLBAR_H:
                            selected = min(x // (WIDTH // len(TOOLS)), len(TOOLS) - 1)
                    elif idx and not mid and not ring and not pinky:
                        mode = "DRAW"
                        clear_started = None
                        if y > TOOLBAR_H:
                            color = TOOLS[selected][1]
                            if color is None:   # eraser cuts out only the parts it touches
                                r = brush * 3
                                p1 = np.array([x, y], dtype=float)
                                p0 = erase_prev if erase_prev is not None else p1
                                if erase_prev is None:      # start of a new erase swipe
                                    push_history(history, strokes)
                                strokes[:] = erase_along(strokes, p0, p1, r)
                                erase_prev = p1
                                erasing_now = True
                                current = None
                                cv2.circle(frame, (x, y), r, (200, 200, 200), 2)
                            else:
                                if current is None:
                                    push_history(history, strokes)
                                    current = {"pts": [], "color": color, "size": brush}
                                    strokes.append(current)
                                add_point(current, x, y)
                                cv2.circle(frame, (x, y), brush // 2 + 2, color, -1)
                        else:
                            current = None
                    else:
                        mode = "IDLE"
                        current = None
                        clear_started = None
            else:
                current = None
                grabbed = None
                last_pos = None
                pinching = False
                smooth = None
                clear_started = None

            if not erasing_now:
                erase_prev = None

            # blend strokes over the video
            layer = render_strokes(strokes, grabbed)
            gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
            mask = gray > 0
            frame[mask] = layer[mask]

            draw_toolbar(frame, selected)
            cv2.putText(frame, f"Mode: {mode}   Brush: {brush}   Strokes: {len(strokes)}",
                        (10, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Air Draw", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                push_history(history, strokes)
                strokes.clear()
                current = grabbed = None
            elif key == ord("u"):
                if history:
                    strokes[:] = history.pop()
                current = grabbed = None
            elif key == ord("s"):
                out_layer = render_strokes(strokes)
                out = cv2.cvtColor(out_layer, cv2.COLOR_BGR2BGRA)
                g = cv2.cvtColor(out_layer, cv2.COLOR_BGR2GRAY)
                out[:, :, 3] = np.where(g > 0, 255, 0).astype(np.uint8)
                name = datetime.now().strftime("airdraw_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, out)
                print("Saved", name)
            elif key in (ord("+"), ord("=")):
                brush = min(brush + 2, 40)
            elif key in (ord("-"), ord("_")):
                brush = max(brush - 2, 2)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
