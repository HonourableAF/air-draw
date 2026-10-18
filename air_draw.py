"""
Air Draw - draw in the air with your index finger using your webcam.
Gruvbox edition.

Gestures
  Index finger up only          -> DRAW
  Index + middle fingers up     -> HOVER (move without drawing; pick colors from the toolbar)
  Pinch (thumb + index) on a
  drawing, then move your hand  -> GRAB and DRAG that stroke; open fingers to drop it
  Colors: click a button with the mouse, hover over it with two fingers,
          or pinch while your hand is over it

Keys
  s = save PNG    c = clear all    u = UNDO (stroke, erase, drag or clear)
  + / - = brush size    h = show key hints    q or ESC = quit
"""

import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

WIDTH, HEIGHT = 1280, 720   # canvas size (window is resizable, so scaling up is free)
CAM_W, CAM_H = 1280, 720    # webcam capture size, only used for hand tracking
SHOW_CAMERA = False         # False = black gruvbox canvas + hand skeleton only; True = show the camera
SCALE = HEIGHT / 720        # sizes below scale with the resolution
TOOLBAR_H = int(90 * SCALE)
SMOOTHING = 0.5            # 0 = very smooth/laggy, 1 = raw/jittery
CLEAR_HOLD_SECONDS = 1.0
GRAB_RADIUS = int(50 * SCALE)  # how close (px) a pinch must be to a stroke to grab it
PINCH_ON = 0.28            # pinch distance / hand size below this -> pinching
PINCH_OFF = 0.40           # above this -> released (hysteresis avoids flicker)

# --- look & feel ---------------------------------------------------------------------
FEED_DIM = 0.55            # how much of the camera shows through (rest is gruvbox bg0)
FEED_DESAT = 0.35          # 0 = full color camera, 1 = black & white
GLOW_STRENGTH = 0.55       # 0 = no stroke glow
HINT_SECONDS = 8.0         # how long the key hints stay up (they fade over the last 2s)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def hex_bgr(h):
    """'#rrggbb' -> OpenCV BGR tuple."""
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


GRUV = {
    "bg0":    hex_bgr("#282828"),
    "bg1":    hex_bgr("#3c3836"),
    "bg2":    hex_bgr("#504945"),
    "fg":     hex_bgr("#ebdbb2"),
    "gray":   hex_bgr("#928374"),
    "red":    hex_bgr("#fb4934"),
    "orange": hex_bgr("#fe8019"),
    "yellow": hex_bgr("#fabd2f"),
    "green":  hex_bgr("#b8bb26"),
    "aqua":   hex_bgr("#8ec07c"),
    "blue":   hex_bgr("#83a598"),
    "purple": hex_bgr("#d3869b"),
}

# name, BGR color (None color = eraser)
TOOLS = [
    ("Red",    GRUV["red"]),
    ("Orange", GRUV["orange"]),
    ("Yellow", GRUV["yellow"]),
    ("Green",  GRUV["green"]),
    ("Aqua",   GRUV["aqua"]),
    ("Blue",   GRUV["blue"]),
    ("Purple", GRUV["purple"]),
    ("Cream",  GRUV["fg"]),
]

MODE_COLORS = {
    "DRAW": GRUV["green"],
    "HOVER": GRUV["yellow"],
    "DRAGGING": GRUV["orange"],
    "PINCH": GRUV["purple"],
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
LM_SPEC = mp_draw.DrawingSpec(color=GRUV["fg"], thickness=-1, circle_radius=max(int(2 * SCALE), 2))
CONN_SPEC = mp_draw.DrawingSpec(color=GRUV["gray"], thickness=max(int(1.5 * SCALE), 1))


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
        draw_one(layer, grabbed, GRUV["fg"], extra=8)
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


# --- styling helpers -----------------------------------------------------------------

def style_feed(frame, tint):
    """Desaturate the camera a bit and sink it toward gruvbox bg0 so strokes pop."""
    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    frame = cv2.addWeighted(frame, 1 - FEED_DESAT, gray, FEED_DESAT, 0)
    return cv2.addWeighted(frame, FEED_DIM, tint, 1 - FEED_DIM, 0)


def add_glow(frame, layer):
    """Soft warm halo around strokes (blurred at 1/4 size, so it's cheap)."""
    small = cv2.resize(layer, (WIDTH // 4, HEIGHT // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3 * SCALE)
    glow = cv2.resize(small, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame, 1.0, glow, GLOW_STRENGTH, 0)


def fill_rounded(img, r, color):
    """Fill an entire image/ROI with a rounded rectangle of `color`."""
    h, w = img.shape[:2]
    r = max(min(r, h // 2, w // 2), 0)
    cv2.rectangle(img, (r, 0), (w - 1 - r, h - 1), color, -1)
    cv2.rectangle(img, (0, r), (w - 1, h - 1 - r), color, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def panel(frame, x0, y0, x1, y1, r, color, alpha, content=None):
    """Translucent rounded panel. `content(overlay)` can paint extra stuff onto it."""
    roi = frame[y0:y1, x0:x1]
    over = roi.copy()
    fill_rounded(over, r, color)
    if content is not None:
        content(over)
    frame[y0:y1, x0:x1] = cv2.addWeighted(over, alpha, roi, 1 - alpha, 0)


def text_centered(frame, text, cx, y, scale, color, thick=1):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(frame, text, (cx - tw // 2, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_toolbar(frame, selected):
    s = SCALE
    n = len(TOOLS)
    w = WIDTH // n
    bar = frame[0:TOOLBAR_H]
    frame[0:TOOLBAR_H] = cv2.addWeighted(np.full_like(bar, GRUV["bg1"]), 0.82, bar, 0.18, 0)
    cv2.line(frame, (0, TOOLBAR_H), (WIDTH, TOOLBAR_H), GRUV["bg2"], max(int(2 * s), 1), cv2.LINE_AA)
    for i, (name, color) in enumerate(TOOLS):
        x0, x1 = i * w, (i + 1) * w
        cx = x0 + w // 2
        cy = int(36 * s)
        if i == selected:
            panel(frame, x0 + int(10 * s), int(6 * s), x1 - int(10 * s), TOOLBAR_H - int(6 * s),
                  int(16 * s), GRUV["bg2"], 0.95)
        if color is not None:
            cv2.circle(frame, (cx, cy), int(19 * s), color, -1, cv2.LINE_AA)
            ring = GRUV["fg"] if i == selected else GRUV["gray"]
            cv2.circle(frame, (cx, cy), int(25 * s), ring, 2 if i == selected else 1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (cx - int(19 * s), cy - int(19 * s)),
                          (cx + int(19 * s), cy + int(19 * s)), GRUV["fg"], 2, cv2.LINE_AA)
        label_col = GRUV["fg"] if i == selected else GRUV["gray"]
        text_centered(frame, name, cx, int(80 * s), 0.5 * s, label_col, 1)


def draw_cursor(frame, cur, color):
    """Fingertip cursor; its look tells you which mode you're in."""
    if cur is None:
        return
    x, y, kind, r = cur
    s = SCALE
    dot = max(int(3 * s), 2)
    if kind == "draw":
        cv2.circle(frame, (x, y), r // 2 + int(9 * s), color, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), dot, GRUV["fg"], -1, cv2.LINE_AA)
    elif kind == "erase":
        cv2.circle(frame, (x, y), r, GRUV["fg"], 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), dot, GRUV["fg"], -1, cv2.LINE_AA)
    elif kind == "hover":
        cv2.circle(frame, (x, y), int(15 * s), GRUV["yellow"], 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), dot, GRUV["yellow"], -1, cv2.LINE_AA)
    elif kind == "grab":
        cv2.circle(frame, (x, y), int(18 * s), GRUV["orange"], 3, cv2.LINE_AA)
        cv2.circle(frame, (x, y), dot, GRUV["orange"], -1, cv2.LINE_AA)
    else:  # pinch with nothing grabbed
        cv2.circle(frame, (x, y), int(13 * s), GRUV["purple"], 2, cv2.LINE_AA)


def draw_hud(frame, mode, brush, n_strokes, color):
    s = SCALE
    name = mode.split(" (")[0]
    mcol = MODE_COLORS.get(name, GRUV["gray"])
    fs = 0.6 * s
    items = [name, f"brush {brush}", f"strokes {n_strokes}"]
    widths = [cv2.getTextSize(t, FONT, fs, 1)[0][0] for t in items]
    th = cv2.getTextSize("Ag", FONT, fs, 1)[0][1]
    pad, gap, dot = int(16 * s), int(26 * s), int(7 * s)
    h = int(44 * s)
    pw = h - 8                                   # brush preview box
    x_dot = pad + dot
    x_name = x_dot + dot + int(10 * s)
    x_prev = x_name + widths[0] + gap
    x_brush = x_prev + pw + int(8 * s)
    x_strokes = x_brush + widths[1] + gap
    total = x_strokes + widths[2] + pad

    x0, y1 = int(18 * s), HEIGHT - int(18 * s)
    y0 = y1 - h
    panel(frame, x0, y0, x0 + total, y1, h // 2, GRUV["bg1"], 0.82)

    cy = y0 + h // 2
    ty = cy + th // 2
    cv2.circle(frame, (x0 + x_dot, cy), dot, mcol, -1, cv2.LINE_AA)
    cv2.putText(frame, name, (x0 + x_name, ty), FONT, fs, GRUV["fg"], 1, cv2.LINE_AA)
    cv2.circle(frame, (x0 + x_prev + pw // 2, cy), max(min(brush // 2, pw // 2), 1),
               color if color is not None else GRUV["fg"], -1, cv2.LINE_AA)
    cv2.putText(frame, items[1], (x0 + x_brush, ty), FONT, fs, GRUV["gray"], 1, cv2.LINE_AA)
    cv2.putText(frame, items[2], (x0 + x_strokes, ty), FONT, fs, GRUV["gray"], 1, cv2.LINE_AA)


def draw_hints(frame, a):
    """Key hints, bottom right. `a` is 0..1 opacity (fades out)."""
    if a <= 0:
        return
    s = SCALE
    text = "s save    c clear    u undo    + / -  brush    h hints    q quit"
    fs = 0.55 * s
    (tw, th), _ = cv2.getTextSize(text, FONT, fs, 1)
    pad, h = int(18 * s), int(40 * s)
    x1, y1 = WIDTH - int(18 * s), HEIGHT - int(18 * s)
    x0, y0 = x1 - tw - 2 * pad, y1 - h

    def content(over):
        cv2.putText(over, text, (pad, h // 2 + th // 2), FONT, fs, GRUV["fg"], 1, cv2.LINE_AA)

    panel(frame, x0, y0, x1, y1, h // 2, GRUV["bg1"], 0.85 * a, content)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)      # V4L2 is the native Linux backend
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera.")
    # MJPG must be set BEFORE the size: most webcams only offer 720p/1080p at 30fps in MJPG,
    # and fall back to blurry low-res / low-fps raw formats otherwise.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
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
    tint = np.full((HEIGHT, WIDTH, 3), GRUV["bg0"], dtype=np.uint8)
    hints_until = time.time() + HINT_SECONDS

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
            cam = cv2.flip(frame, 1)

            # track on a small copy (faster, landmarks are normalized so it doesn't matter)
            small = cv2.resize(cam, (640, 360), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # what you see: a plain gruvbox canvas (or the tinted camera if SHOW_CAMERA)
            if SHOW_CAMERA:
                frame = style_feed(cv2.resize(cam, (WIDTH, HEIGHT)), tint)
            else:
                frame = tint.copy()

            mode = "NO HAND"
            erasing_now = False
            cursor = None                     # (x, y, kind, radius) drawn after the strokes
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                lm = hand.landmark
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, LM_SPEC, CONN_SPEC)

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
                        cursor = (x, y, "grab", 0)
                    else:
                        mode = "PINCH (nothing grabbed)"
                        cursor = (x, y, "pinch", 0)
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
                        cursor = (x, y, "hover", 0)
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
                                cursor = (x, y, "erase", r)
                            else:
                                if current is None:
                                    push_history(history, strokes)
                                    current = {"pts": [], "color": color, "size": brush}
                                    strokes.append(current)
                                add_point(current, x, y)
                                cursor = (x, y, "draw", brush)
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

            # blend strokes over the video (glow first, then the crisp strokes on top)
            layer = render_strokes(strokes, grabbed)
            if strokes:
                frame = add_glow(frame, layer)
            gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
            mask = gray > 0
            frame[mask] = layer[mask]

            cur_color = TOOLS[selected][1]
            draw_cursor(frame, cursor, cur_color)
            draw_toolbar(frame, selected)
            draw_hud(frame, mode, brush, len(strokes), cur_color)
            draw_hints(frame, min(max((hints_until - time.time()) / 2.0, 0.0), 1.0))

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
            elif key == ord("h"):
                hints_until = time.time() + HINT_SECONDS
            elif key in (ord("+"), ord("=")):
                brush = min(brush + 2, 40)
            elif key in (ord("-"), ord("_")):
                brush = max(brush - 2, 2)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
