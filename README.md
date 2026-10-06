# Air Draw

Draw in the air with your index finger using your webcam. Hand tracking by MediaPipe, drawing and display by OpenCV.

## Setup

MediaPipe needs a Python version it ships wheels for (3.12 works well).

```bash
uv venv --python 3.12 venv
source venv/bin/activate        # fish: source venv/bin/activate.fish
uv pip install -r requirements.txt
python air_draw.py
```

## Gestures

| Gesture | Action |
|---|---|
| Index finger up only | Draw |
| Index + middle fingers up | Poke the colors buttons to switch colors |
| Pinch (thumb + index) on a drawing | Grab and drag it; open your fingers to drop it |

Pick a color by clicking a button with the mouse, hovering over it with two fingers, or pinching while your hand is over it.

## Keys

| Key | Action |
|---|---|
| `s` | Save a transparent PNG |
| `c` | Clear everything |
| `u` | Undo (stroke, drag, or clear) |
| `+` / `-` | Brush size |
| `q` / `Esc` | Quit |

## Tweaking

At the top of `air_draw.py`: `WIDTH, HEIGHT` (try 1280x720 on slower machines), `SMOOTHING`, `GRAB_RADIUS`, and `PINCH_ON` / `PINCH_OFF`.
