"""
SafePath AI - Multimodal hazard detection + adaptive indoor evacuation routing.
Run:  streamlit run app.py
"""
import heapq
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROWS, COLS = 20, 30
BLOCK_AT = 0.8      # path cells above this hazard level trigger a warning
DANGER_WEIGHT = 25  # how strongly smoke/heat repels the route


# ---------------------------------------------------------------- floor plan
def build_floor():
    """0 = free, 1 = wall, 2 = exit. A corridor (rows 9-10) with rooms above/below."""
    g = np.zeros((ROWS, COLS), dtype=int)
    g[0, :] = g[-1, :] = 1
    g[:, 0] = g[:, -1] = 1
    g[8, :] = 1
    g[11, :] = 1
    for c in (10, 20):                      # room dividers
        g[1:8, c] = 1
        g[12:19, c] = 1
    for c in (5, 15, 25):                   # doors into the corridor
        g[8, c] = 0
        g[11, c] = 0
    for e in [(9, 0), (10, 0), (9, COLS - 1), (10, COLS - 1), (ROWS - 1, 25)]:
        g[e] = 2                            # exits
    return g


GRID = build_floor()
EXITS = [tuple(p) for p in np.argwhere(GRID == 2)]

ZONES = {  # name: (row0, row1, col0, col1) inclusive-exclusive
    "Corridor West": (9, 11, 1, 10),
    "Corridor Centre": (9, 11, 10, 20),
    "Corridor East": (9, 11, 20, 29),
    "Room Top-Left": (1, 8, 1, 10),
    "Room Top-Centre": (1, 8, 11, 20),
    "Room Top-Right": (1, 8, 21, 29),
    "Room Bottom-Left": (12, 19, 1, 10),
    "Room Bottom-Centre": (12, 19, 11, 20),
    "Room Bottom-Right": (12, 19, 21, 29),
}


# ------------------------------------------------------------ vision module
def smoke_score(frame_bgr):
    """Fraction of pixels that look like smoke: low saturation, mid brightness.
    Replace/extend with a trained CNN or YOLO model for the final version."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    mask = (s < 50) & (v > 70) & (v < 220)
    return float(mask.mean())


def synthetic_frame(density, size=(240, 320)):
    """Fake CCTV frame so you can demo without real footage."""
    rng = np.random.default_rng(0)
    base = np.full((*size, 3), (40, 90, 160), dtype=np.uint8)   # colourful 'room'
    smoke = cv2.GaussianBlur(rng.integers(100, 200, size, dtype=np.uint8), (0, 0), 25)
    smoke = cv2.normalize(smoke, None, 0, 255, cv2.NORM_MINMAX)
    alpha = np.clip(smoke / 255.0 * density * 1.6, 0, 1)[..., None]
    gray = np.full((*size, 3), 150, dtype=np.uint8)
    return (base * (1 - alpha) + gray * alpha).astype(np.uint8)


# ------------------------------------------------------------- sensor fusion
def fuse(vision, temp_c, co_ppm):
    t = np.clip((temp_c - 30) / 60, 0, 1)      # 30C normal -> 90C severe
    c = np.clip(co_ppm / 400, 0, 1)            # 400 ppm ~ dangerous
    return float(np.clip(0.5 * vision * 2 + 0.3 * t + 0.2 * c, 0, 1))


def spread(hazard, steps):
    """Cheap smoke-spread model: hazard leaks to free neighbours and decays."""
    free = GRID != 1
    h = hazard.copy()
    for _ in range(steps):
        p = np.pad(h, 1)
        nb = np.maximum.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        h = np.where(free, np.maximum(h, 0.88 * nb), 0)
    return h


# --------------------------------------------------------------- A* routing
def astar(start, hazard, use_hazard=True):
    goals = EXITS

    def heur(p):
        return min(abs(p[0] - g[0]) + abs(p[1] - g[1]) for g in goals)

    def cell_cost(p):
        return 1 + (DANGER_WEIGHT * hazard[p] if use_hazard else 0)

    pq = [(heur(start), 0.0, start)]
    came, best = {start: None}, {start: 0.0}
    while pq:
        _, g, cur = heapq.heappop(pq)
        if cur in goals:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return path[::-1], g
        if g > best.get(cur, 1e9):
            continue
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cur[0] + dr, cur[1] + dc)
            if not (0 <= n[0] < ROWS and 0 <= n[1] < COLS) or GRID[n] == 1:
                continue
            ng = g + cell_cost(n)
            if ng < best.get(n, 1e9):
                best[n], came[n] = ng, cur
                heapq.heappush(pq, (ng + heur(n), ng, n))
    return None, None


def directions(path):
    names = {(1, 0): "down", (-1, 0): "up", (0, 1): "right", (0, -1): "left"}
    out, i = [], 1
    while i < len(path):
        d = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        n = 0
        while i < len(path) and (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) == d:
            n, i = n + 1, i + 1
        out.append(f"Move {names[d]} {n} step{'s' if n > 1 else ''}")
    return out + ["You have reached an EXIT"]


def exposure(path, hazard):
    return float(sum(hazard[p] for p in path))


# ----------------------------------------------------------------------- UI
st.set_page_config(page_title="SafePath AI", layout="wide")
st.title("SafePath AI - Adaptive Indoor Evacuation")

with st.sidebar:
    st.header("1. Hazard source")
    zone = st.selectbox("Zone where hazard detected", list(ZONES), index=1)
    st.header("2. Camera (vision)")
    up = st.file_uploader("Upload CCTV image (optional)", type=["jpg", "jpeg", "png"])
    demo_density = st.slider("Demo smoke density (if no upload)", 0.0, 1.0, 0.6)
    st.header("3. IoT sensors")
    temp = st.slider("Temperature (C)", 25, 100, 70)
    co = st.slider("CO (ppm)", 0, 500, 250)
    st.header("4. Simulation")
    spread_steps = st.slider("Smoke spread (time steps)", 0, 15, 5)
    st.header("5. Your location")
    sr = st.slider("Row", 1, ROWS - 2, 3)
    sc = st.slider("Column", 1, COLS - 2, 3)

if up is not None:
    frame = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
else:
    frame = synthetic_frame(demo_density)

vision = smoke_score(frame)
severity = fuse(vision, temp, co)

hazard = np.zeros((ROWS, COLS))
r0, r1, c0, c1 = ZONES[zone]
hazard[r0:r1, c0:c1] = severity
hazard[GRID == 1] = 0
hazard = spread(hazard, spread_steps)

start = (sr, sc)
col1, col2 = st.columns([1, 1.4])
with col1:
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Camera feed", use_container_width=True)
    st.metric("Vision smoke score", f"{vision:.2f}")
    st.metric("Fused hazard severity", f"{severity:.2f}")

fig, ax = plt.subplots(figsize=(9, 6))
ax.imshow(GRID, cmap=ListedColormap(["#f4f4f4", "#222222", "#2ecc71"]), vmin=0, vmax=2)
ax.imshow(np.ma.masked_where(hazard < 0.02, hazard), cmap="Reds", alpha=0.75, vmin=0, vmax=1)

if GRID[start] == 1:
    msg = "Selected location is inside a wall - move the sliders."
else:
    safe, cs = astar(start, hazard, True)
    naive, _ = astar(start, hazard, False)
    if naive:
        ax.plot([p[1] for p in naive], [p[0] for p in naive], "--", c="orange", lw=2, label="Static shortest route")
    if safe:
        ax.plot([p[1] for p in safe], [p[0] for p in safe], "-", c="blue", lw=3, label="AI adaptive route")
        msg = None
    else:
        msg = "No safe route exists. Shelter in place and alert responders."
ax.scatter(start[1], start[0], c="black", s=120, marker="o", label="You")
ax.legend(loc="upper right", fontsize=8)
ax.set_xticks([]); ax.set_yticks([])
with col2:
    st.pyplot(fig)

st.subheader("Evacuation guidance")
if msg:
    st.error(msg)
else:
    a, b, c = st.columns(3)
    a.metric("Adaptive route length", len(safe))
    b.metric("Hazard exposure (adaptive)", f"{exposure(safe, hazard):.1f}")
    b2 = exposure(naive, hazard) if naive else float("nan")
    c.metric("Hazard exposure (static)", f"{b2:.1f}")
    if max(hazard[p] for p in safe) > BLOCK_AT:
        st.warning("Every route crosses heavy smoke. Cover nose/mouth, stay low, and move fast; alert responders.")
    for i, d in enumerate(directions(safe), 1):
        st.write(f"{i}. {d}")
