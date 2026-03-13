# ============================================================
#  Expected Threat (xT) Spatial Engine & Pass Evaluation Dashboard
#  Al Ahly SC | Lead Football Data Scientist
#  Single-cell Google Colab Script — Bug-Free & Production Ready
# ============================================================

# ── 0. INSTALL / IMPORTS ─────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ── 1. SYNTHETIC xT MATRIX (12 cols × 8 rows) ────────────────
GRID_COLS = 12   # pitch length segments (left → right = opponent's goal)
GRID_ROWS = 8    # pitch width  segments

PITCH_LEN = 105.0   # metres
PITCH_WID = 68.0    # metres

def build_xt_matrix(cols: int = GRID_COLS, rows: int = GRID_ROWS) -> np.ndarray:
    """
    Exponentially increasing xT towards the opponent's goal (right side).
    Extra bonus applied near the centre of the penalty box.
    Returns shape (rows, cols).
    """
    xt = np.zeros((rows, cols))
    # Penalty-box centre in grid coords (approx): col 10-11, rows 2-5
    goal_col   = cols - 1          # rightmost column
    box_centre_row = (rows - 1) / 2.0

    for r in range(rows):
        for c in range(cols):
            # Normalised horizontal position 0→1 (left to right)
            x_norm = c / (cols - 1)
            # Exponential base threat from horizontal position
            base = np.exp(4.5 * x_norm) / np.exp(4.5)  # range ~(0.011, 1)

            # Vertical distance from pitch centre (normalised 0→1)
            y_dist = abs(r - box_centre_row) / box_centre_row  # 0 at centre, 1 at edge

            # Only apply central bonus for the final third (cols ≥ 8)
            if c >= 8:
                centrality = np.exp(-3.0 * (y_dist ** 2))  # Gaussian peak at centre
            else:
                centrality = 1 - 0.3 * y_dist              # mild taper

            raw = base * centrality
            xt[r, c] = raw

    # Normalise so maximum cell = 0.95 (realistic xT scale)
    xt = xt / xt.max() * 0.95
    # Minimum floor so empty areas aren't exactly 0
    xt = np.clip(xt, 0.002, 0.95)
    return xt

XT_MATRIX = build_xt_matrix()

# ── 2. COORDINATE → GRID CELL MAPPING ────────────────────────
def coord_to_cell(x: float, y: float) -> tuple[int, int]:
    """
    Map pitch coordinate (x: 0–105, y: 0–68) to (row, col) grid index.
    Clamps to valid range to avoid edge errors.
    """
    col = int(np.clip(x / PITCH_LEN * GRID_COLS, 0, GRID_COLS - 1))
    row = int(np.clip(y / PITCH_WID * GRID_ROWS, 0, GRID_ROWS - 1))
    return row, col

def get_xt(x: float, y: float) -> float:
    r, c = coord_to_cell(x, y)
    return float(XT_MATRIX[r, c])

# ── 3. SYNTHETIC PASS DATA (Al Ahly SC) ──────────────────────
np.random.seed(42)

def make_passes(player: str, n: int, prog_ratio: float) -> pd.DataFrame:
    """
    Generate n passes for a player.
    prog_ratio controls the fraction of progressive (forward) passes.
    """
    records = []
    for _ in range(n):
        progressive = np.random.rand() < prog_ratio

        if progressive:
            # Start in middle or defensive third, end deeper
            sx = np.random.uniform(30, 75)
            sy = np.random.uniform(8, 60)
            # End further forward, possibly into a dangerous zone
            ex = np.clip(sx + np.random.uniform(10, 35), 0, 104)
            ey = np.clip(sy + np.random.uniform(-15, 15), 1, 67)
        else:
            # Safe / backward / sideways pass
            sx = np.random.uniform(35, 90)
            sy = np.random.uniform(8, 60)
            ex = np.clip(sx - np.random.uniform(5, 25), 1, 104)
            ey = np.clip(sy + np.random.uniform(-10, 10), 1, 67)

        records.append({
            "Player":   player,
            "Start_X":  round(sx, 2),
            "Start_Y":  round(sy, 2),
            "End_X":    round(ex, 2),
            "End_Y":    round(ey, 2),
        })
    return pd.DataFrame(records)

passes_df = pd.concat([
    make_passes("Emam Ashour",     n=35, prog_ratio=0.72),
    make_passes("Ali Maaloul",     n=30, prog_ratio=0.58),
    make_passes("Hussein El Shahat", n=28, prog_ratio=0.65),
], ignore_index=True)

# ── 4. CALCULATE xT_Added PER PASS ───────────────────────────
passes_df["xT_Start"] = passes_df.apply(
    lambda r: get_xt(r["Start_X"], r["Start_Y"]), axis=1)
passes_df["xT_End"]   = passes_df.apply(
    lambda r: get_xt(r["End_X"],   r["End_Y"]),   axis=1)
passes_df["xT_Added"] = (passes_df["xT_End"] - passes_df["xT_Start"]).round(4)

# Player summary
player_summary = (
    passes_df.groupby("Player")["xT_Added"]
    .agg(Total_xT_Added="sum", Passes="count", Avg_xT_Added="mean")
    .reset_index()
    .sort_values("Total_xT_Added", ascending=True)
    .round(4)
)

# ── 5. DESIGN CONSTANTS ───────────────────────────────────────
BG_PAGE   = "#060a10"
BG_CARD   = "#0d1520"
BG_PITCH  = "#0a1a10"
LINE_COL  = "#2ecc71"
TEXT_COL  = "#e0e8f0"
ACCENT1   = "#00f5d4"  # cyan-teal
ACCENT2   = "#f72585"  # hot pink
PROG_COL  = "#39ff14"  # neon green
NEG_COL   = "#ff3131"  # neon red

FONT_MAIN = "IBM Plex Mono"

# ── 6. PITCH DRAWING HELPER ───────────────────────────────────
def pitch_shapes(x_off=0, y_off=0, xscale=1.0, yscale=1.0):
    """Return list of Plotly shape dicts for a standard pitch outline."""
    W  = PITCH_LEN * xscale
    H  = PITCH_WID * yscale
    xo, yo = x_off, y_off

    def rect(x0, y0, x1, y1, **kw):
        return dict(type="rect", x0=x0+xo, y0=y0+yo, x1=x1+xo, y1=y1+yo,
                    line=dict(color=LINE_COL, width=1.2),
                    fillcolor="rgba(0,0,0,0)", **kw)

    def circle(cx, cy, r, **kw):
        return dict(type="circle", x0=cx-r+xo, y0=cy-r+yo,
                    x1=cx+r+xo, y1=cy+r+yo,
                    line=dict(color=LINE_COL, width=1.2),
                    fillcolor="rgba(0,0,0,0)", **kw)

    s = []
    # Pitch outline
    s.append(rect(0, 0, W, H))
    # Halfway line
    s.append(dict(type="line", x0=W/2+xo, y0=yo, x1=W/2+xo, y1=H+yo,
                  line=dict(color=LINE_COL, width=1.2)))
    # Centre circle
    s.append(circle(W/2, H/2, 9.15*xscale))
    # Centre spot
    s.append(dict(type="circle", x0=W/2-0.5+xo, y0=H/2-0.5+yo,
                  x1=W/2+0.5+xo, y1=H/2+0.5+yo,
                  line=dict(color=LINE_COL, width=1),
                  fillcolor=LINE_COL))

    # Left penalty area (16.5m deep, 40.32m wide centred)
    pb_d, pb_w = 16.5*xscale, 40.32*yscale
    s.append(rect(0, (H-pb_w)/2, pb_d, (H+pb_w)/2))
    # Right penalty area
    s.append(rect(W-pb_d, (H-pb_w)/2, W, (H+pb_w)/2))

    # Left 6-yard box
    sb_d, sb_w = 5.5*xscale, 18.32*yscale
    s.append(rect(0, (H-sb_w)/2, sb_d, (H+sb_w)/2))
    # Right 6-yard box
    s.append(rect(W-sb_d, (H-sb_w)/2, W, (H+sb_w)/2))

    # Left penalty arc (partial circle centred on penalty spot)
    ps_x_l = 11*xscale
    s.append(dict(type="circle", x0=ps_x_l-9.15*xscale+xo, y0=H/2-9.15*yscale+yo,
                  x1=ps_x_l+9.15*xscale+xo, y1=H/2+9.15*yscale+yo,
                  line=dict(color=LINE_COL, width=1.2),
                  fillcolor="rgba(0,0,0,0)"))
    # Right penalty arc
    ps_x_r = W - 11*xscale
    s.append(dict(type="circle", x0=ps_x_r-9.15*xscale+xo, y0=H/2-9.15*yscale+yo,
                  x1=ps_x_r+9.15*xscale+xo, y1=H/2+9.15*yscale+yo,
                  line=dict(color=LINE_COL, width=1.2),
                  fillcolor="rgba(0,0,0,0)"))

    # Goals (behind goal line for visual)
    goal_h = 7.32 * yscale
    s.append(rect(-2*xscale, (H-goal_h)/2, 0, (H+goal_h)/2))
    s.append(rect(W, (H-goal_h)/2, W+2*xscale, (H+goal_h)/2))

    return s

# ── 7. BUILD PLOTLY FIGURE ────────────────────────────────────
# Layout: 1 large pitch panel (row 1), 1 bar chart (row 2)
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.72, 0.28],
    vertical_spacing=0.06,
    subplot_titles=["", ""],
)

# ─── 7a. xT HEATMAP ──────────────────────────────────────────
# Build x / y cell-centre coordinates in pitch metres
cell_w = PITCH_LEN / GRID_COLS
cell_h = PITCH_WID / GRID_ROWS
x_centres = [cell_w * (c + 0.5) for c in range(GRID_COLS)]
y_centres = [cell_h * (r + 0.5) for r in range(GRID_ROWS)]

fig.add_trace(go.Heatmap(
    z=XT_MATRIX,
    x=x_centres,
    y=y_centres,
    colorscale="inferno",
    opacity=0.55,
    showscale=True,
    zmin=0,
    zmax=0.95,
    colorbar=dict(
        title=dict(text="xT Value", font=dict(color=TEXT_COL, size=11, family=FONT_MAIN)),
        tickfont=dict(color=TEXT_COL, size=9, family=FONT_MAIN),
        x=1.02,
        len=0.50,
        thickness=12,
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
    ),
    name="xT Grid",
    hovertemplate="xT: %{z:.3f}<extra></extra>",
), row=1, col=1)

# ─── 7b. PASS VECTORS (arrows via annotations) ───────────────
prog_passes = passes_df[passes_df["xT_Added"] >= 0]
neg_passes  = passes_df[passes_df["xT_Added"] <  0]

def pass_trace(df: pd.DataFrame, color: str, name: str, dash: str = "solid"):
    """Scatter trace with lines for passes (start+end pairs, None separator)."""
    xs, ys = [], []
    for _, row in df.iterrows():
        xs.extend([row["Start_X"], row["End_X"], None])
        ys.extend([row["Start_Y"], row["End_Y"], None])
    return go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color=color, width=1.6, dash=dash),
        name=name,
        opacity=0.7,
        hoverinfo="skip",
    )

fig.add_trace(pass_trace(prog_passes, PROG_COL, "Progressive Pass (xT+)"), row=1, col=1)
fig.add_trace(pass_trace(neg_passes,  NEG_COL,  "Negative/Safe Pass (xT−)", dash="dot"), row=1, col=1)

# Arrow annotations for each pass
annotations = []
for _, row in passes_df.iterrows():
    color = PROG_COL if row["xT_Added"] >= 0 else NEG_COL
    annotations.append(dict(
        x=row["End_X"],   y=row["End_Y"],
        ax=row["Start_X"], ay=row["Start_Y"],
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=1.4,
        arrowcolor=color,
    ))

# ─── 7c. PASS ENDPOINT DOTS ──────────────────────────────────
for df_sub, color, sym, name_sfx in [
    (prog_passes, PROG_COL, "arrow-up",   "End (Prog)"),
    (neg_passes,  NEG_COL,  "arrow-down", "End (Neg)"),
]:
    fig.add_trace(go.Scatter(
        x=df_sub["End_X"], y=df_sub["End_Y"],
        mode="markers",
        marker=dict(color=color, size=6, symbol=sym, opacity=0.85,
                    line=dict(color="white", width=0.5)),
        name=name_sfx,
        customdata=df_sub[["Player","xT_Added"]].values,
        hovertemplate="<b>%{customdata[0]}</b><br>xT Added: %{customdata[1]:.4f}<extra></extra>",
    ), row=1, col=1)

# ─── 7d. PLAYER LEADERBOARD (horizontal bar) ─────────────────
bar_colors = [PROG_COL if v >= 0 else NEG_COL for v in player_summary["Total_xT_Added"]]

fig.add_trace(go.Bar(
    y=player_summary["Player"],
    x=player_summary["Total_xT_Added"],
    orientation="h",
    marker=dict(
        color=bar_colors,
        line=dict(color="rgba(255,255,255,0.15)", width=0.8),
        opacity=0.88,
    ),
    text=[f"{v:+.3f}" for v in player_summary["Total_xT_Added"]],
    textposition="outside",
    textfont=dict(color=TEXT_COL, size=11, family=FONT_MAIN),
    customdata=player_summary[["Passes","Avg_xT_Added"]].values,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Total xT Added: <b>%{x:.4f}</b><br>"
        "Passes: %{customdata[0]}<br>"
        "Avg xT/Pass: %{customdata[1]:.4f}"
        "<extra></extra>"
    ),
    name="Total xT Added",
), row=2, col=1)

# ── 8. LAYOUT ─────────────────────────────────────────────────
AXIS_COMMON = dict(
    showgrid=False, zeroline=False,
    showticklabels=False,
    showline=False,
)

fig.update_layout(
    height=920,
    paper_bgcolor=BG_PAGE,
    plot_bgcolor=BG_PITCH,

    title=dict(
        text=(
            "<span style='font-family:IBM Plex Mono; color:#00f5d4; font-size:20px; "
            "letter-spacing:2px;'>◈ EXPECTED THREAT (xT) SPATIAL ENGINE</span>"
            "<br>"
            "<span style='font-family:IBM Plex Mono; color:#8899aa; font-size:12px; "
            "letter-spacing:4px;'>AL AHLY SC — PASS EVALUATION SYSTEM</span>"
        ),
        x=0.5, y=0.99,
        xanchor="center", yanchor="top",
    ),

    font=dict(family=FONT_MAIN, color=TEXT_COL, size=11),

    legend=dict(
        bgcolor="rgba(13,21,32,0.85)",
        bordercolor=ACCENT1,
        borderwidth=1,
        font=dict(size=10, color=TEXT_COL, family=FONT_MAIN),
        x=0.01, y=0.99,
        xanchor="left", yanchor="top",
        tracegroupgap=4,
    ),

    margin=dict(l=20, r=80, t=80, b=20),

    annotations=annotations + [
        # Subtitle annotations for subplots
        dict(
            text="▸ xT GRID + PASS MAP",
            x=0.01, y=0.985, xref="paper", yref="paper",
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=9, color="#556677", family=FONT_MAIN),
        ),
        dict(
            text="▸ PLAYER xT LEADERBOARD",
            x=0.01, y=0.265, xref="paper", yref="paper",
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=9, color="#556677", family=FONT_MAIN),
        ),
        # Pitch direction label
        dict(
            text="← DEFENSIVE THIRD",
            x=0.04, y=0.97, xref="paper", yref="paper",
            xanchor="left", yanchor="top", showarrow=False,
            font=dict(size=8, color="#445566", family=FONT_MAIN),
        ),
        dict(
            text="ATTACKING THIRD →",
            x=0.96, y=0.97, xref="paper", yref="paper",
            xanchor="right", yanchor="top", showarrow=False,
            font=dict(size=8, color=ACCENT1, family=FONT_MAIN),
        ),
    ],
)

# Pitch subplot axes
fig.update_xaxes(
    **AXIS_COMMON,
    range=[-3, PITCH_LEN + 3],
    row=1, col=1,
)
fig.update_yaxes(
    **AXIS_COMMON,
    range=[-3, PITCH_WID + 3],
    scaleanchor="x",
    scaleratio=1,
    row=1, col=1,
)

# Bar chart axes
fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.07)",
    zeroline=True,
    zerolinecolor="rgba(255,255,255,0.2)",
    zerolinewidth=1,
    tickfont=dict(color=TEXT_COL, size=9, family=FONT_MAIN),
    title=dict(text="Total xT Added", font=dict(color="#8899aa", size=10, family=FONT_MAIN)),
    row=2, col=1,
)
fig.update_yaxes(
    showgrid=False,
    tickfont=dict(color=TEXT_COL, size=11, family=FONT_MAIN),
    row=2, col=1,
)

fig.update_layout(
    shapes=pitch_shapes(),
    plot_bgcolor=BG_PITCH,
)

# The bar chart panel needs its own background
fig.update_layout(
    **{"yaxis2": dict(domain=[0.0, 0.25])},
)

# ── 9. HTML ASSEMBLY ──────────────────────────────────────────
plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>xT Spatial Engine — Al Ahly SC</title>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg-page:  {BG_PAGE};
      --bg-card:  {BG_CARD};
      --accent:   {ACCENT1};
      --accent2:  {ACCENT2};
      --text:     {TEXT_COL};
      --prog:     {PROG_COL};
      --neg:      {NEG_COL};
    }}

    body {{
      background: var(--bg-page);
      color: var(--text);
      font-family: "IBM Plex Mono", monospace;
      min-height: 100vh;
      padding: 0;
    }}

    /* ── TOP HEADER ── */
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 32px 14px;
      border-bottom: 1px solid rgba(0,245,212,0.15);
      background: linear-gradient(90deg, rgba(0,245,212,0.04) 0%, transparent 60%);
    }}

    .header-left {{ display: flex; align-items: center; gap: 16px; }}
    .badge {{
      background: var(--accent);
      color: var(--bg-page);
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 2px;
      padding: 4px 10px;
      border-radius: 2px;
    }}
    .title-block h1 {{
      font-size: 18px;
      font-weight: 600;
      letter-spacing: 3px;
      color: var(--accent);
      line-height: 1.2;
    }}
    .title-block h2 {{
      font-size: 10px;
      font-weight: 300;
      letter-spacing: 5px;
      color: #556677;
      margin-top: 2px;
    }}

    .header-meta {{
      text-align: right;
      font-size: 9px;
      color: #445566;
      letter-spacing: 2px;
      line-height: 1.7;
    }}

    /* ── KPI STRIP ── */
    .kpi-strip {{
      display: flex;
      gap: 1px;
      padding: 0 32px;
      margin: 20px 0;
    }}
    .kpi-card {{
      flex: 1;
      background: var(--bg-card);
      border: 1px solid rgba(0,245,212,0.1);
      border-top: 2px solid var(--accent);
      padding: 14px 18px;
      position: relative;
      overflow: hidden;
    }}
    .kpi-card::after {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(135deg, rgba(0,245,212,0.04), transparent);
      pointer-events: none;
    }}
    .kpi-label {{
      font-size: 8px;
      letter-spacing: 3px;
      color: #556677;
      margin-bottom: 6px;
      text-transform: uppercase;
    }}
    .kpi-value {{
      font-size: 26px;
      font-weight: 600;
      color: var(--accent);
      line-height: 1;
    }}
    .kpi-sub {{
      font-size: 9px;
      color: #445566;
      margin-top: 4px;
      letter-spacing: 1px;
    }}

    /* ── CHART CONTAINER ── */
    .chart-wrapper {{
      margin: 0 24px 20px;
      background: var(--bg-card);
      border: 1px solid rgba(0,245,212,0.08);
      border-radius: 4px;
      padding: 8px;
      box-shadow: 0 4px 40px rgba(0,0,0,0.5);
    }}

    /* ── LEGEND STRIP ── */
    .legend-strip {{
      display: flex;
      gap: 24px;
      padding: 12px 32px;
      border-top: 1px solid rgba(0,245,212,0.08);
      font-size: 10px;
      letter-spacing: 2px;
      color: #556677;
    }}
    .leg-item {{ display: flex; align-items: center; gap: 8px; }}
    .leg-dot {{ width: 10px; height: 4px; border-radius: 2px; }}

    /* ── FOOTER ── */
    footer {{
      padding: 14px 32px;
      border-top: 1px solid rgba(0,245,212,0.08);
      font-size: 8px;
      color: #334455;
      letter-spacing: 2px;
      display: flex;
      justify-content: space-between;
    }}
  </style>
</head>
<body>

  <header>
    <div class="header-left">
      <div class="badge">xT ENGINE</div>
      <div class="title-block">
        <h1>EXPECTED THREAT SPATIAL ENGINE</h1>
        <h2>AL AHLY SC — PASS EVALUATION SYSTEM v1.0</h2>
      </div>
    </div>
    <div class="header-meta">
      GRID: {GRID_COLS}×{GRID_ROWS} CELLS<br>
      PITCH: {int(PITCH_LEN)}m × {int(PITCH_WID)}m<br>
      PASSES ANALYSED: {len(passes_df)}<br>
      MODEL: SYNTHETIC xT
    </div>
  </header>

  <!-- KPI STRIP -->
  <div class="kpi-strip">
    <div class="kpi-card">
      <div class="kpi-label">Total Passes Tracked</div>
      <div class="kpi-value">{len(passes_df)}</div>
      <div class="kpi-sub">3 playmakers</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Progressive Passes</div>
      <div class="kpi-value" style="color:var(--prog)">{(passes_df["xT_Added"]>=0).sum()}</div>
      <div class="kpi-sub">xT_Added &ge; 0</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Negative / Safe Passes</div>
      <div class="kpi-value" style="color:var(--neg)">{(passes_df["xT_Added"]<0).sum()}</div>
      <div class="kpi-sub">xT_Added &lt; 0</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Total xT Generated</div>
      <div class="kpi-value">{passes_df["xT_Added"].sum():+.3f}</div>
      <div class="kpi-sub">sum across all passes</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Top Progressive Passer</div>
      <div class="kpi-value" style="font-size:14px; padding-top:4px;">
        {player_summary.sort_values("Total_xT_Added", ascending=False).iloc[0]["Player"].split()[0].upper()}
      </div>
      <div class="kpi-sub">
        {player_summary.sort_values("Total_xT_Added", ascending=False).iloc[0]["Total_xT_Added"]:+.3f} xT added
      </div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Avg xT / Pass</div>
      <div class="kpi-value">{passes_df["xT_Added"].mean():+.4f}</div>
      <div class="kpi-sub">all players</div>
    </div>
  </div>

  <!-- MAIN CHART -->
  <div class="chart-wrapper">
    {plot_html}
  </div>

  <!-- LEGEND -->
  <div class="legend-strip">
    <div class="leg-item">
      <div class="leg-dot" style="background:{PROG_COL};"></div>
      PROGRESSIVE PASS (xT_ADDED &ge; 0)
    </div>
    <div class="leg-item">
      <div class="leg-dot" style="background:{NEG_COL}; border-top:2px dashed {NEG_COL};"></div>
      SAFE / BACKWARD PASS (xT_ADDED &lt; 0)
    </div>
    <div class="leg-item">
      <div class="leg-dot" style="background: linear-gradient(90deg,#000004,#fcffa4);"></div>
      xT INTENSITY (INFERNO SCALE: LOW → HIGH)
    </div>
  </div>

  <footer>
    <span>◈ EXPECTED THREAT MODEL — SYNTHETIC SIMULATION — AL AHLY SC</span>
    <span>BUILT WITH PLOTLY + PYTHON · LEAD FOOTBALL DATA SCIENTIST</span>
  </footer>

</body>
</html>"""

# ── 10. WRITE FILE & DOWNLOAD ─────────────────────────────────
OUTPUT_FILE = "Expected_Threat_xT_Dashboard.html"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(HTML_TEMPLATE)

print("=" * 60)
print("  xT SPATIAL ENGINE — BUILD COMPLETE")
print("=" * 60)
print(f"  Grid     : {GRID_COLS} × {GRID_ROWS} cells ({GRID_COLS * GRID_ROWS} zones)")
print(f"  Passes   : {len(passes_df)}")
print(f"  Prog     : {(passes_df['xT_Added']>=0).sum()}")
print(f"  Negative : {(passes_df['xT_Added']<0).sum()}")
print()
print("  PLAYER LEADERBOARD:")
for _, row in player_summary.sort_values("Total_xT_Added", ascending=False).iterrows():
    bar = "█" * max(1, int(abs(row["Total_xT_Added"]) * 80))
    sign = "+" if row["Total_xT_Added"] >= 0 else ""
    print(f"  {row['Player']:<22} {sign}{row['Total_xT_Added']:+.3f}  {bar}")
print()
print(f"  Output   : {OUTPUT_FILE}")
print("=" * 60)

if IN_COLAB:
    files.download(OUTPUT_FILE)
    print("  ✓ Download triggered.")
else:
    print(f"  ✓ File saved locally: {OUTPUT_FILE}")
