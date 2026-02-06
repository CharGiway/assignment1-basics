import argparse
import json
import math
from pathlib import Path


def _read_log(p: Path):
    steps = []
    val = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            s = obj.get("step")
            if s is None:
                continue
            steps.append(int(s))
            v = obj.get("val_loss")
            val.append(float(v) if v is not None else math.nan)
    return steps, val


def _scale(v, vmin, vmax, out_min, out_max):
    if vmax == vmin:
        return (out_min + out_max) / 2.0
    t = (v - vmin) / (vmax - vmin)
    return out_min + t * (out_max - out_min)


def _polyline(points, color, width=2):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}" />\n'


def _text(x, y, s, size=12, color="#000"):
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}">{s}</text>\n'


def _line(x1, y1, x2, y2, color="#000", width=1):
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}" />\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=str, nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--title", type=str, default="Batch Size Overlay (val_loss)")
    args = ap.parse_args()

    series = []
    for item in args.logs:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            label, path = Path(item).stem, item
        steps, vals = _read_log(Path(path))
        series.append((label, steps, vals))
    if not series:
        Path(args.out).write_text("", encoding="utf-8")
        return

    w = 1200
    h = 700
    m_left = 80
    m_right = 20
    m_top = 60
    m_bottom = 80
    gx0 = m_left
    gy0 = m_top
    gx1 = w - m_right
    gy1 = h - m_bottom

    x_min = min(min(s for s in steps if not math.isnan(s)) for _, steps, _ in series)
    x_max = max(max(s for s in steps if not math.isnan(s)) for _, steps, _ in series)
    ys = []
    for _, _, vals in series:
        ys.extend([v for v in vals if not math.isnan(v)])
    y_min = min(ys) if ys else 0.0
    y_max = max(ys) if ys else 1.0
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
    y_min -= y_pad
    y_max += y_pad

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">\n')
    svg.append(_text(w / 2, m_top / 2 + 10, args.title, size=18))
    svg.append(_line(gx0, gy0, gx1, gy0, color="#ccc"))
    svg.append(_line(gx0, gy1, gx1, gy1, color="#ccc"))
    svg.append(_line(gx0, gy0, gx0, gy1, color="#ccc"))
    svg.append(_line(gx1, gy0, gx1, gy1, color="#ccc"))

    xticks = 10
    for i in range(xticks + 1):
        xv = x_min + (x_max - x_min) * i / xticks
        x = _scale(xv, x_min, x_max, gx0, gx1)
        svg.append(_line(x, gy0, x, gy1, color="#eee"))
        svg.append(_text(x, gy1 + 20, f"{int(xv)}", size=12))
    yticks = 8
    for i in range(yticks + 1):
        yv = y_min + (y_max - y_min) * i / yticks
        y = _scale(yv, y_min, y_max, gy1, gy0)
        svg.append(_line(gx0, y, gx1, y, color="#eee"))
        svg.append(_text(gx0 - 50, y + 4, f"{yv:.2f}", size=12))

    for idx, (label, steps, vals) in enumerate(series):
        pts = []
        for s, v in zip(steps, vals):
            if math.isnan(v):
                continue
            x = _scale(s, x_min, x_max, gx0, gx1)
            y = _scale(v, y_min, y_max, gy1, gy0)
            pts.append((x, y))
        if pts:
            svg.append(_polyline(pts, colors[idx % len(colors)], width=2.5))

    lx = gx0 + 20
    ly = gy0 + 20
    for idx, (label, _, _) in enumerate(series):
        svg.append(_line(lx, ly, lx + 30, ly, color=colors[idx % len(colors)], width=3))
        svg.append(_text(lx + 40, ly + 4, label, size=14, color=colors[idx % len(colors)]))
        ly += 24

    svg.append("</svg>\n")
    Path(args.out).write_text("".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
