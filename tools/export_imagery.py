"""Drape Sentinel-2 satellite imagery over the terrain tiles in index.html.

For each peak in TERRAIN this script
  1. finds where the tile sits on the ground (centre, rotation) by matching its heights against
     the AWS Terrain Tiles elevation model,
  2. reads the Sentinel-2 L2A true-colour image (TCI, 10 m) for that footprint from the public
     sentinel-cogs bucket on AWS,
  3. resamples it onto the tile's own grid and grades it,
  4. writes the result into the `const IMAGERY = ...; // @imagery` line of index.html.
The page fades each tile's rim into the planet's ground itself.

Imagery: contains modified Copernicus Sentinel data (2024).

    pip install numpy scipy pillow rasterio pyproj
    python tools/export_imagery.py
"""
import base64, io, json, math, os, urllib.request
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.windows import Window
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

PAGE = Path(__file__).resolve().parent.parent / 'index.html'
CACHE = Path(__file__).resolve().parent / '.cache'
S2 = 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/11/U/'
# One satellite pass (23 August 2024): clear sky, no smoke, little seasonal snow. Two MGRS squares cover the five peaks.
SCENE = {'yamnuska': 'PS', 'sisters': 'PS', 'rundle': 'PS', 'cascade': 'NS', 'castle': 'NS'}
DATE = '20240823'
# Rough summit positions, used only as a starting point for the fit.
GUESS = {'yamnuska': (51.1203, -115.1217), 'sisters': (51.0189, -115.3364), 'rundle': (51.1286, -115.4708),
         'cascade': (51.2203, -115.5536), 'castle': (51.2647, -115.9264)}
OUT = 768                 # texture size in pixels (the tiles are 7.5-9 km, so about 10 m a pixel)


def load_terrain():
    line = next(l for l in PAGE.read_text().split('\n') if l.startswith('const TERRAIN = '))
    t, _ = json.JSONDecoder().raw_decode(line[len('const TERRAIN = '):])
    for v in t.values():
        S = v['size']
        v['elev'] = np.frombuffer(base64.b64decode(v['h']), '<u2').reshape(S, S).astype(float)
    return t


# ---------- Elevation reference: AWS Terrain Tiles (terrarium encoding), zoom 13
Z = 13

def _tile(x, y):
    fn = CACHE / f'terrarium_{Z}_{x}_{y}.png'
    if not fn.exists():
        CACHE.mkdir(exist_ok=True)
        urllib.request.urlretrieve(f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{x}/{y}.png', fn)
    a = np.asarray(Image.open(fn).convert('RGB')).astype(float)
    return a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768

def _merc(lat, lon):
    n = 2 ** Z
    return (lon + 180) / 360 * n, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n

X0, Y0 = (int(c) for c in _merc(51.45, -116.1))
X1, Y1 = (int(c) for c in _merc(50.95, -115.0))
DEM = None

def load_dem():
    return np.vstack([np.hstack([_tile(x, y) for x in range(X0, X1 + 1)]) for y in range(Y0, Y1 + 1)])

def dem_at(lat, lon):
    x, y = _merc(lat, lon)
    return map_coordinates(DEM, [(y - Y0) * 256 - 0.5, (x - X0) * 256 - 0.5], order=1)


def tile_grid(lat, lon, bearing, extent, n):
    """Lat/lon of an n x n grid laid out like the page's tiles: columns run left to right,
    rows from the far side to the viewer, and `bearing` is the direction the viewer faces."""
    g = (np.arange(n) / (n - 1) - 0.5) * extent
    u, v = np.meshgrid(g, g)
    b = np.radians(bearing)
    east = u * np.cos(b) - v * np.sin(b)
    north = -u * np.sin(b) - v * np.cos(b)
    return lat + north / 111320, lon + east / (111320 * np.cos(np.radians(lat)))


def locate(t, key):
    """Fit the tile's centre and facing by matching its heights to the elevation model."""
    e, ext = t['elev'], t['extent']
    small, la0, lo0 = e[::4, ::4], *GUESS[key]
    best = (1e9,)
    for b in range(0, 360, 5):
        for dn in range(-5000, 5001, 500):
            for de in range(-5000, 5001, 500):
                la, lo = la0 + dn / 111320, lo0 + de / (111320 * math.cos(math.radians(la0)))
                r = np.sqrt(np.mean((dem_at(*tile_grid(la, lo, b, ext, small.shape[0])) - small) ** 2))
                if r < best[0]: best = (r, la, lo, b)
    _, la, lo, b = best
    f = lambda q: np.sqrt(np.mean((dem_at(*tile_grid(q[0], q[1], q[2], ext, e.shape[0])) - e) ** 2))
    res = minimize(f, [la, lo, b], method='Nelder-Mead',
                   options=dict(xatol=1e-7, fatol=1e-3, maxiter=3000,
                                initial_simplex=[[la, lo, b], [la + 0.002, lo, b], [la, lo + 0.003, b], [la, lo, b + 3]]))
    print(f'{key}: centre {res.x[0]:.5f}, {res.x[1]:.5f}  facing {res.x[2] % 360:.1f}°  height RMSE {res.fun:.1f} m')
    return res.x


def read_imagery(key, lat, lon, bearing, extent):
    os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    sq = SCENE[key]
    url = f'/vsicurl/{S2}{sq}/{DATE[:4]}/{int(DATE[4:6])}/S2B_11U{sq}_{DATE}_0_L2A/TCI.tif'
    glat, glon = tile_grid(lat, lon, bearing, extent, OUT)
    E, N = Transformer.from_crs(4326, 32611, always_xy=True).transform(glon, glat)
    with rasterio.open(url) as ds:
        c, r = ~ds.transform * (E, N)
        c0, r0 = int(c.min()) - 3, int(r.min()) - 3
        a = ds.read(window=Window(c0, r0, int(c.max()) - c0 + 6, int(r.max()) - r0 + 6), boundless=True, fill_value=0)
    return np.stack([map_coordinates(a[b].astype(float), [r - r0 - 0.5, c - c0 - 0.5], order=3) for b in range(3)], -1)


def finish(raw):
    """Grade every tile the same way and encode it."""
    lo = np.percentile(np.concatenate([a.reshape(-1, 3) for a in raw.values()]), 0.2, axis=0)
    out = {}
    for key, a in raw.items():
        x = np.clip((a - lo) / (255 - lo.mean()), 0, 1) ** 0.65           # lift the shadows; TCI is dark
        l = x.mean(-1, keepdims=True); x = np.clip(l + (x - l) * 1.18, 0, 1)
        buf = io.BytesIO()
        Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(buf, 'JPEG', quality=82, optimize=True, progressive=True)
        out[key] = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
        print(f'{key}: {len(buf.getvalue()) // 1024} KB')
    return out


def main():
    global DEM
    DEM = load_dem()
    t = load_terrain()
    raw = {}
    for key, v in t.items():
        lat, lon, bearing = locate(v, key)
        raw[key] = read_imagery(key, lat, lon, bearing, v['extent'])
    imagery = finish(raw)
    lines = PAGE.read_text().split('\n')
    i = next(n for n, l in enumerate(lines) if l.startswith('const IMAGERY = '))
    lines[i] = 'const IMAGERY = ' + json.dumps(imagery, separators=(',', ':')) + '; // @imagery'
    PAGE.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
