"""Synthesise the ground that covers the planet between the peaks.

The texture is built from the real thing rather than painted:
  1. read the Sentinel-2 pass used for the peaks (23 August 2024) over two stretches of the Bow Valley floor,
  2. cut 640 m patches of forest and meadow, keeping only low, gentle, fully vegetated ground with no roads,
     cutlines, streams or haze,
  3. lay smooth forest-stand and meadow colours (sampled from the valley floors inside the peak tiles) on a
     seamless canvas and stamp the patches' fine grain over it with feathered edges, wrapping at the borders,
  4. grade it like the peak imagery and write it into the `const GROUND = ...; // @ground` line of index.html.

Imagery: contains modified Copernicus Sentinel data (2024).

    pip install numpy scipy pillow rasterio pyproj
    python tools/export_ground.py      # run tools/export_imagery.py first; this reuses its elevation tiles
"""
import base64, io, json, os

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter, minimum_filter, sobel, uniform_filter

import export_imagery as ei

S2 = '/vsicurl/https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/11/U/'
# Valley floor between Canmore and Banff, and between Banff and Castle Junction: (west, north) in UTM 11N, 30 km square.
AREAS = {'PS': (605000, 5675000), 'NS': (568000, 5700000)}
SIZE, P, STRIDE = 3000, 64, 24      # source window and patch size in 10 m pixels
N = 2048                            # output texture, about 21 km across
LO = np.array([9.4, 17.1, 11.2])    # black point of the peak imagery's grading, so the two match
# Raw Sentinel colours of the valley floors inside the peak tiles: dense old forest, younger forest, grass, dry grass.
# Nudged a little lighter and greener, since the curved planet catches less light than the tiles' valley floors.
DARK, MID, GRASS, DRY = (np.array(c, float) for c in ([22, 32, 20], [36, 49, 32], [48, 61, 37], [63, 68, 45]))


def read_area(sq, west, north):
    os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    base = f'{S2}{sq}/2024/8/S2B_11U{sq}_20240823_0_L2A/'
    bands = {}
    for name in ('TCI', 'B04', 'B08', 'SCL'):
        with rasterio.open(base + name + '.tif') as ds:
            c0, r0 = ~ds.transform * (west, north)
            k = 10 / ds.transform.a
            bands[name] = ds.read(window=Window(int(c0), int(r0), int(SIZE / k), int(SIZE / k)),
                                  out_shape=(ds.count, SIZE, SIZE), resampling=Resampling.nearest)
    return bands


def terrain(west, north):
    """Elevation and slope on a 30 m grid, upsampled to the 10 m source grid."""
    g = np.arange(0, SIZE, 3)
    E, Nn = np.meshgrid(west + g * 10 + 5, north - g * 10 - 5)
    lon, lat = Transformer.from_crs(32611, 4326, always_xy=True).transform(E, Nn)
    dem = ei.dem_at(lat, lon)
    gy, gx = np.gradient(dem, 30)
    up = lambda a: np.kron(a, np.ones((3, 3)))[:SIZE, :SIZE]
    return up(dem), up(np.degrees(np.arctan(np.hypot(gx, gy))))


def detail(a):
    """A patch's luminance as a ratio to its local mean: keeps the grain of the canopy, drops its colour."""
    l = a.astype(float).mean(-1) + 4
    return l / gaussian_filter(l, 8, mode='reflect')


def cut_patches():
    keep = []
    for sq, (west, north) in AREAS.items():
        b = read_area(sq, west, north)
        dem, slope = terrain(west, north)
        b4, b8 = b['B04'][0].astype(float), b['B08'][0].astype(float)
        ndvi = (b8 - b4) / np.maximum(b8 + b4, 1)
        tci = b['TCI'].transpose(1, 2, 0)
        ok = np.isin(b['SCL'][0], [4, 5]) & (slope < 20) & (dem < 1900) & (ndvi > 0.3)   # vegetation or bare soil
        whole = uniform_filter(ok.astype(float), P) > 0.96
        nd_min = minimum_filter(ndvi, 5)
        h = P // 2
        for i in range(h, SIZE - h, STRIDE):
            for j in range(h, SIZE - h, STRIDE):
                if not whole[i, j] or (nd_min[i - h:i + h, j - h:j + h] < 0.2).any(): continue
                a = tci[i - h:i + h, j - h:j + h].astype(float)
                l = a.mean(-1)
                m = a.reshape(-1, 3).mean(0)
                if (l > np.median(l) * 1.6 + 6).mean() > 0.004: continue                    # roads, gravel, buildings
                if (np.hypot(sobel(l, 0), sobel(l, 1)) > 40).mean() > 0.01: continue          # cutlines, streams, edges
                if m[1] < m[2] * 1.08: continue                                                # haze or water
                keep.append((a, l.mean()))
    cut = np.percentile([k[1] for k in keep], 65)
    forest = [detail(a) for a, l in keep if l <= cut]
    meadow = [detail(a) for a, l in keep if l > cut]
    print(f'{len(forest)} forest and {len(meadow)} meadow patches')
    return forest, meadow


def noise(scale, seed):
    """Smooth noise that wraps at the borders (white noise low-passed on a torus)."""
    r = np.random.default_rng(seed).standard_normal((N, N))
    f = np.fft.fftfreq(N)[:, None] ** 2 + np.fft.fftfreq(N)[None, :] ** 2
    x = np.real(np.fft.ifft2(np.fft.fft2(r) * np.exp(-f * (N / scale) ** 2 * 2)))
    return (x - x.mean()) / x.std()


def synthesise(forest, meadow_patches):
    rng = np.random.default_rng(7)
    smooth = lambda x: x * x * (3 - 2 * x)
    macro = noise(6, 1) * 0.8 + noise(14, 2) * 0.45 + noise(40, 3) * 0.25 + noise(60, 8) * 0.35 + noise(150, 9) * 0.2
    meadow = smooth(np.clip((macro - 1.05) / 0.6, 0, 1))
    stand = np.clip(0.5 + 0.22 * noise(10, 4) + 0.2 * noise(30, 5) + 0.15 * noise(90, 6), 0, 1)
    dry = np.clip(0.5 + 0.5 * noise(20, 7), 0, 1)[..., None]
    base = DARK + (MID - DARK) * stand[..., None]
    base += (GRASS * (1 - dry) + DRY * dry - base) * meadow[..., None]

    grain = np.ones((N, N))
    for i in range(0, N, P):                      # an even first layer so nothing is left bare
        for j in range(0, N, P):
            grain[i:i + P, j:j + P] = forest[rng.integers(len(forest))]
    yy, xx = np.mgrid[0:P, 0:P]
    fe = smooth(np.clip(np.minimum.reduce([yy, xx, P - 1 - yy, P - 1 - xx]) / 14, 0, 1))
    spots = [(i, j) for i in range(0, N, 32) for j in range(0, N, 32)] + [tuple(x) for x in rng.integers(0, N, (1500, 2))]
    rng.shuffle(spots)
    for i, j in spots:
        i, j = int(i + rng.integers(-12, 13)) % N, int(j + rng.integers(-12, 13)) % N
        src = meadow_patches if meadow[(i + P // 2) % N, (j + P // 2) % N] > 0.5 else forest
        a = np.rot90(src[rng.integers(len(src))], rng.integers(4))
        if rng.random() < 0.5: a = a[:, ::-1]
        ix = np.ix_((np.arange(P) + i) % N, (np.arange(P) + j) % N)
        grain[ix] = grain[ix] * (1 - fe) + a * fe
    return base * np.clip(1 + (grain - 1) * 1.9, 0.35, 2.4)[..., None]


def encode(raw):
    x = np.clip((raw - LO) / (255 - LO.mean()), 0, 1) ** 0.65
    l = x.mean(-1, keepdims=True)
    x = np.clip(l + (x - l) * 1.18, 0, 1)
    buf = io.BytesIO()
    Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(buf, 'JPEG', quality=80, optimize=True, progressive=True)
    print(f'ground: {len(buf.getvalue()) // 1024} KB')
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def main():
    ei.DEM = ei.load_dem()
    ground = encode(synthesise(*cut_patches()))
    lines = ei.PAGE.read_text().split('\n')
    i = next(n for n, l in enumerate(lines) if l.startswith('const GROUND = '))
    lines[i] = 'const GROUND = ' + json.dumps(ground) + '; // @ground'
    ei.PAGE.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
