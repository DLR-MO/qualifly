# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

"""
Draws the source positions (e.g. onboard records) of all flights in the data
folder as transparent red lines on an OpenStreetMap map of Germany and saves it
as a PDF.

The source is the one of config.yaml; its positions are read from
data/<flight>/<source>.csv, written by run.py.

    python scripts/plot_flight_map.py [--step 100] [--zoom 7] [--out data/flight_map.pdf]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Germany plus a margin (lon_min, lon_max, lat_min, lat_max): covers nearly
# all flights, only the far outliers (e.g. over Greece or the Baltics) fall outside
GERMANY = (4.0, 17.5, 45.5, 56.5)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=int, default=100, help="Draw every n-th position (default 100)")
    parser.add_argument("--zoom", type=int, default=7, help="Zoom level of the map tiles (default 7)")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="Data folder (default data/)")
    parser.add_argument("--out", type=Path, help="Output PDF (default <data>/flight_map.pdf)")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step must be >= 1")

    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from run import load_config
    from qualifly_lib.data_fetching.data_fetcher import load_config as load_columns

    config = load_config()
    source = load_columns(config["source"], config["target"])["source"]
    files = sorted(args.data.glob(f"*/{source.IDENTIFIER}.csv"))
    if not files:
        sys.exit(f"No {source.IDENTIFIER}.csv in the flight folders of {args.data}.")

    class WashedOutOSM(cimgt.OSM):
        """OSM tiles with the colours pulled toward grey so the flights stand out."""

        def image_for_domain(self, target_domain, target_z):
            from PIL import Image, ImageEnhance

            img, extent, origin = super().image_for_domain(target_domain, target_z)
            img = ImageEnhance.Color(Image.fromarray(img)).enhance(0.35)
            return np.asarray(img), extent, origin

    tiles = WashedOutOSM()
    fig = plt.figure(figsize=(8.3, 11.7))  # A4 portrait
    ax = fig.add_subplot(projection=tiles.crs)
    ax.set_extent(GERMANY, crs=ccrs.PlateCarree())
    ax.add_image(tiles, args.zoom)

    for file in files:
        positions = pd.read_csv(file, usecols=[source.LAT, source.LON]).dropna().iloc[::args.step]
        ax.plot(positions[source.LON], positions[source.LAT], color="red", alpha=0.4, linewidth=0.6,
                transform=ccrs.PlateCarree())

    ax.text(0.995, 0.005, "\u00a9 OpenStreetMap contributors", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, backgroundcolor="white")
    out = args.out or args.data / "flight_map.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
