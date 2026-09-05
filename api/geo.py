"""Offline US ZIP-code -> (lat, lng) lookup.

Data source: GeoNames postal-codes (public domain / CC-BY 4.0).
The bundled CSV ships ~41 k US ZIPs in api/data/us_zip_codes.csv.
ZIP_DATA_PATH can be overridden; if the file is missing the table is
empty and every lookup returns None (graceful degradation).
"""
import csv
import os

_ZIP_PATH = os.environ.get("ZIP_DATA_PATH", "/app/data/us_zip_codes.csv")

_ZIPS: dict[str, tuple[str, str, float, float]] = {}


def _load() -> None:
    try:
        with open(_ZIP_PATH, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                code = row["zip"]
                _ZIPS[code] = (
                    row["place"],
                    row["state"],
                    float(row["lat"]),
                    float(row["lng"]),
                )
    except FileNotFoundError:
        pass  # empty table; /zip will 404 until the CSV is provided


_load()


def lookup(code: str) -> dict | None:
    """Return ``{"zip", "place", "state", "lat", "lng"}`` or None."""
    entry = _ZIPS.get(code)
    if entry is None:
        return None
    place, state, lat, lng = entry
    return {"zip": code, "place": place, "state": state, "lat": lat, "lng": lng}
