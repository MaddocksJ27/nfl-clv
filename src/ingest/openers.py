"""Fetch and parse historical NFL open/close odds from sportsbookreviewsonline.com.

Fetch and store only — no feature engineering here.

Site quirks handled defensively (verified against seasons 2016-17..2021-22):
- The site blocks requests without a browser-like User-Agent (returns 404).
- Each season's odds are rendered as one big HTML table (not the .xlsx files the
  archive originally shipped) with two rows per game: away team then home team,
  or "N"/"N" for a small number of neutral-site games (row order is still
  away-then-home in that case).
- The Open/Close columns don't separate spread from total. For a given column,
  whichever of the two rows has the LARGER magnitude is the total; the smaller
  is the point spread, attributed to whichever team's row it's printed on
  (that team was favoured by that many points). Almost always given as an
  unsigned magnitude, occasionally with an explicit minus sign — either way
  home_spread ends up negative when the home row carries the smaller-magnitude
  number, positive otherwise, 0 for a pick'em ("pk").
- A "pk" close is sometimes a missing-value placeholder rather than a true
  even line — cross-checked against the moneyline (see PK_MONEYLINE_SUSPECT_THRESHOLD).
- Team name spellings drift within and across seasons (e.g. "TampaBay" vs
  "Tampa", "LasVegas" vs "LVRaiders", a "Washingtom" typo) — normalized via
  TEAM_NAME_MAP.
"""

import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INDEX_URL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl/nfloddsarchives.htm"
SEASON_LINK_RE = re.compile(r"/scoresoddsarchives/nfl-odds-(\d{4})-(\d{2})/?$")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = 1.0

MIN_SEASON = 2016

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "openers"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
OUTPUT_PATH = INTERIM_DIR / "openers.parquet"

# Plausibility bounds for the max=total/min=spread heuristic below.
MIN_PLAUSIBLE_TOTAL = 28.0
MAX_PLAUSIBLE_SPREAD = 25.0

# A "pk" (pick'em, i.e. 0-point spread) is sometimes used by the source as a
# missing-value placeholder rather than a literal even line. If the moneyline
# on either side is more decisive than this, treat the pk as unreliable.
PK_MONEYLINE_SUSPECT_THRESHOLD = 120.0

TEAM_NAME_MAP = {
    "Arizona": "ARI",
    "Atlanta": "ATL",
    "Baltimore": "BAL",
    "Buffalo": "BUF",
    "Carolina": "CAR",
    "Chicago": "CHI",
    "Cincinnati": "CIN",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "GreenBay": "GB",
    "Houston": "HOU",
    "Indianapolis": "IND",
    "Jacksonville": "JAX",
    "Kansas": "KC",
    "KansasCity": "KC",
    "KCChiefs": "KC",
    "LAChargers": "LAC",
    "LARams": "LA",
    "LasVegas": "LV",
    "LVRaiders": "LV",
    "LosAngeles": "LA",
    "Miami": "MIA",
    "Minnesota": "MIN",
    "NewEngland": "NE",
    "NewOrleans": "NO",
    "NYGiants": "NYG",
    "NYJets": "NYJ",
    "Oakland": "OAK",
    "Philadelphia": "PHI",
    "Pittsburgh": "PIT",
    "SanDiego": "SD",
    "SanFrancisco": "SF",
    "Seattle": "SEA",
    "Tampa": "TB",
    "TampaBay": "TB",
    "Tennessee": "TEN",
    "Washington": "WAS",
    "Washingtom": "WAS",  # typo present in the source HTML
}

REQUIRED_COLUMNS = {"VH", "Team", "Open", "Close"}


class SeasonFetchError(Exception):
    pass


def _get(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _cache_path(name: str) -> Path:
    return RAW_DIR / name


def _fetch_cached(url: str, cache_name: str, force_refresh: bool = False) -> str:
    path = _cache_path(cache_name)
    if path.exists() and not force_refresh:
        return path.read_text(encoding="utf-8")

    html = _get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    time.sleep(REQUEST_DELAY_SECONDS)
    return html


def discover_seasons(force_refresh: bool = False) -> dict:
    """Return {season_start_year: season_page_url} for seasons >= MIN_SEASON."""
    html = _fetch_cached(INDEX_URL, "index.htm", force_refresh)
    soup = BeautifulSoup(html, "html.parser")

    seasons = {}
    for a in soup.find_all("a", href=True):
        match = SEASON_LINK_RE.search(a["href"])
        if not match:
            continue
        start_year = int(match.group(1))
        if start_year < MIN_SEASON:
            continue
        seasons[start_year] = urljoin(INDEX_URL, a["href"])

    if not seasons:
        raise SeasonFetchError(f"No season links found on index page {INDEX_URL}")

    return dict(sorted(seasons.items()))


def _parse_num(raw: str):
    raw = raw.strip()
    if raw == "" or raw.upper() == "NL":
        return None
    if raw.lower() == "pk":
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return None


def _resolve_date(date_token: str, season: int, source_url: str, row_idx: int):
    date_token = date_token.strip()
    if len(date_token) == 3:
        month, day = int(date_token[0]), int(date_token[1:])
    elif len(date_token) == 4:
        month, day = int(date_token[:2]), int(date_token[2:])
    else:
        logger.warning(
            "%s: row %d has unparseable date token %r, skipping game",
            source_url, row_idx, date_token,
        )
        return None

    year = season + 1 if month <= 6 else season
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        logger.warning(
            "%s: row %d has invalid calendar date %r (season %d), skipping game",
            source_url, row_idx, date_token, season,
        )
        return None


def _team_code(raw_name: str, source_url: str, row_idx: int):
    code = TEAM_NAME_MAP.get(raw_name)
    if code is None:
        logger.warning(
            "%s: row %d has unmapped team name %r, skipping game",
            source_url, row_idx, raw_name,
        )
    return code


def _spread_and_total(
    away_val, home_val, label: str, source_url: str, row_idx: int,
    ml_away=None, ml_home=None,
):
    """Split a (away, home) pair of raw numbers into (home_spread, total).

    Larger magnitude = total, smaller magnitude = spread, attributed to
    whichever side it was printed on (that side is favoured by that many
    points). Almost all rows give the spread as an unsigned magnitude, but a
    handful explicitly sign the favourite's number (e.g. "-6") — abs() makes
    both forms compare correctly, and the sign is re-derived from which side
    won the magnitude comparison rather than trusted directly.

    ml_away/ml_home (only meaningful for the closing line, since opening
    moneylines aren't tracked in this dataset) let us catch "pk" entries that
    are actually a missing-value placeholder rather than a true even line.
    """
    if away_val is None or home_val is None:
        return None, None

    away_mag, home_mag = abs(away_val), abs(home_val)

    if away_mag <= home_mag:
        spread_mag, total = away_mag, home_mag
        home_spread = spread_mag  # away favoured -> home is the underdog (+)
    else:
        spread_mag, total = home_mag, away_mag
        home_spread = -spread_mag  # home favoured -> negative

    if spread_mag == 0:
        home_spread = 0.0
        worst_ml = max(abs(ml_away) if ml_away is not None else 0.0,
                        abs(ml_home) if ml_home is not None else 0.0)
        if worst_ml > PK_MONEYLINE_SUSPECT_THRESHOLD:
            logger.warning(
                "%s: row %d %s spread reads pick'em but moneyline (away=%s, "
                "home=%s) implies a real favourite — treating spread as missing",
                source_url, row_idx, label, ml_away, ml_home,
            )
            home_spread = None

    if total < MIN_PLAUSIBLE_TOTAL or spread_mag > MAX_PLAUSIBLE_SPREAD:
        logger.warning(
            "%s: row %d %s values (away=%s, home=%s) failed plausibility check "
            "(total=%s, spread=%s) — dropping both",
            source_url, row_idx, label, away_val, home_val, total, spread_mag,
        )
        return None, None

    return home_spread, total


def _parse_season_table(html: str, season: int, source_url: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")

    table = None
    for candidate in soup.find_all("table"):
        header_row = candidate.find("tr")
        if header_row is None:
            continue
        headers = {c.get_text(strip=True) for c in header_row.find_all(["th", "td"])}
        if REQUIRED_COLUMNS.issubset(headers):
            table = candidate
            break

    if table is None:
        raise SeasonFetchError(f"{source_url}: no table with required columns {REQUIRED_COLUMNS} found")

    rows = table.find_all("tr")
    header_cells = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
    col_idx = {name: i for i, name in enumerate(header_cells)}
    for required in REQUIRED_COLUMNS | {"Date"}:
        if required not in col_idx:
            raise SeasonFetchError(f"{source_url}: missing required column {required!r}")

    data_rows = rows[1:]
    if len(data_rows) % 2 != 0:
        logger.warning(
            "%s: odd number of data rows (%d), dropping the last unpaired row",
            source_url, len(data_rows),
        )
        data_rows = data_rows[:-1]

    records = []
    for i in range(0, len(data_rows), 2):
        row_a_cells = [c.get_text(strip=True) for c in data_rows[i].find_all(["td", "th"])]
        row_b_cells = [c.get_text(strip=True) for c in data_rows[i + 1].find_all(["td", "th"])]

        if len(row_a_cells) != len(header_cells) or len(row_b_cells) != len(header_cells):
            logger.warning(
                "%s: rows %d-%d have unexpected cell counts (%d, %d), skipping game",
                source_url, i, i + 1, len(row_a_cells), len(row_b_cells),
            )
            continue

        vh_a, vh_b = row_a_cells[col_idx["VH"]], row_b_cells[col_idx["VH"]]
        if vh_a == "H" and vh_b == "V":
            row_a_cells, row_b_cells = row_b_cells, row_a_cells
        elif not ((vh_a == "V" and vh_b == "H") or (vh_a == "N" and vh_b == "N")):
            logger.warning(
                "%s: rows %d-%d have unexpected VH pattern (%r, %r), skipping game",
                source_url, i, i + 1, vh_a, vh_b,
            )
            continue

        neutral_site = vh_a == "N"

        away_team = _team_code(row_a_cells[col_idx["Team"]], source_url, i)
        home_team = _team_code(row_b_cells[col_idx["Team"]], source_url, i)
        if away_team is None or home_team is None:
            continue

        game_date = _resolve_date(row_a_cells[col_idx["Date"]], season, source_url, i)
        if game_date is None:
            continue

        away_open = _parse_num(row_a_cells[col_idx["Open"]])
        home_open = _parse_num(row_b_cells[col_idx["Open"]])
        away_close = _parse_num(row_a_cells[col_idx["Close"]])
        home_close = _parse_num(row_b_cells[col_idx["Close"]])

        ml_away = ml_home = None
        if "ML" in col_idx:
            ml_away = _parse_num(row_a_cells[col_idx["ML"]])
            ml_home = _parse_num(row_b_cells[col_idx["ML"]])

        open_spread, open_total = _spread_and_total(away_open, home_open, "open", source_url, i)
        close_spread, close_total = _spread_and_total(
            away_close, home_close, "close", source_url, i, ml_away=ml_away, ml_home=ml_home,
        )

        records.append({
            "season": season,
            "date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "neutral_site": neutral_site,
            "open_spread": open_spread,
            "close_spread": close_spread,
            "open_total": open_total,
            "close_total": close_total,
        })

    return pd.DataFrame.from_records(records)


def fetch_season(season: int, url: str, force_refresh: bool = False) -> pd.DataFrame:
    cache_name = f"nfl_odds_{season}-{str(season + 1)[-2:]}.htm"
    html = _fetch_cached(url, cache_name, force_refresh)
    return _parse_season_table(html, season, url)


def fetch_all_seasons(force_refresh: bool = False) -> tuple:
    """Fetch every discoverable season >= MIN_SEASON.

    Returns (combined DataFrame, report dict of season -> {status, rows, error}).
    """
    seasons = discover_seasons(force_refresh)
    report = {}
    frames = []

    for season, url in seasons.items():
        try:
            df = fetch_season(season, url, force_refresh)
        except (requests.RequestException, SeasonFetchError) as exc:
            logger.error("Season %d failed: %s", season, exc)
            report[season] = {"status": "failed", "rows": 0, "error": str(exc)}
            continue

        report[season] = {"status": "ok", "rows": len(df), "error": None}
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, report


def _print_report(report: dict) -> None:
    ok_seasons = sorted(s for s, r in report.items() if r["status"] == "ok")
    failed_seasons = sorted(s for s, r in report.items() if r["status"] == "failed")

    print("\nOpeners ingestion report")
    print("=" * 40)
    for season in sorted(report):
        r = report[season]
        if r["status"] == "ok":
            print(f"  {season}: OK, {r['rows']} games")
        else:
            print(f"  {season}: FAILED — {r['error']}")

    print(f"\nSucceeded: {ok_seasons}")
    print(f"Failed:    {failed_seasons}")
    print(f"Latest season available: {max(ok_seasons) if ok_seasons else 'none'}")
    print("=" * 40)


def main(force_refresh: bool = False) -> pd.DataFrame:
    combined, report = fetch_all_seasons(force_refresh)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(combined), OUTPUT_PATH)

    _print_report(report)
    return combined


if __name__ == "__main__":
    main()
