"""Map odds-book full team names (e.g. "Carolina Panthers") to nflverse
team abbreviations (e.g. "CAR"). Used to join odds-API events against
nflverse schedules/rosters.
"""

import nfl_data_py as nfl

# nflverse's team_desc includes retired/legacy abbreviations for relocated
# franchises (LAR, OAK, SD, STL) alongside the current ones (LA, LV, LAC) —
# drop the legacy rows so each full team name maps to exactly one abbreviation.
_LEGACY_ABBRS = {"LAR", "OAK", "SD", "STL"}


# Historical franchise names odds books use for older seasons that
# nflverse's team_desc doesn't carry (it only has the current name).
_HISTORICAL_ALIASES = {
    "Washington Football Team": "WAS",  # used 2020-2021, before "Commanders"
}


def team_name_map() -> dict:
    """Return {full_team_name: nflverse_abbr} for all 32 current teams,
    plus historical name aliases for older seasons."""
    td = nfl.import_team_desc()
    td = td[~td.team_abbr.isin(_LEGACY_ABBRS)]
    mapping = dict(zip(td.team_name, td.team_abbr))
    mapping.update(_HISTORICAL_ALIASES)
    return mapping


# Legacy team-code abbreviations (used by some nflverse datasets/seasons for
# relocated franchises) mapped to the current, canonical abbreviation. Apply
# to any column of team abbreviations before treating a team's history as
# continuous across a relocation.
LEGACY_ABBR_MAP = {
    "SD": "LAC",   # San Diego -> LA Chargers (2017)
    "OAK": "LV",   # Oakland -> Las Vegas Raiders (2020)
    "STL": "LA",   # St. Louis -> LA Rams (2016)
    "LAR": "LA",   # alternate current Rams code seen in some datasets
}


def normalize_team_abbr(abbrs):
    """Map legacy relocation-era abbreviations to their current code.
    Accepts a pandas Series or a single string; unmapped values pass through
    unchanged (already-current abbreviations, or unrelated values)."""
    if hasattr(abbrs, "replace") and not isinstance(abbrs, str):
        return abbrs.replace(LEGACY_ABBR_MAP)
    return LEGACY_ABBR_MAP.get(abbrs, abbrs)
