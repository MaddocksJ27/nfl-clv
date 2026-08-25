"""Map odds-book full team names (e.g. "Carolina Panthers") to nflverse
team abbreviations (e.g. "CAR"). Used to join odds-API events against
nflverse schedules/rosters.
"""

import nfl_data_py as nfl

# nflverse's team_desc includes retired/legacy abbreviations for relocated
# franchises (LAR, OAK, SD, STL) alongside the current ones (LA, LV, LAC) —
# drop the legacy rows so each full team name maps to exactly one abbreviation.
_LEGACY_ABBRS = {"LAR", "OAK", "SD", "STL"}


def team_name_map() -> dict:
    """Return {full_team_name: nflverse_abbr} for all 32 current teams."""
    td = nfl.import_team_desc()
    td = td[~td.team_abbr.isin(_LEGACY_ABBRS)]
    return dict(zip(td.team_name, td.team_abbr))
