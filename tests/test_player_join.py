"""Regression tests for src.features.player_join's name resolver.

Uses stdlib unittest (no pytest in requirements.txt) and synthetic roster
pools only — no dependency on cached data/ files, so these run in a fresh
clone with no ingest pipeline run first.
"""

import unittest

import pandas as pd

from src.features.player_join import normalize_name, resolve_player_name


def make_pool(rows):
    """rows: list of (player_id, player_name, position, team) tuples."""
    pool = pd.DataFrame(rows, columns=["player_id", "player_name", "position", "team"])
    pool["normalized_name"] = pool["player_name"].map(normalize_name)
    return pool


class TestAbbreviatedInitialMatching(unittest.TestCase):
    """Regression coverage for a real silent-wrong-join bug found on live
    data (2023 week 1, KC@DET): the book's outcome description
    'M. Jones Jr.' fuzzy-matched to 'Cam Jones' (KC LB, difflib ratio 0.875)
    instead of the correct 'Marvin Jones' (DET WR, ratio 0.737) — a short
    WRONG candidate beat the correct long one purely because
    difflib.SequenceMatcher.ratio() is length-sensitive: dropping "Marvin"
    to "M." costs more ratio against a longer correct name than against an
    unrelated short one.

    This is exactly the dangerous failure mode: both outcomes report as
    "resolved" with a status and a ratio, so it's invisible in aggregate
    match-rate stats. Only a specific case-by-case check catches it. These
    tests exist so a future refactor of the matching tiers (e.g. someone
    "simplifying" resolve_player_name back down to a single generic fuzzy
    pass) reintroduces this and breaks a test, not silently ships a bad
    join.
    """

    def test_real_bug_marvin_jones_not_cam_jones(self):
        # Reproduces the exact adversarial pair from the production data:
        # a short wrong-initial candidate closer in length to the abbreviated
        # query than the correct long-initial candidate.
        pool = make_pool([
            ("00-CAM", "Cam Jones", "LB", "KC"),
            ("00-MARVIN", "Marvin Jones", "WR", "DET"),
        ])
        result = resolve_player_name("M. Jones Jr.", pool)
        self.assertEqual(result["player_id"], "00-MARVIN")
        self.assertEqual(result["player_name"], "Marvin Jones")
        self.assertNotEqual(result["player_id"], "00-CAM")

    def test_generic_fuzzy_ratio_would_have_picked_the_wrong_name(self):
        # Documents *why* the bug happened: proves the naive ratio genuinely
        # favors the wrong candidate, so the dedicated tier is load-bearing
        # and not solving a problem that didn't exist.
        import difflib
        wrong_ratio = difflib.SequenceMatcher(None, "m jones", "cam jones").ratio()
        right_ratio = difflib.SequenceMatcher(None, "m jones", "marvin jones").ratio()
        self.assertGreater(wrong_ratio, right_ratio)

    def test_correct_initial_single_candidate_still_matches(self):
        pool = make_pool([("00-CAM", "Cam Jones", "LB", "KC")])
        result = resolve_player_name("C. Jones", pool)
        self.assertEqual(result["player_id"], "00-CAM")
        self.assertEqual(result["status"], "fuzzy")

    def test_wrong_initial_with_no_valid_candidate_is_unresolved_not_guessed(self):
        # The residual gap found while writing this test: even with no
        # exact-last-name candidate sharing the query's initial, the generic
        # fuzzy fallback must stay constrained to that initial rather than
        # falling through to the full (unconstrained) pool.
        pool = make_pool([("00-CAM", "Cam Jones", "LB", "KC")])
        result = resolve_player_name("M. Jones", pool)
        self.assertIsNone(result["player_id"])
        self.assertEqual(result["status"], "unresolved")

    def test_multiple_same_initial_candidates_are_ambiguous_not_guessed(self):
        pool = make_pool([
            ("00-A", "Anthony Smith", "WR", "NYG"),
            ("00-B", "Aaron Smith", "RB", "NYG"),
        ])
        result = resolve_player_name("A. Smith", pool)
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["player_id"])

    def test_position_hint_disambiguates_same_initial_candidates(self):
        pool = make_pool([
            ("00-A", "Anthony Smith", "WR", "NYG"),
            ("00-B", "Aaron Smith", "RB", "NYG"),
        ])
        result = resolve_player_name("A. Smith", pool, position_hint="RB")
        self.assertEqual(result["player_id"], "00-B")
        self.assertEqual(result["status"], "fuzzy")


if __name__ == "__main__":
    unittest.main()
