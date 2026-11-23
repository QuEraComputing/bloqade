"""Tests for word creation helpers."""

from bloqade.gemini.arch.build.word_factory import WordGrid, create_zone_words


class TestWordGrid:
    def _make_grid(self) -> WordGrid:
        """Create a 2x4 entangling grid with zone-local word IDs."""
        return create_zone_words(num_rows=2, num_cols=4, sites_per_word=3)

    def test_word_count(self) -> None:
        grid = self._make_grid()
        assert len(grid.words) == 8
        assert grid.num_rows == 2
        assert grid.num_cols == 4

    def test_word_id_at(self) -> None:
        # Word IDs are zone-local (0..Nw-1) under the shared template.
        grid = self._make_grid()
        assert grid.word_id_at(0, 0) == 0
        assert grid.word_id_at(0, 3) == 3
        assert grid.word_id_at(1, 0) == 4
        assert grid.word_id_at(1, 3) == 7

    def test_word_at_has_correct_sites(self) -> None:
        grid = self._make_grid()
        word = grid.word_at(0, 0)
        assert len(word.site_indices) == 3
        # Even-column word: sites at even x indices (interleaved CZ pairing)
        assert word.site_indices == ((0, 0), (2, 0), (4, 0))

    def test_cz_pairs(self) -> None:
        grid = self._make_grid()
        pairs = list(grid.cz_pairs())
        assert pairs == [(0, 1), (2, 3), (4, 5), (6, 7)]


class TestCreateZoneWords:
    def test_entangling_zone_cz_pairing(self) -> None:
        grid = create_zone_words(num_rows=1, num_cols=2, sites_per_word=3)

        assert len(grid.words) == 2
        # CZ pairing is now at the architecture level via cz_pairs()
        pairs = list(grid.cz_pairs())
        assert pairs == [(0, 1)]

    def test_word_has_correct_site_count(self) -> None:
        """Words have the correct number of sites."""
        grid = create_zone_words(num_rows=1, num_cols=2, sites_per_word=5)

        for word in grid.words:
            assert len(word.site_indices) == 5
