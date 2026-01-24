"""Tests for predictor.model.priceregion module."""


from predictor.model.priceregion import PriceRegion, PriceRegionName


class TestPriceRegionName:
    """Tests for PriceRegionName enum."""

    def test_all_region_names_defined(self):
        """Test that all expected region names are defined."""
        expected = ["DE", "AT", "BE", "NL"]
        actual = [r.value for r in PriceRegionName]
        assert actual == expected

    def test_to_region_de(self):
        """Test conversion from PriceRegionName to PriceRegion for DE."""
        region = PriceRegionName.DE.to_region()
        assert region == PriceRegion.DE
        assert region.country_code == "DE"

    def test_to_region_at(self):
        """Test conversion from PriceRegionName to PriceRegion for AT."""
        region = PriceRegionName.AT.to_region()
        assert region == PriceRegion.AT
        assert region.country_code == "AT"

    def test_to_region_be(self):
        """Test conversion from PriceRegionName to PriceRegion for BE."""
        region = PriceRegionName.BE.to_region()
        assert region == PriceRegion.BE
        assert region.country_code == "BE"

    def test_to_region_nl(self):
        """Test conversion from PriceRegionName to PriceRegion for NL."""
        region = PriceRegionName.NL.to_region()
        assert region == PriceRegion.NL
        assert region.country_code == "NL"


class TestPriceRegion:
    """Tests for PriceRegion enum."""

    def test_holidays_initialized(self):
        """Test that holidays are properly initialized for all regions."""
        for region in PriceRegion:
            assert region.holidays is not None
            assert len(region.holidays) > 0

    def test_de_has_subdivisions(self):
        """Test that DE has multiple holiday subdivisions (Bundesländer)."""
        region = PriceRegion.DE
        # Germany has 16 states, so should have multiple holiday sets
        assert len(region.holidays) > 1

    def test_coordinates_valid_ranges(self):
        """Test that all coordinates are within valid geographic ranges."""
        for region in PriceRegion:
            for lat in region.latitudes:
                assert -90 <= lat <= 90, f"Invalid latitude {lat} for {region}"
            for lon in region.longitudes:
                assert -180 <= lon <= 180, f"Invalid longitude {lon} for {region}"

    def test_coordinates_in_europe(self):
        """Test that all coordinates are roughly within European bounds."""
        for region in PriceRegion:
            for lat in region.latitudes:
                assert 35 <= lat <= 72, f"Latitude {lat} not in Europe for {region}"
            for lon in region.longitudes:
                assert -25 <= lon <= 45, f"Longitude {lon} not in Europe for {region}"
