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

    def test_de_properties(self):
        """Test DE region properties."""
        region = PriceRegion.DE
        assert region.country_code == "DE"
        assert region.timezone == "Europe/Berlin"
        assert region.bidding_zone == "DE-LU"
        assert len(region.latitudes) == 6
        assert len(region.longitudes) == 6
        assert len(region.latitudes) == len(region.longitudes)

    def test_at_properties(self):
        """Test AT region properties."""
        region = PriceRegion.AT
        assert region.country_code == "AT"
        assert region.timezone == "Europe/Berlin"
        assert region.bidding_zone == "AT"
        assert len(region.latitudes) == 5
        assert len(region.longitudes) == 5

    def test_be_properties(self):
        """Test BE region properties."""
        region = PriceRegion.BE
        assert region.country_code == "BE"
        assert region.timezone == "Europe/Berlin"
        assert region.bidding_zone == "BE"
        assert len(region.latitudes) == 3
        assert len(region.longitudes) == 3

    def test_nl_properties(self):
        """Test NL region properties."""
        region = PriceRegion.NL
        assert region.country_code == "NL"
        assert region.timezone == "Europe/Amsterdam"
        assert region.bidding_zone == "NL"
        assert len(region.latitudes) == 3
        assert len(region.longitudes) == 3

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
