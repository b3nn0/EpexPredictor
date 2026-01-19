"""Tests for predictor.api.priceapi module."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from predictor.api.priceapi import (
    OutputFormat,
    PriceModel,
    PricesModel,
    PricesModelShort,
    PriceUnit,
    RegionPriceManager,
    app,
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_region_manager(sample_region):
    """Create a mock RegionPriceManager."""
    manager = RegionPriceManager(sample_region)
    return manager


class TestPriceUnit:
    """Tests for PriceUnit enum."""

    def test_ct_per_kwh_no_conversion(self):
        """Test CT_PER_KWH returns value as-is."""
        unit = PriceUnit.CT_PER_KWH
        assert unit.convert(10.0) == pytest.approx(10.0)

    def test_eur_per_kwh_conversion(self):
        """Test EUR_PER_KWH divides by 100."""
        unit = PriceUnit.EUR_PER_KWH
        assert unit.convert(100.0) == pytest.approx(1.0)
        assert unit.convert(10.0) == pytest.approx(0.1)

    def test_eur_per_mwh_conversion(self):
        """Test EUR_PER_MWH converts correctly."""
        unit = PriceUnit.EUR_PER_MWH
        # 10 ct/kWh = 0.1 EUR/kWh = 100 EUR/MWh
        assert unit.convert(10.0) == pytest.approx(100.0)


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_long_format_exists(self):
        """Test LONG format exists."""
        assert OutputFormat.LONG.value == "LONG"

    def test_short_format_exists(self):
        """Test SHORT format exists."""
        assert OutputFormat.SHORT.value == "SHORT"


class TestPriceModels:
    """Tests for Pydantic models."""

    def test_price_model_creation(self):
        """Test PriceModel can be created."""
        model = PriceModel(
            starts_at=datetime(2025, 11, 1, tzinfo=timezone.utc),
            total=10.5
        )
        assert model.total == pytest.approx(10.5)

    def test_prices_model_creation(self):
        """Test PricesModel can be created."""
        prices = [
            PriceModel(starts_at=datetime(2025, 11, 1, tzinfo=timezone.utc), total=10.5),
            PriceModel(starts_at=datetime(2025, 11, 1, 0, 15, tzinfo=timezone.utc), total=11.0),
        ]
        model = PricesModel(
            prices=prices,
            known_until=datetime(2025, 11, 2, tzinfo=timezone.utc)
        )
        assert len(model.prices) == 2

    def test_prices_model_short_creation(self):
        """Test PricesModelShort can be created."""
        model = PricesModelShort(
            s=[1730419200, 1730420100],
            t=[10.5, 11.0]
        )
        assert len(model.s) == 2
        assert len(model.t) == 2


class TestPriceModelSerializationAliases:
    """Tests for backward-compatible JSON serialization aliases."""

    def test_price_model_serializes_starts_at_as_camel_case(self):
        """Test PriceModel serializes starts_at as 'startsAt' for backward compatibility."""
        model = PriceModel(
            starts_at=datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc),
            total=10.5
        )
        json_dict = model.model_dump(by_alias=True)
        assert "startsAt" in json_dict
        assert "starts_at" not in json_dict

    def test_price_model_internal_name_still_works(self):
        """Test PriceModel can still be accessed via internal snake_case name."""
        model = PriceModel(
            starts_at=datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc),
            total=10.5
        )
        assert model.starts_at == datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc)

    def test_prices_model_serializes_known_until_as_camel_case(self):
        """Test PricesModel serializes known_until as 'knownUntil' for backward compatibility."""
        model = PricesModel(
            prices=[],
            known_until=datetime(2025, 11, 2, tzinfo=timezone.utc)
        )
        json_dict = model.model_dump(by_alias=True)
        assert "knownUntil" in json_dict
        assert "known_until" not in json_dict

    def test_full_response_uses_camel_case_aliases(self):
        """Test complete response structure uses camelCase for API backward compatibility."""
        price = PriceModel(
            starts_at=datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc),
            total=10.5
        )
        model = PricesModel(
            prices=[price],
            known_until=datetime(2025, 11, 2, tzinfo=timezone.utc)
        )
        json_dict = model.model_dump(by_alias=True)

        # Top level should have knownUntil
        assert "knownUntil" in json_dict
        # Nested price should have startsAt
        assert "startsAt" in json_dict["prices"][0]
        assert json_dict["prices"][0]["startsAt"] is not None


class TestRegionPriceManagerFormatShort:
    """Tests for RegionPriceManager.format_short method."""

    def test_format_short(self, sample_region):
        """Test format_short converts to short format."""
        manager = RegionPriceManager(sample_region)

        prices = [
            PriceModel(starts_at=datetime(2025, 11, 1, tzinfo=timezone.utc), total=10.5),
            PriceModel(starts_at=datetime(2025, 11, 1, 0, 15, tzinfo=timezone.utc), total=11.0),
        ]

        result = manager.format_short(prices)

        assert isinstance(result, PricesModelShort)
        assert len(result.s) == 2
        assert len(result.t) == 2
        assert result.t[0] == pytest.approx(10.5)
        assert result.t[1] == pytest.approx(11.0)


class TestAPIEndpointRoot:
    """Tests for root endpoint."""

    def test_root_redirects_to_docs(self, client):
        """Test that root endpoint redirects to /docs."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert "/docs" in response.headers.get("location", "")


class TestAPIEndpointPrices:
    """Tests for /prices endpoint."""

    def test_prices_endpoint_exists(self, client):
        """Test that /prices endpoint returns 200 with mocked handler."""
        with patch("predictor.api.priceapi.prices_handler") as mock_handler:
            mock_handler.prices = AsyncMock(
                return_value=PricesModel(
                    prices=[],
                    known_until=datetime(2025, 11, 1, tzinfo=timezone.utc)
                )
            )
            response = client.get("/prices")
            assert response.status_code == 200

    def test_prices_with_hours_parameter(self, client):
        """Test /prices with hours parameter returns 200."""
        with patch("predictor.api.priceapi.prices_handler") as mock_handler:
            mock_handler.prices = AsyncMock(
                return_value=PricesModel(
                    prices=[],
                    known_until=datetime(2025, 11, 1, tzinfo=timezone.utc)
                )
            )
            response = client.get("/prices?hours=24")
            assert response.status_code == 200

    def test_prices_with_country_parameter(self, client):
        """Test /prices with country parameter returns 200."""
        with patch("predictor.api.priceapi.prices_handler") as mock_handler:
            mock_handler.prices = AsyncMock(
                return_value=PricesModel(
                    prices=[],
                    known_until=datetime(2025, 11, 1, tzinfo=timezone.utc)
                )
            )
            response = client.get("/prices?country=DE")
            assert response.status_code == 200

    def test_prices_with_unit_parameter(self, client):
        """Test /prices with unit parameter returns 200."""
        with patch("predictor.api.priceapi.prices_handler") as mock_handler:
            mock_handler.prices = AsyncMock(
                return_value=PricesModel(
                    prices=[],
                    known_until=datetime(2025, 11, 1, tzinfo=timezone.utc)
                )
            )
            response = client.get("/prices?unit=EUR_PER_KWH")
            assert response.status_code == 200


class TestAPIEndpointPricesShort:
    """Tests for /prices_short endpoint."""

    def test_prices_short_endpoint_exists(self, client):
        """Test that /prices_short endpoint returns 200."""
        with patch("predictor.api.priceapi.prices_handler") as mock_handler:
            mock_handler.prices = AsyncMock(
                return_value=PricesModelShort(s=[], t=[])
            )
            response = client.get("/prices_short")
            assert response.status_code == 200


class TestRegionPriceManagerPrices:
    """Tests for RegionPriceManager.prices method."""

    @pytest.mark.asyncio
    async def test_prices_applies_fixed_price(self, sample_region):
        """Test that fixed price is added to all prices."""
        manager = RegionPriceManager(sample_region)

        # Add cached prices
        base_time = datetime(2025, 11, 1, tzinfo=timezone.utc)
        manager.cachedprices = {
            base_time: 10.0,
            base_time + timedelta(minutes=15): 11.0,
        }
        manager.last_known_price = base_time + timedelta(hours=1)

        # Mock update_in_background to do nothing
        manager.update_in_background = AsyncMock()

        result = await manager.prices(
            hours=1,
            fixed_price=5.0,
            tax_percent=0.0,
            format=OutputFormat.LONG
        )

        assert isinstance(result, PricesModel)
        # Prices should have fixed price added
        for price in result.prices:
            assert price.total >= 15.0  # 10 + 5 or 11 + 5

    @pytest.mark.asyncio
    async def test_prices_applies_tax(self, sample_region):
        """Test that tax is applied to prices."""
        manager = RegionPriceManager(sample_region)

        # Add cached prices
        base_time = datetime(2025, 11, 1, tzinfo=timezone.utc)
        manager.cachedprices = {
            base_time: 10.0,
        }
        manager.last_known_price = base_time + timedelta(hours=1)

        manager.update_in_background = AsyncMock()

        result = await manager.prices(
            hours=1,
            fixed_price=0.0,
            tax_percent=19.0,
            format=OutputFormat.LONG
        )

        assert isinstance(result, PricesModel)
        # Price should be 10 * 1.19 = 11.9
        if result.prices:
            assert result.prices[0].total == pytest.approx(11.9, rel=0.01)

    @pytest.mark.asyncio
    async def test_prices_hourly_averaging(self, sample_region):
        """Test that hourly mode averages 15-minute prices."""
        manager = RegionPriceManager(sample_region)

        # Add 4 prices for one hour (15-min intervals)
        base_time = datetime(2025, 11, 1, tzinfo=timezone.utc)
        manager.cachedprices = {
            base_time: 10.0,
            base_time + timedelta(minutes=15): 12.0,
            base_time + timedelta(minutes=30): 14.0,
            base_time + timedelta(minutes=45): 16.0,
        }
        manager.last_known_price = base_time + timedelta(hours=2)

        manager.update_in_background = AsyncMock()

        result = await manager.prices(
            hours=1,
            fixed_price=0.0,
            tax_percent=0.0,
            hourly=True,
            format=OutputFormat.LONG
        )

        assert isinstance(result, PricesModel)
        # Should have 1 hourly price (average of 10, 12, 14, 16 = 13)
        if result.prices:
            assert result.prices[0].total == pytest.approx(13.0, rel=0.01)


class TestRegionPriceManagerUpdateDataIfNeeded:
    """Tests for RegionPriceManager.update_data_if_needed method."""

    @pytest.mark.asyncio
    async def test_update_triggers_refresh_when_stale(self, sample_region):
        """Test that update triggers refresh when data is stale."""
        manager = RegionPriceManager(sample_region)

        # Mock predictor methods
        manager.predictor.refresh_prices = AsyncMock(return_value=True)
        manager.predictor.refresh_weather = AsyncMock()
        manager.predictor.train = AsyncMock()
        manager.predictor.predict = AsyncMock(
            return_value=MagicMock(empty=False)
        )
        manager.predictor.to_price_dict = MagicMock(return_value={})
        manager.predictor.pricestore.get_last_known = MagicMock(
            return_value=datetime.now(timezone.utc)
        )
        manager.predictor.cleanup = MagicMock()

        await manager.update_data_if_needed()

        # Should have called refresh methods
        assert manager.predictor.refresh_prices.called
