"""Exercise config flows and entities against a real HA test instance."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from aiohttp import ClientConnectionError
import pytest

pytest.importorskip("homeassistant")
from pytest_homeassistant_custom_component.common import MockConfigEntry
from homeassistant import config_entries
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.service_info.hassio import HassioServiceInfo
from custom_components.epexpredictor.api import validate_forecast

CONFIG = {"url": "http://predictor:8000", "region": "NL", "surcharge": 2.0, "tax_percent": 21.0, "source": "custom"}


@pytest.fixture(autouse=True)
def custom_integration(enable_custom_integrations):
    yield


@pytest.fixture
def feed():
    stamp = int(datetime.now(timezone.utc).timestamp()) // 900 * 900
    data = {"s": [stamp, stamp + 900], "t": [-0.04, 0.12]}
    with patch("custom_components.epexpredictor.config_flow.fetch", new_callable=AsyncMock, return_value=data) as flow_fetch, patch("custom_components.epexpredictor.fetch", new_callable=AsyncMock, return_value=data) as coordinator_fetch:
        yield flow_fetch, coordinator_fetch


@pytest.mark.asyncio
async def test_custom_setup_and_duplicate(hass, feed):
    result = await hass.config_entries.flow.async_init("epexpredictor", context={"source": "user"})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"source": "custom"})
    assert result["step_id"] == "custom"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"url": CONFIG["url"] + "/"})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {k: CONFIG[k] for k in ("region", "surcharge", "tax_percent")})
    assert result["type"] == "create_entry"
    await hass.async_block_till_done()
    duplicate = await hass.config_entries.flow.async_init("epexpredictor", context={"source": "user"})
    duplicate = await hass.config_entries.flow.async_configure(duplicate["flow_id"], {"source": "custom"})
    duplicate = await hass.config_entries.flow.async_configure(duplicate["flow_id"], {"url": CONFIG["url"]})
    duplicate = await hass.config_entries.flow.async_configure(duplicate["flow_id"], {k: CONFIG[k] for k in ("region", "surcharge", "tax_percent")})
    assert duplicate["reason"] == "already_configured"


@pytest.mark.asyncio
async def test_evaluation_is_not_requested_until_enabled(hass, feed):
    entry = MockConfigEntry(domain="epexpredictor", data=CONFIG)
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    registry = er.async_get(hass)
    evaluation_id = registry.async_get_entity_id("sensor", "epexpredictor", f"{entry.entry_id}_evaluation")
    prediction_id = registry.async_get_entity_id("sensor", "epexpredictor", f"{entry.entry_id}_prediction")
    assert registry.async_get(evaluation_id).disabled_by == er.RegistryEntryDisabler.INTEGRATION
    assert hass.states.get(prediction_id).state == "-0.04"
    assert all(not call.kwargs.get("evaluation") for call in feed[1].call_args_list)
    assert await hass.config_entries.async_unload(entry.entry_id)
    registry.async_update_entity(evaluation_id, disabled_by=None)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert any(call.kwargs.get("evaluation") is True for call in feed[1].call_args_list)
    assert hass.states.get(evaluation_id).state == "-0.04"


@pytest.mark.asyncio
async def test_public_server_and_missing_local_app(hass, feed):
    result = await hass.config_entries.flow.async_init("epexpredictor", context={"source": "user"})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"source": "local"})
    assert result["errors"]["base"] == "no_local_app"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"source": "public"})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {k: CONFIG[k] for k in ("region", "surcharge", "tax_percent")})
    assert result["data"]["url"] == "https://epexpredictor.batzill.com"
    await hass.async_block_till_done()


@pytest.mark.asyncio
async def test_evaluation_failure_keeps_regular_forecast(hass, feed):
    entry = MockConfigEntry(domain="epexpredictor", data=CONFIG)
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    coordinator = entry.runtime_data
    prediction = coordinator.data["prediction"]

    async def fetch(session, config, evaluation=False):
        if evaluation:
            raise ClientConnectionError("evaluation unavailable")
        return prediction

    with patch.object(coordinator, "async_contexts", return_value=iter(["evaluation"])), patch("custom_components.epexpredictor.fetch", side_effect=fetch):
        await coordinator.async_refresh()
    assert coordinator.last_update_success
    assert coordinator.data["prediction"] == prediction
    assert coordinator.data["evaluation"] is None


@pytest.mark.asyncio
async def test_discovery_and_options(hass, feed):
    info = HassioServiceInfo(config={"url": CONFIG["url"]}, name="EpexPredictor", slug="example_epexpredictor", uuid="example")
    result = await hass.config_entries.flow.async_init("epexpredictor", context={"source": "hassio"}, data=info)
    assert result["step_id"] == "profile"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {k: CONFIG[k] for k in ("region", "surcharge", "tax_percent")})
    entry = result["result"]
    await hass.async_block_till_done()
    options = await hass.config_entries.options.async_init(entry.entry_id)
    options = await hass.config_entries.options.async_configure(options["flow_id"], {"url": CONFIG["url"], "region": "DE", "surcharge": 3.0, "tax_percent": 19.0})
    assert options["type"] == "create_entry"
    await hass.async_block_till_done()
    assert entry.runtime_data.config["region"] == "DE"


@pytest.mark.parametrize("data", [{}, {"s": [1, 2], "t": [1]}, {"s": [2, 1], "t": [1, 2]}, {"s": [1], "t": [float("nan")]}, {"s": [1], "t": [True]}])
def test_invalid_forecasts(data):
    with pytest.raises(ValueError):
        validate_forecast(data)
