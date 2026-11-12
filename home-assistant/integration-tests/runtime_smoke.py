"""Check the integration in HA without depending on version-specific test fixtures."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import homeassistant  # Initialize HA's HTTP aliases before importing other clients.
from homeassistant import loader
from homeassistant.bootstrap import async_from_config_dict
from homeassistant.core import HomeAssistant
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.db_schema import StateAttributes, States, StatesMeta
from homeassistant.components.recorder.util import session_scope
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component
from sqlalchemy import select


def recorded_forecasts(hass, entity_id):
    with session_scope(hass=hass, read_only=True) as session:
        return list(session.execute(
            select(States.state, StateAttributes.shared_attrs)
            .join(StatesMeta, States.metadata_id == StatesMeta.metadata_id)
            .join(StateAttributes, States.attributes_id == StateAttributes.attributes_id)
            .where(StatesMeta.entity_id == entity_id)
        ))


async def check(config_dir):
    hass = HomeAssistant(str(config_dir))
    stamp = int(datetime.now(timezone.utc).timestamp()) // 900 * 900
    feed = {"s": [stamp, stamp + 900], "t": [-0.04, 0.12]}
    try:
        loader.async_setup(hass)
        config = {"recorder": {"db_url": "sqlite://", "auto_purge": False}}
        await async_from_config_dict({}, hass)
        assert await async_setup_component(hass, "recorder", config)
        await hass.async_start()
        with patch("custom_components.epexpredictor.config_flow.fetch", new_callable=AsyncMock, return_value=feed), patch("custom_components.epexpredictor.fetch", new_callable=AsyncMock, return_value=feed) as fetch:
            flow = hass.config_entries.flow
            result = await flow.async_init("epexpredictor", context={"source": "user"})
            result = await flow.async_configure(result["flow_id"], {"source": "custom"})
            result = await flow.async_configure(result["flow_id"], {"url": "http://predictor:8000"})
            result = await flow.async_configure(result["flow_id"], {"region": "NL", "surcharge": 2.0, "tax_percent": 21.0})
            assert result["type"] == "create_entry", result
            entry = result["result"]
            await hass.async_block_till_done()
            registry = er.async_get(hass)
            entity_id = registry.async_get_entity_id("sensor", "epexpredictor", f"{entry.entry_id}_prediction")
            assert hass.states.get(entity_id).state == "-0.04"
            assert hass.states.get(entity_id).attributes["t"] == feed["t"]
            evaluation_id = registry.async_get_entity_id("sensor", "epexpredictor", f"{entry.entry_id}_evaluation")
            assert registry.async_get(evaluation_id).disabled_by == er.RegistryEntryDisabler.INTEGRATION
            assert all(not call.kwargs.get("evaluation") for call in fetch.call_args_list)
            await get_instance(hass).async_block_till_done()
            rows = await hass.async_add_executor_job(recorded_forecasts, hass, entity_id)
            assert rows, "Forecast state must still be recorded"
            assert any(state == "-0.04" for state, _ in rows)
            for _, attributes in rows:
                attributes = json.loads(attributes)
                assert "s" not in attributes and "t" not in attributes
            assert await hass.config_entries.async_unload(entry.entry_id)
            print("Runtime smoke passed: UI setup, live forecast, Recorder exclusion, disabled evaluation, unload")
    finally:
        await hass.async_stop(force=True)


if __name__ == "__main__":
    with TemporaryDirectory() as temp:
        config_dir = Path(temp)
        source = Path(__file__).resolve().parents[2] / "custom_components"
        shutil.copytree(source, config_dir / "custom_components")
        sys.path.insert(0, temp)
        asyncio.run(check(config_dir))
