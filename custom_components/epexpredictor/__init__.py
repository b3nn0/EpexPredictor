"""Connect Home Assistant to a local or remote EpexPredictor server."""

from datetime import timedelta
import logging

from aiohttp import ClientError
from homeassistant.const import Platform
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .api import fetch
from .const import DOMAIN, settings

LOGGER = logging.getLogger(__name__)


class ForecastCoordinator(DataUpdateCoordinator):
    def __init__(self, hass, entry):
        super().__init__(hass, LOGGER, name=DOMAIN, config_entry=entry, update_interval=timedelta(hours=1))
        self.config = settings(entry)
        self.session = async_get_clientsession(hass)

    async def _async_update_data(self):
        try:
            prediction = await fetch(self.session, self.config)
        except (ClientError, TimeoutError, ValueError) as error:
            raise UpdateFailed(f"Unable to fetch forecast: {error}") from error
        evaluation = None
        if "evaluation" in self.async_contexts():
            try:
                evaluation = await fetch(self.session, self.config, evaluation=True)
            except (ClientError, TimeoutError, ValueError) as error:
                LOGGER.warning("Unable to fetch evaluation forecast: %s", error)
        return {"prediction": prediction, "evaluation": evaluation, "fetched": dt_util.utcnow()}


async def async_setup_entry(hass, entry):
    coordinator = ForecastCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator
    entry.async_on_unload(entry.add_update_listener(_reload))
    await hass.config_entries.async_forward_entry_setups(entry, [Platform.SENSOR])
    return True


async def _reload(hass, entry):
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass, entry):
    return await hass.config_entries.async_unload_platforms(entry, [Platform.SENSOR])
