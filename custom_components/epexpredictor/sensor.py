"""Forecast entities with compact timestamp/price attributes."""

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.const import EntityCategory
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import DOMAIN


async def async_setup_entry(hass, entry, async_add_entities):
    async_add_entities([ForecastSensor(entry, key) for key in ("prediction", "evaluation", "fetched")])


class ForecastSensor(CoordinatorEntity, SensorEntity):
    _attr_has_entity_name = True

    def __init__(self, entry, key):
        super().__init__(entry.runtime_data, context=key)
        self.key = key
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_translation_key = key
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)}, "name": f"EpexPredictor {self.coordinator.config['region']}",
            "manufacturer": "EpexPredictor", "configuration_url": self.coordinator.config["url"] + "/docs",
        }
        if key == "fetched":
            self._attr_device_class = SensorDeviceClass.TIMESTAMP
            self._attr_entity_category = EntityCategory.DIAGNOSTIC
        else:
            self._attr_native_unit_of_measurement = "EUR/kWh"
            self._attr_icon = "mdi:chart-line"
        self._attr_entity_registry_enabled_default = key != "evaluation"

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        if self.key != "fetched":
            self.async_on_remove(async_track_time_change(self.hass, lambda now: self.async_write_ha_state(), minute=[0, 15, 30, 45], second=0))
        if self.key == "evaluation":
            await self.coordinator.async_request_refresh()

    @property
    def available(self):
        return super().available and self.coordinator.data.get(self.key) is not None

    @property
    def native_value(self):
        value = self.coordinator.data.get(self.key)
        if self.key == "fetched" or value is None:
            return value
        now = dt_util.utcnow().timestamp()
        index = max((i for i, stamp in enumerate(value["s"]) if stamp <= now), default=0)
        return value["t"][index]

    @property
    def extra_state_attributes(self):
        return self.coordinator.data.get(self.key) if self.key != "fetched" else None
