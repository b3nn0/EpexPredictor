"""UI setup, Supervisor discovery, and editable per-region pricing settings."""

from aiohttp import ClientError
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import SelectSelector, SelectSelectorConfig

from .api import fetch
from .const import DEFAULTS, DOMAIN, PUBLIC_URL, REGIONS, normalize_url, settings


def schema(defaults, include_url=False):
    fields = {vol.Required("url", default=defaults.get("url", "")): str} if include_url else {}
    fields.update({
        vol.Required("region", default=defaults.get("region", "DE")): SelectSelector(SelectSelectorConfig(options=REGIONS)),
        vol.Required("surcharge", default=defaults.get("surcharge", 0.0)): vol.Coerce(float),
        vol.Required("tax_percent", default=defaults.get("tax_percent", 0.0)): vol.All(vol.Coerce(float), vol.Range(min=0, max=100)),
    })
    return vol.Schema(fields)


async def validate(hass, config, exclude=None):
    config = config | {"url": normalize_url(config["url"])}
    for entry in hass.config_entries.async_entries(DOMAIN):
        existing = settings(entry)
        if entry.entry_id != exclude and existing["url"] == config["url"] and existing["region"] == config["region"]:
            raise DuplicateProfile
    await fetch(async_get_clientsession(hass), config)
    return config


class DuplicateProfile(Exception):
    """This server and region already have a configuration entry."""


class EpexPredictorFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    def __init__(self):
        self.url = ""
        self.source = "local"

    async def async_step_user(self, user_input=None):
        if user_input:
            self.source = user_input["source"]
            if self.source == "public":
                self.url = PUBLIC_URL
                return await self.async_step_profile()
            if self.source == "custom":
                return await self.async_step_custom()
            return await self.async_step_local()
        return self.async_show_form(step_id="user", data_schema=vol.Schema({vol.Required("source", default="local"): SelectSelector(SelectSelectorConfig(options=["local", "public", "custom"], translation_key="source"))}))

    async def async_step_local(self, user_input=None):
        urls = []
        if "hassio" in self.hass.config.components:
            from homeassistant.components.hassio.handler import get_supervisor_client
            from aiohasupervisor import SupervisorError

            try:
                messages = await get_supervisor_client(self.hass).discovery.list()
                urls = sorted({normalize_url(m.config["url"]) for m in messages if m.service == DOMAIN})
            except (SupervisorError, KeyError, ValueError):
                pass
        if not urls:
            return self.async_show_form(step_id="user", errors={"base": "no_local_app"}, data_schema=vol.Schema({vol.Required("source", default="custom"): SelectSelector(SelectSelectorConfig(options=["local", "public", "custom"], translation_key="source"))}))
        if user_input or len(urls) == 1:
            self.url = user_input["url"] if user_input else urls[0]
            return await self.async_step_profile()
        return self.async_show_form(step_id="local", data_schema=vol.Schema({vol.Required("url"): vol.In(urls)}))

    async def async_step_custom(self, user_input=None):
        errors = {}
        if user_input:
            try:
                self.url = normalize_url(user_input["url"])
            except ValueError:
                errors["base"] = "invalid_url"
            else:
                return await self.async_step_profile()
        return self.async_show_form(step_id="custom", data_schema=vol.Schema({vol.Required("url"): str}), errors=errors)

    async def async_step_hassio(self, discovery_info):
        self.url = normalize_url(discovery_info.config["url"])
        await self.async_set_unique_id(discovery_info.uuid)
        self._abort_if_unique_id_configured()
        if any(settings(entry)["url"] == self.url for entry in self._async_current_entries()):
            return self.async_abort(reason="already_configured")
        self.context["title_placeholders"] = {"name": discovery_info.name}
        return await self.async_step_profile()

    async def async_step_profile(self, user_input=None):
        errors = {}
        if user_input:
            try:
                config = await validate(self.hass, user_input | {"url": self.url, "source": self.source})
            except DuplicateProfile:
                return self.async_abort(reason="already_configured")
            except (ClientError, TimeoutError, ValueError):
                errors["base"] = "cannot_connect"
            else:
                return self.async_create_entry(title=f"EpexPredictor {config['region']}", data=config)
        return self.async_show_form(step_id="profile", data_schema=schema(DEFAULTS | (user_input or {})), errors=errors)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return EpexPredictorOptions()


class EpexPredictorOptions(config_entries.OptionsFlow):
    async def async_step_init(self, user_input=None):
        errors = {}
        if user_input:
            try:
                config = await validate(self.hass, settings(self.config_entry) | user_input, self.config_entry.entry_id)
            except DuplicateProfile:
                errors["base"] = "already_configured"
            except (ClientError, TimeoutError, ValueError):
                errors["base"] = "cannot_connect"
            else:
                return self.async_create_entry(data=config)
        return self.async_show_form(step_id="init", data_schema=schema(settings(self.config_entry) | (user_input or {}), include_url=True), errors=errors)
