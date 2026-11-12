# Home Assistant packaging

The App directory is a Supervisor build context. It downloads the upstream source
revision pinned in its Dockerfile; it does not copy source from the parent checkout.
Update that pin deliberately when syncing upstream. Upstream dependencies remain
unpinned, so the source pin alone does not make builds fully reproducible.

Keep packaging here and preserve upstream forecasting logic. The pinned source
includes the upstream Energy-Charts rate-limit fix; no build-time source patch
is needed.

For local development, copy the `epexpredictor` directory to
`/addons/epexpredictor` and install from Local apps. Validate configuration,
container build and API startup before updating the published App version.

The App uses Home Assistant's Debian base image with S6 init, a Python virtual
environment and a single foreground launcher. `with-contenv` passes Supervisor
variables to that launcher; the existing Python wrapper handles discovery.

The GitHub App workflow builds natively on amd64 and aarch64. It runs launcher
tests, checks the real API schema, and trains/predicts a small LightGBM model.
An offline upstream stand-in then exercises the real S6 startup, options,
Supervisor discovery, persistent data and graceful stop/restart. It does not
download live forecast data or train the full forecast model. Live validation
must also check forecasts, persistent caches and resource use.

## Custom integration

The integration lives in `custom_components/epexpredictor`, separately from the
App. For a development install, copy that directory into the Home Assistant
configuration directory's `custom_components` folder, then restart HA. The App
must also include discovery support to test its discovered setup flow.

The Marketplace installs the integration separately from the App store. Both packages live
in this repository; installing either one does not install the other.

Its dedicated workflow runs the full config-flow and entity suite on the minimum
supported HA version, plus a runtime smoke check on that version and the current
stable release. The smoke check covers UI setup, forecast state/attributes,
disabled evaluation, Recorder attribute exclusion and unloading, with API requests
mocked. The current release
uses this check until the published HA test fixtures support it. Update the
matrix and extend the full suite to that version when compatible fixtures are
available. Packaging is checked separately by hassfest and the HACS action.
Forecast server and App tests remain independent.
