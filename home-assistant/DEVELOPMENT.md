# Home Assistant packaging

The App directory is a Supervisor build context. It downloads the upstream source
revision pinned in its Dockerfile; it does not copy source from the parent checkout.
Update that pin deliberately when syncing upstream. Upstream dependencies remain
unpinned, so the source pin alone does not make builds fully reproducible.

Keep packaging and necessary compatibility patches here. Preserve upstream
forecasting logic and prefer upstream fixes over maintaining parallel behavior.
The Energy-Charts transport patch is applied during the container build; patch
application must fail if a future source revision is incompatible.

For local development, copy the `epexpredictor` directory to
`/addons/epexpredictor` and install from Local apps. Validate configuration,
container build and API startup before updating the published App version.

The GitHub App workflow runs launcher tests, parses the App manifest, builds the
container and checks the API schema without starting network downloads/training.
Live validation must also check forecasts, persistent caches and resource use.

## Custom integration

The integration lives in `custom_components/epexpredictor`, separately from the
App. For a development install, copy that directory into the Home Assistant
configuration directory's `custom_components` folder, then restart HA. The App
must also include discovery support to test its discovered setup flow.

The integration branch is not the repository's published default branch. HACS
reads the selected repository version; do not assume adding the current default
App repository also installs the development integration. Release packaging and
the HACS installation path must be verified before publishing the integration.

Its dedicated workflow installs the supported HA version and runs mocked
config-flow and entity tests in Linux. Forecast server and App tests remain
independent. Keep network calls mocked in integration unit tests.
