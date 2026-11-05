# Project constraints

- Keep the Home Assistant wrapper small. Reuse the upstream server and API.
- Do not rewrite prediction, training, refresh scheduling, or data-source logic.
- Keep Home Assistant packaging under `home-assistant/`; upstream implementation files stay unchanged. The root README may describe installing the App.
- Add interface checks only where they verify real configuration or lifecycle behavior.
- Never log API keys. Do not add a custom integration until the App is working.
- The App builds a pinned upstream revision; update that pin deliberately when syncing upstream.
- Keep necessary transport compatibility fixes as small documented build-time patches in the App directory; do not rewrite the engine.
- Write App documentation for general users. Keep installation-specific notes out of published docs, distinguish server settings from request parameters, and put build details in developer documentation.
- Do not document numeric transport limits, timeouts or health-check grace periods; those implementation values may change.
- Put App installation instructions in the root README, installed-App usage and configuration in DOCS.md, and manual development setup in DEVELOPMENT.md.
