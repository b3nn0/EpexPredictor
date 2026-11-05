# Project constraints

- Keep the Home Assistant wrapper small. Reuse the upstream server and API.
- Do not rewrite prediction, training, refresh scheduling, or data-source logic.
- Keep Home Assistant packaging under `home-assistant/`; upstream files stay unchanged.
- Add interface checks only where they verify real configuration or lifecycle behavior.
- Never log API keys. Do not add a custom integration until the App is working.
- The App builds a pinned upstream revision; update that pin deliberately when syncing upstream.
