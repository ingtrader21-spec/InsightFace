# Codestra Recognition Engine

This repository is the Codestra-owned service/wrapper boundary around a recognition-engine runtime.

## Planned API

- GET /healthz
- GET /readyz
- GET /metrics
- GET /v1/model/status
- GET /v1/runtime/capabilities
- POST /v1/embeddings
- POST /v1/verify
- POST /v1/search
- POST /v1/batch/search

## Production gates

Model version and checksum authority, explicit model provenance/license, reproducible CPU/GPU runtime profiles, configurable thresholds, calibration fixtures, performance/concurrency tests, SBOM, security scanning, staging benchmarks and rollback/readback.

Codestra branding applies to Codestra-owned wrappers and product surfaces. Upstream licenses, copyright notices, NOTICE files, model-license restrictions, and required third-party attribution must remain intact.
