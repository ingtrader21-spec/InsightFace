# Codestra Recognition Engine API

## Platform
GET /healthz
GET /readyz
GET /metrics

## Runtime
GET /v1/model/status
GET /v1/runtime/capabilities

## Recognition
POST /v1/embeddings
POST /v1/verify
POST /v1/search
POST /v1/batch/search

The engine is intentionally replaceable behind this API. Consumers should not depend directly on a specific third-party model package or internal runtime implementation.
