# Codestra Recognition Engine
CPU-first local face-embedding service used by Codestra FACE-ID.

Default implementation uses OpenCV YuNet for detection and OpenCV SFace for recognition. Model files are downloaded during the Docker image build and are not committed.

API: /healthz, /readyz, /metrics, /v1/model/status, /v1/runtime/capabilities, /v1/embeddings, /v1/verify, /v1/search, /v1/batch/search.
