"""Web-agnostic application core for the gender-aware subtitle server.

Holds orchestration, runtime bootstrap, concurrency, translation-backend selection,
audio I/O, side-file output, and lifecycle warm-up — everything that is NOT the HTTP
layer. Importable without FastAPI, which is the seam a future standalone CLI uses:
``core.cuda.bootstrap()`` then ``core.orchestrator.run_pipeline_async(...)``.
"""
