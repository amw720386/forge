# forge-cloud

The Phase 7 CLI lives in the API module:

```powershell
go run ./api/cmd/forge-cloud status
```

Shared library code: `api/internal/cloud`.

This directory is reserved for a standalone module packaging in Phase 8 if we vendor the Lightning Go SDK separately from the inference server.
