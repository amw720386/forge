# Local Windows helper: print the remote workflow. Does NOT start GPUs.
# Actual spend requires: forge-cloud start --gpu L4 --confirm ...

Write-Host @"
Forge remote CUDA workflow (Windows → Lightning)

1. Push local changes:
     git push

2. Cost preview (no spend):
     go run ./api/cmd/forge-cloud machines
     go run ./api/cmd/forge-cloud start --gpu l4 --duration 20m --dry-run

3. After you explicitly approve a session:
     go run ./api/cmd/forge-cloud start --gpu l4 --duration 20m --confirm

4. On first Studio, clone once (adjust URL):
     go run ./api/cmd/forge-cloud exec --command "git clone <your-forge-repo-url> ~/forge"

5. Build + test on GPU:
     go run ./api/cmd/forge-cloud exec --command "bash ~/forge/scripts/remote_cuda_workflow.sh"

   Or end-to-end (still requires --confirm):
     go run ./api/cmd/forge-cloud benchmark --gpu l4 --confirm

6. Always stop and verify:
     go run ./api/cmd/forge-cloud stop

Local cleanup does not guarantee shutdown if the process dies — auto-sleep is the provider backstop.
See docs/LIGHTNING.md and docs/CUDA_ARCHITECTURE.md
"@
