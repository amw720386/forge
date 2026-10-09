// Package cloud orchestrates Lightning AI Studios for remote CUDA work.
//
// Verified against the official Go SDK:
//   https://github.com/Lightning-AI/sdk/tree/main/go
//
// Auth: LIGHTNING_USER_ID + LIGHTNING_API_KEY
// Studio: CreateStudio / GetStudio / Start / Stop / Run / SetAutoSleep*
// Machines: lit.MachineL4, MachineT4, …
package cloud
