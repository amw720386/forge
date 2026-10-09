package cloud

import (
	"fmt"
	"strings"

	lit "github.com/lightning-ai/sdk/go"
)

// ResolveMachine maps a user-facing alias to a Lightning SDK Machine constant.
func ResolveMachine(alias string) (lit.Machine, error) {
	switch strings.ToLower(strings.TrimSpace(alias)) {
	case "", "l4":
		return lit.MachineL4, nil
	case "t4":
		return lit.MachineT4, nil
	case "t4-small", "t4small":
		return lit.MachineT4Small, nil
	case "cpu":
		return lit.MachineCPU, nil
	case "l40s":
		return lit.MachineL40S, nil
	case "a10g":
		return lit.MachineA10G, nil
	case "a100":
		return lit.MachineA100, nil
	case "h100":
		return lit.MachineH100, nil
	default:
		return "", fmt.Errorf("unsupported --gpu %q (use l4|t4|cpu|l40s|a10g|a100|h100)", alias)
	}
}

// ListKnownMachines returns aliases shown by `forge-cloud machines`.
func ListKnownMachines() []string {
	return []string{
		"cpu   → lit.MachineCPU",
		"t4    → lit.MachineT4",
		"l4    → lit.MachineL4  (default; preferred for Phase 8A)",
		"l40s  → lit.MachineL40S",
		"a10g  → lit.MachineA10G",
		"a100  → lit.MachineA100",
		"h100  → lit.MachineH100",
	}
}
