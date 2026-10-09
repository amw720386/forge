package cloud

import (
	"fmt"
	"strings"
	"time"
)

// EstimatedUSDPerHour are **planning estimates** for Lightning Studio machines.
// Rates change; override with FORGE_CLOUD_RATE_<SLUG>_USD (e.g. FORGE_CLOUD_RATE_L4_USD).
// These are not invoices — always check the Lightning UI / billing before long runs.
var defaultRatesUSD = map[string]float64{
	"cpu":  0.10,
	"t4":   0.45,
	"l4":   0.65, // preferred Phase 8A default
	"l40s": 1.50,
	"a10g": 1.00,
	"a100": 2.50,
	"h100": 4.00,
}

// EstimateSession returns estimated USD for running machine for duration.
func EstimateSession(gpuAlias string, duration time.Duration) (ratePerHour float64, estimated float64, err error) {
	alias := strings.ToLower(strings.TrimSpace(gpuAlias))
	if alias == "" {
		alias = "l4"
	}
	rate, ok := defaultRatesUSD[alias]
	if !ok {
		// try env override only
		rate = envFloat("FORGE_CLOUD_RATE_"+strings.ToUpper(alias)+"_USD", 0)
		if rate <= 0 {
			return 0, 0, fmt.Errorf("unknown GPU alias %q (known: cpu,t4,l4,l40s,a10g,a100,h100)", alias)
		}
	} else if override := envFloat("FORGE_CLOUD_RATE_"+strings.ToUpper(alias)+"_USD", 0); override > 0 {
		rate = override
	}
	hours := duration.Hours()
	if hours <= 0 {
		hours = 1.0 / 60.0 // minimum one minute billing estimate
	}
	return rate, rate * hours, nil
}

// FormatEstimate prints a human-readable cost preview.
func FormatEstimate(gpu string, duration time.Duration, rate, est, limit float64) string {
	return fmt.Sprintf(
		"GPU=%s duration=%s rate≈$%.2f/h estimate≈$%.2f spend_limit=$%.2f",
		gpu, duration, rate, est, limit,
	)
}
