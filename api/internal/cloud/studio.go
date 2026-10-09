package cloud

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	lit "github.com/lightning-ai/sdk/go"
)

// Client wraps Lightning Studio operations used by forge-cloud.
type Client struct {
	Cfg Config
	// active is the Studio handle returned from Start(), with in-memory
	// status "running". Prefer this for Run/Upload — GetStudio() status is
	// cloudspace-level (often CLOUD_SPACE_STATE_READY) and blocks Run.
	active *lit.Studio
}

func (c *Client) teamspace() (*lit.Teamspace, error) {
	ts, _, err := c.ResolveTeamspace()
	return ts, err
}

func (c *Client) getOrCreateStudio(ts *lit.Teamspace, machine lit.Machine) (*lit.Studio, error) {
	name := c.Cfg.StudioName
	st, err := lit.GetStudio(name, lit.StudioOptions{Teamspace: ts})
	if err == nil && st != nil {
		return st, nil
	}
	autoSleep := c.Cfg.AutoSleepMinutes > 0
	st, err = lit.CreateStudio(name, lit.StudioOptions{
		Teamspace:     ts,
		Machine:       machine,
		AutoSleep:     autoSleep,
		AutoSleepTime: c.Cfg.AutoSleepMinutes,
	})
	if err != nil {
		return nil, fmt.Errorf("CreateStudio: %w", err)
	}
	return st, nil
}

// StartOptions controls billable Studio start.
type StartOptions struct {
	GPUAlias string
	Confirm  bool
	Duration time.Duration
	DryRun   bool
}

// StartResult is returned after a successful (or dry-run) start.
type StartResult struct {
	StudioName   string
	Status       string
	Machine      string
	EstimateUSD  float64
	RatePerHour  float64
	Duration     time.Duration
	DryRun       bool
	AutoSleepMin int
	Note         string
}

// Start provisions/starts a Studio. Refuses without Confirm unless DryRun.
func (c *Client) Start(opt StartOptions) (*StartResult, error) {
	if !opt.DryRun && c.Cfg.UserID == "" {
		if _, err := LoadConfig(); err != nil {
			return nil, err
		}
	}
	gpu := opt.GPUAlias
	if gpu == "" {
		gpu = c.Cfg.DefaultGPU
	}
	dur := opt.Duration
	if dur <= 0 {
		dur = c.Cfg.MaxSession
	}
	if dur > c.Cfg.MaxSession {
		return nil, fmt.Errorf("requested duration %s exceeds FORGE_CLOUD_MAX_SESSION %s", dur, c.Cfg.MaxSession)
	}
	rate, est, err := EstimateSession(gpu, dur)
	if err != nil {
		return nil, err
	}
	if est > c.Cfg.SpendLimitUSD {
		return nil, fmt.Errorf("estimated $%.2f exceeds spend limit $%.2f", est, c.Cfg.SpendLimitUSD)
	}

	res := &StartResult{
		StudioName:   c.Cfg.StudioName,
		Machine:      gpu,
		EstimateUSD:  est,
		RatePerHour:  rate,
		Duration:     dur,
		DryRun:       opt.DryRun,
		AutoSleepMin: c.Cfg.AutoSleepMinutes,
		Note: FormatEstimate(gpu, dur, rate, est, c.Cfg.SpendLimitUSD) +
			"; provider auto-sleep is the idle safety net if the local process dies",
	}

	if opt.DryRun {
		res.Status = "dry-run"
		res.Note += " (no API calls)"
		return res, nil
	}

	if !opt.Confirm {
		return nil, fmt.Errorf(
			"refusing to provision: pass --confirm after reviewing estimate (%s)",
			FormatEstimate(gpu, dur, rate, est, c.Cfg.SpendLimitUSD),
		)
	}

	machine, err := ResolveMachine(gpu)
	if err != nil {
		return nil, err
	}
	ts, err := c.teamspace()
	if err != nil {
		return nil, err
	}
	st, err := c.getOrCreateStudio(ts, machine)
	if err != nil {
		return nil, err
	}

	if c.Cfg.AutoSleepMinutes > 0 {
		_ = st.SetAutoSleep(true)
		_ = st.SetAutoSleepTime(c.Cfg.AutoSleepMinutes * 60)
	}

	maxRuntimeSec := int(dur.Seconds())
	if err := st.Start(lit.StartStudioOptions{
		Machine:    machine,
		MaxRuntime: maxRuntimeSec,
	}); err != nil {
		return nil, fmt.Errorf("studio.Start: %w", err)
	}
	c.active = st

	_ = SaveState(SessionState{
		StudioName:   st.Name(),
		StudioID:     st.ID(),
		Org:          c.Cfg.OrgName,
		Teamspace:    c.Cfg.TeamspaceName,
		Machine:      gpu,
		StartedAt:    time.Now().UTC(),
		MaxSession:   dur.String(),
		SpendLimit:   c.Cfg.SpendLimitUSD,
		EstimateUSD:  est,
		AutoSleepMin: c.Cfg.AutoSleepMinutes,
	})

	if err := c.waitInstanceRunnable(st, 8*time.Minute); err != nil {
		res.Status = "start-issued"
		res.Note += "; WARNING: start issued but instance not runnable — call forge-cloud stop"
		return res, fmt.Errorf("wait instance: %w", err)
	}
	res.Status = "running"
	return res, nil
}

func (c *Client) waitInstanceRunnable(st *lit.Studio, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		out, err := st.Run("echo forge-ready")
		if err == nil && strings.Contains(out, "forge-ready") {
			return nil
		}
		lastErr = err
		if ip, ipErr := st.PublicIP(); ipErr == nil && strings.TrimSpace(ip) != "" {
			lastErr = fmt.Errorf("public IP present but Run not ready: %v", err)
		}
		time.Sleep(8 * time.Second)
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("timeout")
	}
	return lastErr
}

func (c *Client) runnableStudio() (*lit.Studio, error) {
	if c.active != nil {
		return c.active, nil
	}
	ts, err := c.teamspace()
	if err != nil {
		return nil, err
	}
	name := c.Cfg.StudioName
	studioID := ""
	if st, err := LoadState(); err == nil && st.StudioName != "" {
		name = st.StudioName
		studioID = st.StudioID
	}
	studio, err := lit.GetStudio(name, lit.StudioOptions{Teamspace: ts})
	if err != nil {
		return nil, err
	}
	if studioID == "" {
		studioID = studio.ID()
	}
	// GetStudio by name leaves cloudspace status (CLOUD_SPACE_STATE_READY), which
	// blocks Run. ID-based handle seeded with Status "running" matches Start().
	if studioID != "" {
		running, err := lit.GetStudio(name, lit.StudioOptions{
			ID:       studioID,
			Teamspace: ts,
			Status:   "running",
		})
		if err == nil {
			studio = running
		}
	}
	if out, err := studio.Run("echo forge-reattach"); err == nil && strings.Contains(out, "forge-reattach") {
		c.active = studio
		return studio, nil
	} else if err != nil {
		return nil, fmt.Errorf("studio run after reattach: %w", err)
	}
	c.active = studio
	return studio, nil
}

// UploadPath uploads a local directory/file into the Studio. Never use for .env.
func (c *Client) UploadPath(localPath, remotePath string) error {
	base := strings.ToLower(filepath.Base(localPath))
	if base == ".env" || strings.Contains(base, ".env") {
		return fmt.Errorf("refusing to upload secret file %q", base)
	}
	studio, err := c.runnableStudio()
	if err != nil {
		return err
	}
	fi, err := os.Stat(localPath)
	if err != nil {
		return err
	}
	if fi.IsDir() {
		return studio.UploadFolder(localPath, remotePath)
	}
	return studio.UploadFile(localPath, remotePath)
}

// Stop tears down the Studio compute instance.
func (c *Client) Stop() (statusBefore, statusAfter string, err error) {
	studio, err := c.runnableStudio()
	if err != nil {
		return "", "", err
	}
	statusBefore = studio.Status()
	if err := studio.Stop(); err != nil {
		return statusBefore, "", fmt.Errorf("studio.Stop: %w", err)
	}
	c.active = nil

	// Instance stop succeeded at the API. Cloudspace GetStudio status often
	// remains CLOUD_SPACE_STATE_READY (definition exists) — that is OK.
	// Verify compute is down by ensuring Run fails / PublicIP clears.
	deadline := time.Now().Add(2 * time.Minute)
	statusAfter = "stopped"
	for time.Now().Before(deadline) {
		ip, _ := studio.PublicIP()
		_, runErr := studio.Run("true")
		if strings.TrimSpace(ip) == "" && runErr != nil {
			statusAfter = "stopped (instance unreachable)"
			_ = ClearState()
			return statusBefore, statusAfter, nil
		}
		time.Sleep(5 * time.Second)
	}
	// Soft success: Stop API returned nil; warn if probes still look live.
	ip, _ := studio.PublicIP()
	if strings.TrimSpace(ip) != "" {
		_ = ClearState()
		return statusBefore, "stop-api-ok-but-ip-still-present", fmt.Errorf(
			"Stop API succeeded but PublicIP still set (%s) — verify in Lightning UI", ip)
	}
	_ = ClearState()
	return statusBefore, statusAfter, nil
}

// Exec runs a command on the Studio via SDK Run.
func (c *Client) Exec(command string) (string, error) {
	command = strings.TrimSpace(command)
	if command == "" {
		return "", fmt.Errorf("empty --command")
	}
	studio, err := c.runnableStudio()
	if err != nil {
		return "", err
	}
	return studio.Run(command)
}

// StudioStatus fetches cloudspace status (not the same as instance running).
func (c *Client) StudioStatus() (name, status, machine string, err error) {
	ts, err := c.teamspace()
	if err != nil {
		return "", "", "", err
	}
	name = c.Cfg.StudioName
	if st, err := LoadState(); err == nil && st.StudioName != "" {
		name = st.StudioName
	}
	studio, err := lit.GetStudio(name, lit.StudioOptions{Teamspace: ts})
	if err != nil {
		return name, "", "", err
	}
	return studio.Name(), studio.Status(), studio.Machine(), nil
}
