package main

import (
	"encoding/json"
	"fmt"
	"github.com/forge-ml/forge/api/internal/cloud"
	lit "github.com/lightning-ai/sdk/go"
	"net/http"
	"strings"
	"time"
)

// Read-only provider inspection; credentials never leave request headers.
func cmdInspectPhase13(opt cloud.LoadOptions) error {
	cfg, err := mustConfig(opt)
	if err != nil {
		return err
	}
	c := &cloud.Client{Cfg: cfg}
	ts, _, err := c.ResolveTeamspace()
	if err != nil {
		return err
	}
	st, err := lit.GetStudio("forge-gpu", lit.StudioOptions{Teamspace: ts})
	if err != nil {
		return err
	}
	fmt.Printf("studio=%s cloud=%s machine=%s cloudspace_status=%s auto_sleep=%v idle_seconds=%d\n", st.Name(), st.Cloud(), st.Machine(), st.Status(), st.AutoSleep(), st.AutoSleepTime())
	client := &http.Client{Timeout: 30 * time.Second}
	studios, err := ts.Studios()
	if err != nil {
		return err
	}
	for _, other := range studios {
		fmt.Printf("visible_studio=%s status=%s\n", other.Name(), other.Status())
		probe, _ := http.NewRequest("GET", "https://lightning.ai/v1/projects/"+ts.ID()+"/cloudspaces/"+other.ID()+"/codestatus", nil)
		probe.SetBasicAuth(cfg.UserID, cfg.APIKey)
		resp, e := client.Do(probe)
		if e != nil {
			return e
		}
		var state map[string]interface{}
		e = json.NewDecoder(resp.Body).Decode(&state)
		resp.Body.Close()
		if e != nil {
			return e
		}
		for _, key := range []string{"requested", "inUse"} {
			b, _ := json.Marshal(state[key])
			fmt.Printf("  %s=%s\n", key, b)
		}
	}
	req, _ := http.NewRequest("GET", "https://lightning.ai/v1/projects/"+ts.ID()+"/clusters/lightning-public-prod/accelerators", nil)
	req.SetBasicAuth(cfg.UserID, cfg.APIKey)
	res, err := client.Do(req)
	if err != nil {
		return err
	}
	var catalog map[string]interface{}
	err = json.NewDecoder(res.Body).Decode(&catalog)
	res.Body.Close()
	if err != nil {
		return err
	}
	fmt.Printf("accelerator_catalog_http=%d\n", res.StatusCode)
	for _, value := range catalog {
		if list, ok := value.([]interface{}); ok {
			for _, item := range list {
				b, _ := json.Marshal(item)
				if strings.Contains(string(b), "lit-l4-1") {
					fmt.Printf("l4_catalog=%s\n", b)
				}
			}
		}
	}
	for _, endpoint := range []string{"codestatus", "required-balance-status"} {
		req, _ := http.NewRequest("GET", "https://lightning.ai/v1/projects/"+ts.ID()+"/cloudspaces/"+st.ID()+"/"+endpoint, nil)
		req.SetBasicAuth(cfg.UserID, cfg.APIKey)
		res, err := client.Do(req)
		if err != nil {
			return err
		}
		var data map[string]interface{}
		err = json.NewDecoder(res.Body).Decode(&data)
		res.Body.Close()
		fmt.Printf("%s http=%d\n", endpoint, res.StatusCode)
		if err != nil {
			return err
		}
		for _, key := range []string{"phase", "status", "state", "isRunning", "requiredBalance", "balance", "hasRequiredBalance", "sufficientBalance", "instance", "codeStatus", "computeConfig", "requested", "inUse", "collabStatus", "reason"} {
			if v, ok := data[key]; ok {
				b, _ := json.Marshal(v)
				fmt.Printf("  %s=%s\n", key, b)
			}
		}
		// Print keys to discover supported non-secret billing fields without dumping payloads.
		for key := range data {
			fmt.Printf("  field=%s\n", key)
		}
	}
	return nil
}
