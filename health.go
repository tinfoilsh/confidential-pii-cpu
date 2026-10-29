package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"time"
)

// watchHealth polls the inference server's health endpoint and reports on the
// returned channel once it has failed healthFailureThreshold times in a row.
// Unless the caller has already seen the worker healthy (ready), failures
// before the first success are the model loading, not a wedge.
func watchHealth(ctx context.Context, url string, interval time.Duration, ready bool) <-chan error {
	failed := make(chan error, 1)
	go func() {
		client := &http.Client{Timeout: healthProbeTimeout}
		failures := 0
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
			err := probe(ctx, client, url)
			if err == nil {
				ready = true
				failures = 0
				continue
			}
			if !ready {
				continue
			}
			failures++
			slog.Warn("inference health probe failed", "failures", failures, "error", err)
			if failures >= healthFailureThreshold {
				failed <- err
				return
			}
		}
	}()
	return failed
}

func probe(ctx context.Context, client *http.Client, url string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	return nil
}
