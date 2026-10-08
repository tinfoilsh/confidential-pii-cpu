package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/google/uuid"
	usage "github.com/tinfoilsh/usage-reporting-go"
	usageclient "github.com/tinfoilsh/usage-reporting-go/client"
)

type eventReporter interface {
	AddEvent(usage.Event)
	Stats() usageclient.Stats
}

// worker is one inference process. The gateway dispatches each request to the
// worker with the fewest requests in flight, so independent processes — each
// with its own interpreter and thread pool — serve concurrent requests
// without contending on a shared lock.
type worker struct {
	target   *url.URL
	proxy    *httputil.ReverseProxy
	inflight atomic.Int64
}

func pickWorker(workers []*worker) *worker {
	best := workers[0]
	for _, w := range workers[1:] {
		if w.inflight.Load() < best.inflight.Load() {
			best = w
		}
	}
	return best
}

// The shim authenticates the bearer credential before this handler is reached.
// The inference servers are private; only this handler emits billing events.
func newGateway(targets []*url.URL, reporter eventReporter) http.Handler {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = inferenceTimeout
	transport.DisableCompression = true
	workers := make([]*worker, len(targets))
	for i, target := range targets {
		workers[i] = &worker{
			target: target,
			proxy: &httputil.ReverseProxy{
				Transport: transport,
				Rewrite: func(r *httputil.ProxyRequest) {
					r.SetURL(target)
					r.Out.Header.Del("Authorization")
					r.Out.Header.Del(usage.HeaderContext)
					r.Out.Header.Del(usage.HeaderUsageContextSignature)
					r.Out.Header.Del(billableRequestsHeader)
					r.Out.Header.Del("Accept-Encoding")
				},
				ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
					slog.Error("privacy filter upstream unavailable")
					http.Error(w, "privacy filter unavailable", http.StatusBadGateway)
				},
			},
		}
	}

	healthClient := &http.Client{Transport: transport, Timeout: healthProbeTimeout}

	mux := http.NewServeMux()
	mux.HandleFunc("POST "+redactPath, func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Fields(r.Header.Get("Authorization"))
		if len(r.Header.Values("Authorization")) != 1 || len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") {
			http.Error(w, "bearer credential required", http.StatusUnauthorized)
			return
		}
		// Read a bounded body before proxying so oversize inputs never reach inference.
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBytes))
		if err != nil {
			http.Error(w, "invalid or oversized request body", http.StatusRequestEntityTooLarge)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		ctx, cancel := context.WithTimeout(r.Context(), inferenceTimeout)
		defer cancel()
		picked := pickWorker(workers)
		picked.inflight.Add(1)
		defer picked.inflight.Add(-1)
		requestProxy := *picked.proxy
		requestProxy.ModifyResponse = func(resp *http.Response) error {
			resp.Header.Del(billableRequestsHeader)
			if resp.StatusCode == http.StatusOK {
				reporter.AddEvent(usage.Event{
					RequestID:        uuid.NewString(),
					APIKey:           parts[1],
					Operation:        usage.Operation{Service: usage.ServicePIIFilter, Name: usage.OperationPIIFilterRedact},
					CustomerRequests: 1,
				})
				resp.Header.Set(billableRequestsHeader, "1")
			}
			return nil
		}
		requestProxy.ServeHTTP(w, r.WithContext(ctx))
	})
	// Healthy means every worker is healthy: a single wedged worker must fail
	// the container's healthcheck (and the boot gate) even while its siblings
	// still serve, matching the single-worker semantics.
	mux.HandleFunc("GET "+healthPath, func(w http.ResponseWriter, r *http.Request) {
		var wg sync.WaitGroup
		failures := make([]error, len(workers))
		for i, wk := range workers {
			wg.Add(1)
			go func(i int, target *url.URL) {
				defer wg.Done()
				failures[i] = probeHealth(r.Context(), healthClient, target.String()+healthPath)
			}(i, wk.target)
		}
		wg.Wait()
		for i, err := range failures {
			if err != nil {
				slog.Warn("worker unhealthy", "worker", i, "error", err)
				http.Error(w, fmt.Sprintf("worker %d unhealthy", i), http.StatusServiceUnavailable)
				return
			}
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	mux.HandleFunc("GET "+metricsPath, func(w http.ResponseWriter, r *http.Request) {
		// /metrics is admin-authenticated by the shim, just like Python's
		// metrics. Workers export identical unlabeled series, which cannot be
		// concatenated into valid Prometheus text, so this serves worker 0's
		// model metrics plus the gateway's usage totals. Aggregating
		// per-worker series (with a worker label) is a follow-up.
		metricsProxy := *workers[0].proxy
		metricsProxy.ModifyResponse = func(resp *http.Response) error {
			if resp.StatusCode != http.StatusOK {
				return nil
			}
			stats := reporter.Stats()
			suffix := fmt.Sprintf("\npii_usage_enqueued_total %d\npii_usage_delivered_total %d\npii_usage_failed_total %d\npii_usage_dropped_total %d\n",
				stats.Enqueued, stats.DeliveredEvents, stats.FailedEvents, stats.DroppedDisabled+stats.DroppedBufferFull)
			resp.Body = &appendedBody{Reader: io.MultiReader(resp.Body, strings.NewReader(suffix)), Closer: resp.Body}
			resp.ContentLength = -1
			resp.Header.Del("Content-Length")
			return nil
		}
		metricsProxy.ServeHTTP(w, r)
	})
	return mux
}

type appendedBody struct {
	io.Reader
	io.Closer
}

func probeHealth(ctx context.Context, client *http.Client, url string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	return nil
}
