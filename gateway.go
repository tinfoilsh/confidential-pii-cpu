package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	usage "github.com/tinfoilsh/usage-reporting-go"
	usageclient "github.com/tinfoilsh/usage-reporting-go/client"
)

var errMissingInputTokens = errors.New("redact response carried no input token count")

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
	inflight atomic.Int64 // /redact requests being coordinated
	forwards int          // model forwards running; guarded by forwardScheduler.mu
	// wedged is set when a forward on this worker exceeded forwardTimeout.
	// It is sticky: /health fails until the container is replaced, since a
	// forward that outlived ~40 windows' worth of time is not coming back.
	wedged atomic.Bool
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
//
// A /redact request is coordinated by one worker, which tokenizes, cuts the
// input into overlapping windows and posts each window back to this
// gateway's /forward. The forward scheduler queues windows per caller and
// runs them on whichever worker has a free lane, so long documents use every
// worker and short requests from other callers interleave with them.
func newGateway(targets []*url.URL, reporter eventReporter, limits forwardLimits) http.Handler {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = limits.maxRequest
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
					r.Out.Header.Del(keyHashHeader)
					r.Out.Header.Del(inputTokensHeader)
				},
				ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
					// Also reached when a 200 carried no token count: fail closed
					// rather than return a result we cannot meter.
					if !errors.Is(err, errMissingInputTokens) {
						slog.Error("privacy filter upstream unavailable")
					}
					http.Error(w, "privacy filter unavailable", http.StatusBadGateway)
				},
			},
		}
	}

	healthClient := &http.Client{Transport: transport, Timeout: healthProbeTimeout}
	forwardClient := &http.Client{Transport: transport}
	scheduler := newForwardScheduler(workers, limits)
	var forwardTimeouts atomic.Int64

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
		ctx, cancel := context.WithTimeout(r.Context(), limits.requestTimeout(len(body)))
		defer cancel()
		picked := pickWorker(workers)
		picked.inflight.Add(1)
		defer picked.inflight.Add(-1)
		requestProxy := *picked.proxy
		baseRewrite, hash := picked.proxy.Rewrite, keyHash(parts[1])
		requestProxy.Rewrite = func(pr *httputil.ProxyRequest) {
			baseRewrite(pr)
			pr.Out.Header.Set(keyHashHeader, hash)
		}
		requestProxy.ModifyResponse = func(resp *http.Response) error {
			resp.Header.Del(billableRequestsHeader)
			// The coordinator reports the input token count it tokenized; it
			// rides on the usage event as an input_tokens meter (one billable
			// request as before) and never reaches the client.
			inputTokens, err := strconv.ParseInt(resp.Header.Get(inputTokensHeader), 10, 64)
			resp.Header.Del(inputTokensHeader)
			if resp.StatusCode == http.StatusOK {
				if err != nil || inputTokens < 0 {
					slog.Error("redact response carried no input token count")
					return errMissingInputTokens
				}
				reporter.AddEvent(usage.Event{
					RequestID:        uuid.NewString(),
					APIKey:           parts[1],
					Operation:        usage.Operation{Service: usage.ServicePIIFilter, Name: usage.OperationPIIFilterRedact},
					CustomerRequests: 1,
					Meters:           []usage.Meter{{Name: usage.MeterInputTokens, Quantity: inputTokens}},
				})
				resp.Header.Set(billableRequestsHeader, "1")
			}
			return nil
		}
		requestProxy.ServeHTTP(w, r.WithContext(ctx))
	})
	// Internal: one model forward over one window (binary in, binary out;
	// the payloads are opaque here). Reached only from the coordinating
	// workers on this host; the shim never exposes it.
	mux.HandleFunc("POST "+forwardPath, func(w http.ResponseWriter, r *http.Request) {
		if !isLoopback(r.RemoteAddr) {
			http.Error(w, "forward is internal", http.StatusForbidden)
			return
		}
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBytes))
		if err != nil {
			http.Error(w, "invalid or oversized request body", http.StatusRequestEntityTooLarge)
			return
		}
		item, err := scheduler.enqueue(r.Header.Get(keyHashHeader))
		if err != nil {
			http.Error(w, "privacy filter is at capacity", http.StatusServiceUnavailable)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), limits.maxRequest)
		defer cancel()
		var picked *worker
		select {
		case picked = <-item.ready:
		case <-ctx.Done():
			if !scheduler.cancel(item) {
				scheduler.release(<-item.ready)
			}
			http.Error(w, "privacy filter is at capacity", http.StatusServiceUnavailable)
			return
		}
		defer scheduler.release(picked)
		start := time.Now()
		// The queue wait above is bounded by the generous request ceiling;
		// the forward itself gets the sharp per-window deadline.
		forwardCtx, cancelForward := context.WithTimeout(ctx, limits.forwardTimeout)
		defer cancelForward()
		req, err := http.NewRequestWithContext(forwardCtx, http.MethodPost, picked.target.String()+forwardPath, bytes.NewReader(body))
		if err != nil {
			http.Error(w, "privacy filter unavailable", http.StatusBadGateway)
			return
		}
		req.Header.Set("Content-Type", r.Header.Get("Content-Type"))
		resp, err := forwardClient.Do(req)
		if err != nil {
			if errors.Is(forwardCtx.Err(), context.DeadlineExceeded) && ctx.Err() == nil {
				forwardTimeouts.Add(1)
				picked.wedged.Store(true)
				slog.Error("forward timed out; worker marked unhealthy", "worker", picked.target.Host, "timeout", limits.forwardTimeout)
				http.Error(w, "privacy filter forward timed out", http.StatusGatewayTimeout)
				return
			}
			slog.Error("forward upstream unavailable", "error", err)
			http.Error(w, "privacy filter unavailable", http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			slog.Warn("forward failed", "status", resp.StatusCode, "elapsed", time.Since(start))
		}
		w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
		if shape := resp.Header.Get(forwardShapeHeader); shape != "" {
			w.Header().Set(forwardShapeHeader, shape)
		}
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
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
		for i, wk := range workers {
			if wk.wedged.Load() {
				failures[i] = errors.New("forward timed out")
			}
		}
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
			queued, inflight := scheduler.stats()
			suffix := fmt.Sprintf("\npii_usage_enqueued_total %d\npii_usage_delivered_total %d\npii_usage_failed_total %d\npii_usage_dropped_total %d\npii_forward_queued %d\npii_forward_inflight %d\npii_forward_timeouts_total %d\n",
				stats.Enqueued, stats.DeliveredEvents, stats.FailedEvents, stats.DroppedDisabled+stats.DroppedBufferFull, queued, inflight, forwardTimeouts.Load())
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
