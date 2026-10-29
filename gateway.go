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
	index    int
	target   *url.URL
	proxy    *httputil.ReverseProxy
	inflight atomic.Int64 // /redact requests being coordinated
	forwards int          // model forwards running; guarded by forwardScheduler.mu
	// wedged is set when a forward on this worker exceeded forwardTimeout
	// (a forward that outlived ~10 windows' worth of time is not coming
	// back) and cleared when the supervisor has replaced the process. While
	// set, the worker gets no new windows or requests and /health fails.
	wedged atomic.Bool
}

// pickWorker chooses the worker to coordinate a /redact: fewest in flight,
// skipping any being replaced. nil when none is usable.
func pickWorker(workers []*worker) *worker {
	var best *worker
	for _, w := range workers {
		if w.wedged.Load() {
			continue
		}
		if best == nil || w.inflight.Load() < best.inflight.Load() {
			best = w
		}
	}
	return best
}

// gateway is the front end plus the handles the supervisor needs: which
// worker wedged, and a nudge to resume dispatch once it is back.
type gateway struct {
	http.Handler
	workers   []*worker
	scheduler *forwardScheduler
	onWedge   func(worker int)
}

// resume lets the scheduler use a replaced worker again.
func (g *gateway) resume(int) { g.scheduler.dispatch() }

// The shim authenticates the bearer credential before this handler is reached.
// The inference servers are private; only this handler emits billing events.
//
// A /redact request is coordinated by one worker, which tokenizes, cuts the
// input into overlapping windows and posts each window back to this
// gateway's /forward. The forward scheduler queues windows per caller and
// runs them on whichever worker has a free lane, so long documents use every
// worker and short requests from other callers interleave with them.
func newGateway(targets []*url.URL, reporter eventReporter, limits forwardLimits) http.Handler {
	return buildGateway(targets, reporter, limits, "")
}

func newGatewayWithUsageContext(targets []*url.URL, reporter eventReporter, limits forwardLimits, usageContextSecret string) http.Handler {
	return buildGateway(targets, reporter, limits, usageContextSecret)
}

// newGatewayWithUsageContext also verifies the signed usage context a parent
// service (websearch under a router chat) sends with a call, so the usage
// event carries root_request_id / parent_service / depth and the control
// plane can tell a nested tool call from a customer's own request.
func buildGateway(targets []*url.URL, reporter eventReporter, limits forwardLimits, usageContextSecret string) *gateway {
	g := &gateway{}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = limits.maxRequest
	transport.DisableCompression = true
	transport.IdleConnTimeout = workerIdleConnTimeout
	workers := make([]*worker, len(targets))
	for i, target := range targets {
		workers[i] = &worker{
			index:  i,
			target: target,
			proxy: &httputil.ReverseProxy{
				Transport: transport,
				Rewrite: func(r *httputil.ProxyRequest) {
					r.SetURL(target)
					r.Out.Header.Del("Authorization")
					r.Out.Header.Del(usage.HeaderContext)
					r.Out.Header.Del(usage.HeaderUsageContextSignature)
					r.Out.Header.Del(billableRequestsHeader)
					r.Out.Header.Del(billableInputTokensHeader)
					r.Out.Header.Del("Accept-Encoding")
					r.Out.Header.Del(keyHashHeader)
					r.Out.Header.Del(requestIDHeader)
					r.Out.Header.Del(inputTokensHeader)
				},
				ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
					// Also reached when a 200 carried no token count: fail closed
					// rather than return a result we cannot meter.
					if !errors.Is(err, errMissingInputTokens) {
						slog.Error("privacy filter upstream unavailable", "error", err, "worker", r.URL.Host)
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
		parent, err := usageContext(r.Header, parts[1], usageContextSecret)
		if err != nil {
			slog.Warn("rejected usage context", "error", err)
			http.Error(w, "invalid usage context", http.StatusBadRequest)
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
		if picked == nil {
			http.Error(w, "privacy filter unavailable", http.StatusServiceUnavailable)
			return
		}
		picked.inflight.Add(1)
		defer picked.inflight.Add(-1)
		requestProxy := *picked.proxy
		baseRewrite, hash, requestID := picked.proxy.Rewrite, keyHash(parts[1]), uuid.NewString()
		// Whatever ends this request - answered, deadline, client gone -
		// nothing of it stays queued for a lane.
		defer scheduler.cancelRequest(requestID)
		requestProxy.Rewrite = func(pr *httputil.ProxyRequest) {
			baseRewrite(pr)
			pr.Out.URL.Path = chunkPath // the worker coordinates this /redact
			pr.Out.Header.Set(keyHashHeader, hash)
			pr.Out.Header.Set(requestIDHeader, requestID)
		}
		requestProxy.ModifyResponse = func(resp *http.Response) error {
			resp.Header.Del(billableRequestsHeader)
			// The coordinator reports the input token count it tokenized on an
			// internal header; it rides on the usage event as an input_tokens
			// meter and is echoed back as the billing receipt detail.
			inputTokens, err := strconv.ParseInt(resp.Header.Get(inputTokensHeader), 10, 64)
			resp.Header.Del(inputTokensHeader)
			resp.Header.Del(billableInputTokensHeader)
			if resp.StatusCode == http.StatusOK {
				if err != nil || inputTokens < 0 {
					slog.Error("redact response carried no input token count")
					return errMissingInputTokens
				}
				// Reported in the router's model_request shape so the control
				// plane prices it from the model catalog: input tokens at the
				// per-token rate plus the per-request price (the floor). A
				// parent that already counted the customer's request (its
				// context says so) is not charged a second request; the
				// tokens still are, as in websearch.
				customerRequests := int64(1)
				if !parent.billRequest {
					customerRequests = 0
				}
				reporter.AddEvent(usage.Event{
					RequestID:        requestID,
					APIKey:           parts[1],
					Operation:        usage.Operation{Service: usage.ServiceRouter, Name: usage.OperationRouterModelRequest},
					CustomerRequests: customerRequests,
					Meters:           []usage.Meter{{Name: usage.MeterInputTokens, Quantity: inputTokens}},
					Attributes:       eventAttributes(parent.attributes),
				})
				resp.Header.Set(billableRequestsHeader, "1")
				resp.Header.Set(billableInputTokensHeader, strconv.FormatInt(inputTokens, 10))
			}
			return nil
		}
		requestProxy.ServeHTTP(w, r.WithContext(ctx))
	})
	// Internal: one model forward over one window (binary in, binary out;
	// the payloads are opaque here). Reached only from the coordinating
	// workers on this host; the shim never exposes it.
	mux.HandleFunc("POST "+queuePath, func(w http.ResponseWriter, r *http.Request) {
		if !isLoopback(r.RemoteAddr) {
			http.Error(w, "forward is internal", http.StatusForbidden)
			return
		}
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBytes))
		if err != nil {
			http.Error(w, "invalid or oversized request body", http.StatusRequestEntityTooLarge)
			return
		}
		// Window length from the int32 body; the request's total from the
		// coordinator's header (absent → long, the conservative reading).
		requestTokens, _ := strconv.Atoi(r.Header.Get(requestTokensHeader))
		short := requestTokens > 0 && requestTokens <= limits.shortTokens
		item, err := scheduler.enqueue(r.Header.Get(keyHashHeader), r.Header.Get(requestIDHeader), len(body)/4, short)
		if err != nil {
			http.Error(w, "privacy filter is at capacity", http.StatusServiceUnavailable)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), limits.maxRequest)
		defer cancel()
		var slot *lane
		select {
		case slot = <-item.ready:
		case <-ctx.Done():
			if !scheduler.cancel(item) {
				if assigned := <-item.ready; assigned != nil {
					scheduler.release(assigned, false)
				}
			}
			http.Error(w, "privacy filter is at capacity", http.StatusServiceUnavailable)
			return
		}
		if slot == nil {
			// The /redact this window belongs to has ended; the coordinator
			// is being torn down and nothing of it should run.
			http.Error(w, "request no longer active", http.StatusConflict)
			return
		}
		picked := slot.worker
		completed := false
		defer func() { scheduler.release(slot, completed) }()
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
		// The window's absolute token offset: the worker runs the window at
		// its true positions so rotary angles match an unchunked pass exactly.
		if offset := r.Header.Get(positionOffsetHeader); offset != "" {
			req.Header.Set(positionOffsetHeader, offset)
		}
		resp, err := forwardClient.Do(req)
		if err != nil {
			if errors.Is(forwardCtx.Err(), context.DeadlineExceeded) && ctx.Err() == nil {
				forwardTimeouts.Add(1)
				picked.wedged.Store(true)
				slog.Error("forward timed out; worker marked unhealthy", "worker", picked.target.Host, "timeout", limits.forwardTimeout)
				if g.onWedge != nil {
					go g.onWedge(picked.index)
				}
				http.Error(w, "privacy filter forward timed out", http.StatusGatewayTimeout)
				return
			}
			slog.Error("forward upstream unavailable", "error", err)
			http.Error(w, "privacy filter unavailable", http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()
		completed = resp.StatusCode == http.StatusOK
		if !completed {
			slog.Warn("forward failed", "status", resp.StatusCode, "elapsed", time.Since(start))
		}
		w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
		if shape := resp.Header.Get(forwardShapeHeader); shape != "" {
			w.Header().Set(forwardShapeHeader, shape)
		}
		// Relay the length so a body cut short by a worker hiccup is a
		// transport error at the coordinator (retried) rather than a
		// silently short score matrix.
		if resp.ContentLength >= 0 {
			w.Header().Set("Content-Length", strconv.FormatInt(resp.ContentLength, 10))
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
			fs := scheduler.stats()
			suffix := fmt.Sprintf("\npii_usage_enqueued_total %d\npii_usage_delivered_total %d\npii_usage_failed_total %d\npii_usage_dropped_total %d\npii_forward_queued %d\npii_forward_inflight %d\npii_forward_timeouts_total %d\npii_forward_hold_seconds_total %.3f\npii_forward_ms_per_token %.4f\n",
				stats.Enqueued, stats.DeliveredEvents, stats.FailedEvents, stats.DroppedDisabled+stats.DroppedBufferFull, fs.queued, fs.inflight, forwardTimeouts.Load(), fs.holdSeconds, fs.msPerToken) + fs.shortWaitMetrics()
			resp.Body = &appendedBody{Reader: io.MultiReader(resp.Body, strings.NewReader(suffix)), Closer: resp.Body}
			resp.ContentLength = -1
			resp.Header.Del("Content-Length")
			return nil
		}
		metricsProxy.ServeHTTP(w, r)
	})
	g.Handler, g.workers, g.scheduler = mux, workers, scheduler
	return g
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

// parentContext is what a verified parent usage context contributes to the
// usage event: nesting attributes, and whether the parent has already
// counted this call as the customer's request.
type parentContext struct {
	attributes map[string]string
	// billRequest is false when the parent says it already counted the
	// customer request; the filter's own per-call fee still applies.
	billRequest bool
}

// usageContext verifies the parent's signed usage context, if any. No
// context (a direct customer call) yields a direct, billable call; a present
// but invalid one is an error, as in websearch.
func usageContext(header http.Header, apiKey, secret string) (parentContext, error) {
	direct := parentContext{billRequest: true}
	if secret == "" {
		return direct, nil
	}
	ctx, present, err := usage.FromHeaders(header, secret, time.Now(), usageContextMaxSkew)
	if err != nil {
		return parentContext{}, err
	}
	if !present {
		return direct, nil
	}
	if !usage.VerifyAPIKeyHash(apiKey, ctx.APIKeyHash) {
		return parentContext{}, errors.New("usage context api key mismatch")
	}
	if ctx.Depth > usageContextMaxDepth {
		return parentContext{}, fmt.Errorf("usage context depth %d exceeds %d", ctx.Depth, usageContextMaxDepth)
	}
	attributes := map[string]string{}
	if ctx.ContextID != "" {
		attributes["context_id"] = ctx.ContextID
	}
	if ctx.RootRequestID != "" {
		attributes["root_request_id"] = ctx.RootRequestID
	}
	if ctx.ParentService != "" {
		attributes["parent_service"] = ctx.ParentService
	}
	if ctx.Depth > 0 {
		attributes["depth"] = strconv.Itoa(ctx.Depth)
	}
	return parentContext{attributes: attributes, billRequest: ctx.BillCustomerRequest}, nil
}

func eventAttributes(contextAttributes map[string]string) map[string]string {
	attributes := map[string]string{"model": billingModel, "route": redactPath, "streaming": "false"}
	for k, v := range contextAttributes {
		attributes[k] = v
	}
	return attributes
}
