package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	usage "github.com/tinfoilsh/usage-reporting-go"
	usageclient "github.com/tinfoilsh/usage-reporting-go/client"
)

func TestGatewayOwnsBilling(t *testing.T) {
	const secret = "test-reporting-secret"
	const query = `{"text":"private@example.com"}`
	for _, tc := range []struct {
		name, method, path, auth               string
		upstreamStatus, wantStatus, wantEvents int
	}{
		{"redact with detections", "POST", redactPath, "Bearer tk_customer", 200, 200, 1},
		{"redact with no detections", "POST", redactPath, "Bearer tk_clean", 200, 200, 1},
		{"validation error", "POST", redactPath, "Bearer tk_customer", 422, 422, 0},
		{"inference error", "POST", redactPath, "Bearer tk_customer", 500, 500, 0},
		{"no credential", "POST", redactPath, "", 200, 401, 0},
		{"malformed credential", "POST", redactPath, "Basic bad", 200, 401, 0},
		{"health", "GET", healthPath, "", 200, 200, 0},
		{"metrics", "GET", metricsPath, "Bearer tk_admin", 200, 200, 0},
		{"unsupported path", "POST", "/v1/chat/completions", "Bearer tk_customer", 200, 404, 0},
		{"wrong method", "GET", redactPath, "Bearer tk_customer", 200, 405, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			responseBody := `{"detected_spans":[],"redacted_text":"unchanged"}`
			if tc.name == "redact with detections" {
				responseBody = `{"detected_spans":[{"label":"private_email","text":"private@example.com"}],"redacted_text":"<PRIVATE_EMAIL>"}`
			}
			batches := make(chan usage.Batch, 2)
			ingestion := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				id, ts, nonce, sig, err := usage.HeaderValues(r.Header)
				if err != nil || !usage.VerifyBatch(r.Method, r.URL.Path, id, ts, nonce, body, secret, sig) {
					t.Error("invalid usage batch signature")
					w.WriteHeader(http.StatusUnauthorized)
					return
				}
				if strings.Contains(string(body), "private@example.com") {
					t.Error("billing event leaked request text")
				}
				var batch usage.Batch
				if err := json.Unmarshal(body, &batch); err != nil {
					t.Error(err)
				}
				batches <- batch
			}))
			defer ingestion.Close()
			reporter := usageclient.New(usageclient.Config{Endpoint: ingestion.URL + usage.IngestionPath, ReporterID: reporterID, Secret: secret, HTTPClient: ingestion.Client(), FlushInterval: time.Hour})
			defer reporter.Stop(context.Background())
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "" || r.Header.Get(usage.HeaderContext) != "" {
					t.Error("credentials or billing context reached model process")
				}
				if r.URL.Path == chunkPath {
					body, _ := io.ReadAll(r.Body)
					if string(body) != query {
						t.Errorf("body changed: %s", body)
					}
				}
				w.Header().Set("Content-Type", "application/json")
				if r.URL.Path == chunkPath {
					w.Header().Set(inputTokensHeader, "7")
				}
				w.WriteHeader(tc.upstreamStatus)
				_, _ = io.WriteString(w, responseBody)
			}))
			defer backend.Close()
			target, _ := url.Parse(backend.URL)
			handler := newGateway([]*url.URL{target}, reporter, testLimits)
			request := httptest.NewRequest(tc.method, tc.path, strings.NewReader(query))
			if tc.auth != "" {
				request.Header.Set("Authorization", tc.auth)
			}
			request.Header.Set(billableRequestsHeader, "0")
			request.Header.Set(inputTokensHeader, "0")
			request.Header.Set(usage.HeaderContext, "attacker-requests-free-inference")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != tc.wantStatus {
				t.Fatalf("status %d, want %d", response.Code, tc.wantStatus)
			}
			if response.Header().Get(inputTokensHeader) != "" {
				t.Fatal("input token count leaked to the client")
			}
			reporter.Flush(context.Background())
			if tc.wantEvents == 0 {
				if len(batches) != 0 || response.Header().Get(billableRequestsHeader) != "" {
					t.Fatal("non-billable request emitted a charge")
				}
				return
			}
			if len(batches) != 1 {
				t.Fatal("missing usage batch")
			}
			if response.Body.String() != responseBody {
				t.Fatal("inference response was modified")
			}
			batch := <-batches
			if len(batch.Events) != 1 {
				t.Fatalf("events: %+v", batch.Events)
			}
			event := batch.Events[0]
			if event.APIKey != strings.TrimPrefix(tc.auth, "Bearer ") || event.CustomerRequests != 1 || event.Operation.Service != usage.ServiceRouter || event.Operation.Name != usage.OperationRouterModelRequest || event.Attributes["model"] != billingModel || event.Attributes["route"] != redactPath {
				t.Fatalf("incorrect attribution: %+v", event)
			}
			if event.RequestID == "" || event.EventID == "" || response.Header().Get(billableRequestsHeader) != "1" || response.Header().Get(billableInputTokensHeader) != "7" {
				t.Fatalf("missing event identity or receipt: %v", response.Header())
			}
			if len(event.Meters) != 1 || event.Meters[0].Name != usage.MeterInputTokens || event.Meters[0].Quantity != 7 {
				t.Fatalf("input tokens not metered: %+v", event.Meters)
			}
		})
	}
}

type recordedUsage struct{ events []usage.Event }

func (r *recordedUsage) AddEvent(event usage.Event) { r.events = append(r.events, event) }
func (r *recordedUsage) Stats() usageclient.Stats   { return usageclient.Stats{} }

func TestGatewayBoundsAndRequestIdentity(t *testing.T) {
	upstreamCalls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.Header().Set(inputTokensHeader, "1")
		_, _ = io.WriteString(w, `{}`)
	}))
	defer backend.Close()
	target, _ := url.Parse(backend.URL)
	reporter := &recordedUsage{}
	handler := newGateway([]*url.URL{target}, reporter, testLimits)
	oversized := httptest.NewRequest(http.MethodPost, redactPath, strings.NewReader(strings.Repeat("x", maxRequestBytes+1)))
	oversized.Header.Set("Authorization", "Bearer tk_customer")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, oversized)
	if response.Code != http.StatusRequestEntityTooLarge || upstreamCalls != 0 || len(reporter.events) != 0 {
		t.Fatal("oversized input reached inference or billing")
	}
	for range 2 {
		request := httptest.NewRequest(http.MethodPost, redactPath, strings.NewReader(`{"text":"hello"}`))
		request.Header.Set("Authorization", "Bearer tk_customer")
		request.Header.Set("X-Request-Id", "same-caller-id")
		handler.ServeHTTP(httptest.NewRecorder(), request)
	}
	if len(reporter.events) != 2 || reporter.events[0].RequestID == reporter.events[1].RequestID {
		t.Fatal("repeated requests collapsed into one charge")
	}
}

// TestGatewayDispatchesConcurrentRequestsToIdleWorkers holds one worker busy
// and checks that a second concurrent request lands on the other worker
// instead of queueing behind the first.
func TestGatewayDispatchesConcurrentRequestsToIdleWorkers(t *testing.T) {
	release := make(chan struct{})
	firstArrived := make(chan struct{})
	var hits [2]atomic.Int64
	makeBackend := func(i int, slow bool) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			hits[i].Add(1)
			if slow {
				close(firstArrived)
				<-release
			}
			w.Header().Set(inputTokensHeader, "1")
			_, _ = io.WriteString(w, `{}`)
		}))
	}
	slowBackend := makeBackend(0, true)
	defer slowBackend.Close()
	fastBackend := makeBackend(1, false)
	defer fastBackend.Close()
	slowURL, _ := url.Parse(slowBackend.URL)
	fastURL, _ := url.Parse(fastBackend.URL)
	handler := newGateway([]*url.URL{slowURL, fastURL}, &recordedUsage{}, testLimits)
	server := httptest.NewServer(handler)
	defer server.Close()

	send := func() {
		req, _ := http.NewRequest(http.MethodPost, server.URL+redactPath, strings.NewReader(`{"text":"hi"}`))
		req.Header.Set("Authorization", "Bearer tk_customer")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Error(err)
			return
		}
		resp.Body.Close()
	}
	done := make(chan struct{})
	go func() { defer close(done); send() }()
	<-firstArrived // worker 0 is now mid-request
	send()         // must go to worker 1, not queue behind worker 0
	if got := hits[1].Load(); got != 1 {
		close(release)
		t.Fatalf("second request did not reach the idle worker (hits: %d, %d)", hits[0].Load(), got)
	}
	close(release)
	<-done
}

// A 200 without a token count cannot be metered; the gateway fails closed
// instead of returning an unbilled result.
func TestGatewayRefusesUnmeteredSuccess(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{}`)
	}))
	defer backend.Close()
	target, _ := url.Parse(backend.URL)
	reporter := &recordedUsage{}
	handler := newGateway([]*url.URL{target}, reporter, testLimits)
	request := httptest.NewRequest(http.MethodPost, redactPath, strings.NewReader(`{"text":"hello"}`))
	request.Header.Set("Authorization", "Bearer tk_customer")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway || len(reporter.events) != 0 {
		t.Fatalf("unmetered 200 passed through: status %d, events %d", response.Code, len(reporter.events))
	}
}

func TestWorkerCount(t *testing.T) {
	t.Setenv("OPF_WORKERS", "")
	if n, err := workerCount(); n != 1 || err != nil {
		t.Fatalf("default workers: %d, %v", n, err)
	}
	t.Setenv("OPF_WORKERS", "2")
	if n, err := workerCount(); n != 2 || err != nil {
		t.Fatalf("workers: %d, %v", n, err)
	}
	for _, bad := range []string{"0", "-1", "9", "two"} {
		t.Setenv("OPF_WORKERS", bad)
		if _, err := workerCount(); err == nil {
			t.Fatalf("accepted OPF_WORKERS=%q", bad)
		}
	}
}

func TestReportingConfigFailsClosed(t *testing.T) {
	t.Setenv("USAGE_REPORTER_SECRET", "")
	if _, err := reportingConfig(); err == nil {
		t.Fatal("missing reporting secret accepted")
	}
	t.Setenv("USAGE_REPORTER_SECRET", "test-secret")
	for _, endpoint := range []string{"http://localhost", "https://user:password@example.com", "https://example.com?token=secret"} {
		t.Setenv("CONTROL_PLANE_URL", endpoint)
		if _, err := reportingConfig(); err == nil {
			t.Fatalf("invalid endpoint accepted: %s", endpoint)
		}
	}
	t.Setenv("CONTROL_PLANE_URL", "")
	cfg, err := reportingConfig()
	if err != nil || cfg.Endpoint != controlplaneURL+usage.IngestionPath {
		t.Fatalf("config: %+v, %v", cfg, err)
	}
}

// The gateway must drop idle worker connections before uvicorn does, or a
// reused connection races the worker's close and a POST fails with EOF.
func TestWorkerKeepAliveOutlivesGatewayIdle(t *testing.T) {
	if workerIdleConnTimeout*2 > workerKeepAlive {
		t.Fatalf("gateway idle %v must be well under worker keep-alive %v", workerIdleConnTimeout, workerKeepAlive)
	}
}

// A parent service's signed usage context rides onto the event as nesting
// attributes; a forged or mismatched one is refused; none is a direct call.
func TestGatewayPropagatesUsageContext(t *testing.T) {
	const secret, apiKey = "ctx-secret", "tk_customer"
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get(usage.HeaderContext) != "" || r.Header.Get(usage.HeaderUsageContextSignature) != "" {
			t.Error("usage context reached model process")
		}
		w.Header().Set(inputTokensHeader, "3")
		_, _ = io.WriteString(w, `{}`)
	}))
	defer backend.Close()
	target, _ := url.Parse(backend.URL)
	call := func(t *testing.T, sign func(http.Header)) (*recordedUsage, int) {
		t.Helper()
		reporter := &recordedUsage{}
		handler := newGatewayWithUsageContext([]*url.URL{target}, reporter, testLimits, secret)
		request := httptest.NewRequest(http.MethodPost, redactPath, strings.NewReader(`{"text":"hi"}`))
		request.Header.Set("Authorization", "Bearer "+apiKey)
		sign(request.Header)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return reporter, response.Code
	}
	parent := usage.Context{ContextID: "ctx-1", RootRequestID: "root-1", ParentService: usage.ServiceWebsearch, Depth: 2, BillCustomerRequest: true, APIKeyHash: usage.HashAPIKey(apiKey), IssuedAt: time.Now()}

	reporter, code := call(t, func(h http.Header) {
		if err := usage.SetHeaders(h, parent, secret); err != nil {
			t.Fatal(err)
		}
	})
	if code != http.StatusOK || len(reporter.events) != 1 {
		t.Fatalf("nested call: status %d, events %+v", code, reporter.events)
	}
	got := reporter.events[0].Attributes
	if got["parent_service"] != usage.ServiceWebsearch || got["depth"] != "2" || got["root_request_id"] != "root-1" || got["context_id"] != "ctx-1" || got["model"] != billingModel {
		t.Fatalf("attributes: %+v", got)
	}

	if reporter.events[0].CustomerRequests != 1 {
		t.Fatalf("parent billing the request: %+v", reporter.events[0])
	}

	// A parent that already counted the customer's request: tokens metered,
	// no second request.
	counted := parent
	counted.BillCustomerRequest = false
	reporter, code = call(t, func(h http.Header) { _ = usage.SetHeaders(h, counted, secret) })
	if code != http.StatusOK || len(reporter.events) != 1 || reporter.events[0].CustomerRequests != 0 || reporter.events[0].Meters[0].Quantity != 3 {
		t.Fatalf("parent-counted call: status %d, events %+v", code, reporter.events)
	}

	reporter, code = call(t, func(http.Header) {})
	if code != http.StatusOK || len(reporter.events) != 1 || reporter.events[0].Attributes["parent_service"] != "" || reporter.events[0].CustomerRequests != 1 {
		t.Fatalf("direct call: status %d, events %+v", code, reporter.events)
	}

	forged := parent
	forged.APIKeyHash = usage.HashAPIKey("tk_other")
	reporter, code = call(t, func(h http.Header) { _ = usage.SetHeaders(h, forged, secret) })
	if code != http.StatusBadRequest || len(reporter.events) != 0 {
		t.Fatalf("mismatched key hash: status %d, events %+v", code, reporter.events)
	}
	reporter, code = call(t, func(h http.Header) { _ = usage.SetHeaders(h, parent, "wrong-secret") })
	if code != http.StatusBadRequest || len(reporter.events) != 0 {
		t.Fatalf("bad signature: status %d, events %+v", code, reporter.events)
	}
}

// A coordinator that posts a request's windows must stop when the /redact
// ends: here the client goes away mid-request, and the windows still queued
// for that request are refused (409) instead of running, while another
// request's window is untouched.
func TestGatewayCancelsQueuedWindowsWhenRedactEnds(t *testing.T) {
	// One lane, held busy for a while by a window of an unrelated request so
	// everything else queues behind it.
	lane := newFakeWorker(t, 2*time.Second)
	workerSawRequestID := make(chan string, 1)
	// The "coordinator" backend: records the request id the gateway minted
	// and hangs until the client gives up.
	coordinator := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == chunkPath {
			_, _ = io.Copy(io.Discard, r.Body) // read, as a real worker does; the server only watches for disconnects after
			workerSawRequestID <- r.Header.Get(requestIDHeader)
			<-r.Context().Done()
			return
		}
		lane.server.Config.Handler.ServeHTTP(w, r)
	}))
	defer coordinator.Close()
	target, _ := url.Parse(coordinator.URL)
	front := newGateway([]*url.URL{target}, &recordedUsage{}, limitsWith(1, 64))
	gateway := httptest.NewServer(front)
	defer gateway.Close()
	queue := serveQueue(t, front)

	// Occupy the lane.
	go forwardWindow(t, queue, "someone", 1, 1_000_000)
	waitFor(t, func() bool { return lane.busy.Load() == 1 })

	// Client starts a /redact, then disconnects.
	ctx, cancelClient := context.WithCancel(context.Background())
	redact, _ := http.NewRequestWithContext(ctx, http.MethodPost, gateway.URL+redactPath, strings.NewReader(`{"text":"long"}`))
	redact.Header.Set("Authorization", "Bearer tk_customer")
	go func() { _, _ = http.DefaultClient.Do(redact) }()
	requestID := <-workerSawRequestID
	if requestID == "" {
		t.Fatal("coordinator was not told the request id")
	}

	// The coordinator posts two windows of that request and one of another;
	// all queue behind the busy lane.
	type result struct{ status int }
	results := make(chan result, 3)
	post := func(id string) {
		req, _ := http.NewRequest(http.MethodPost, queue+queuePath, bytes.NewReader(make([]byte, 4)))
		req.Header.Set("Content-Type", "application/octet-stream")
		req.Header.Set(keyHashHeader, "caller")
		req.Header.Set(requestIDHeader, id)
		req.Header.Set(requestTokensHeader, "1000000")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			results <- result{0}
			return
		}
		defer resp.Body.Close()
		_, _ = io.Copy(io.Discard, resp.Body)
		results <- result{resp.StatusCode}
	}
	go post(requestID)
	go post(requestID)
	go post("other-request")
	waitFor(t, func() bool { return lane.calls.Load() == 1 }) // nothing else dispatched
	time.Sleep(50 * time.Millisecond)

	cancelClient()
	got := map[int]int{}
	for range 2 {
		select {
		case r := <-results:
			got[r.status]++
		case <-time.After(5 * time.Second):
			t.Fatal("cancelled request's windows were not refused")
		}
	}
	if got[http.StatusConflict] != 2 {
		t.Fatalf("expected two 409s for the ended request, got %v", got)
	}
	// The other request's window is untouched: it runs once the lane frees.
	select {
	case r := <-results:
		if r.status != http.StatusOK {
			t.Fatalf("the other request's window was disturbed: %v", r)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the other request's window never ran")
	}
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("condition not met in time")
		}
		time.Sleep(5 * time.Millisecond)
	}
}
