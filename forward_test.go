package main

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

var testLimits = forwardLimits{maxInflight: 4, maxQueued: 512, windowTokens: 1024, windowOverlap: 128, windowTimeout: 5 * time.Second, maxRequest: 15 * time.Minute, forwardTimeout: 30 * time.Second}

func limitsWith(maxInflight, maxQueued int) forwardLimits {
	l := testLimits
	l.maxInflight, l.maxQueued = maxInflight, maxQueued
	return l
}

// fakeWorker serves /forward with a fixed latency and counts concurrency.
type fakeWorker struct {
	server  *httptest.Server
	latency time.Duration
	busy    atomic.Int64
	maxBusy atomic.Int64
	calls   atomic.Int64
	done    chan struct{}
}

func newFakeWorker(t *testing.T, latency time.Duration) *fakeWorker {
	t.Helper()
	fw := &fakeWorker{latency: latency, done: make(chan struct{})}
	fw.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != forwardPath {
			_, _ = io.WriteString(w, `{}`)
			return
		}
		fw.calls.Add(1)
		busy := fw.busy.Add(1)
		defer fw.busy.Add(-1)
		for {
			observed := fw.maxBusy.Load()
			if busy <= observed || fw.maxBusy.CompareAndSwap(observed, busy) {
				break
			}
		}
		select {
		case <-time.After(fw.latency):
		case <-fw.done: // test over: let a "stuck" handler return so Close does not hang
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set(forwardShapeHeader, "1,1")
		_, _ = w.Write([]byte{0, 0, 0, 0})
	}))
	t.Cleanup(func() {
		close(fw.done)
		fw.server.Close()
	})
	return fw
}

func (fw *fakeWorker) url() *url.URL {
	u, _ := url.Parse(fw.server.URL)
	return u
}

func forwardRequest(t *testing.T, gateway, key string) (int, time.Duration) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, gateway+forwardPath, bytes.NewReader([]byte{1, 0, 0, 0}))
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set(keyHashHeader, key)
	start := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Error(err)
		return 0, 0
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode, time.Since(start)
}

// One caller with 50 queued windows must not starve a caller with one: the
// single window completes within about two window-times.
func TestForwardSchedulerIsFairAcrossCallers(t *testing.T) {
	const windowTime = 40 * time.Millisecond
	workers := []*url.URL{newFakeWorker(t, windowTime).url(), newFakeWorker(t, windowTime).url()}
	gateway := httptest.NewServer(newGateway(workers, &recordedUsage{}, limitsWith(2, 512)))
	defer gateway.Close()

	var wg sync.WaitGroup
	for range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if status, _ := forwardRequest(t, gateway.URL, "big-caller"); status != http.StatusOK {
				t.Errorf("big caller window: %d", status)
			}
		}()
	}
	time.Sleep(windowTime / 2) // the big caller's windows are queued and both lanes busy
	status, elapsed := forwardRequest(t, gateway.URL, "small-caller")
	wg.Wait()
	if status != http.StatusOK {
		t.Fatalf("small caller window: %d", status)
	}
	if elapsed > 3*windowTime {
		t.Fatalf("small caller waited %v behind a 50-window caller (window time %v)", elapsed, windowTime)
	}
}

func TestForwardSchedulerCapsInFlightAndBalancesWorkers(t *testing.T) {
	const windowTime = 30 * time.Millisecond
	fakes := []*fakeWorker{newFakeWorker(t, windowTime), newFakeWorker(t, windowTime)}
	gateway := httptest.NewServer(newGateway([]*url.URL{fakes[0].url(), fakes[1].url()}, &recordedUsage{}, limitsWith(2, 512)))
	defer gateway.Close()

	var wg sync.WaitGroup
	for i := range 12 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if status, _ := forwardRequest(t, gateway.URL, "caller"); status != http.StatusOK {
				t.Errorf("window %d: %d", i, status)
			}
		}(i)
	}
	wg.Wait()
	for i, fw := range fakes {
		if fw.maxBusy.Load() != 1 {
			t.Fatalf("worker %d ran %d forwards at once; cap is 1 per worker", i, fw.maxBusy.Load())
		}
		if fw.calls.Load() != 6 {
			t.Fatalf("worker %d ran %d of 12 windows; dispatch is not least-busy", i, fw.calls.Load())
		}
	}
}

func TestForwardSchedulerShedsLoadOverQueueBudget(t *testing.T) {
	fw := newFakeWorker(t, 50*time.Millisecond)
	gateway := httptest.NewServer(newGateway([]*url.URL{fw.url()}, &recordedUsage{}, limitsWith(1, 2)))
	defer gateway.Close()

	var wg sync.WaitGroup
	var ok, shed atomic.Int64
	for range 6 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			switch status, _ := forwardRequest(t, gateway.URL, "caller"); status {
			case http.StatusOK:
				ok.Add(1)
			case http.StatusServiceUnavailable:
				shed.Add(1)
			default:
				t.Errorf("unexpected status %d", status)
			}
		}()
		time.Sleep(5 * time.Millisecond) // arrive in order: 1 running, 2 queued, 3 shed
	}
	wg.Wait()
	if ok.Load() != 3 || shed.Load() != 3 {
		t.Fatalf("ok=%d shed=%d; want 3 served and 3 refused with a queue budget of 2", ok.Load(), shed.Load())
	}
	if status, _ := forwardRequest(t, gateway.URL, "caller"); status != http.StatusOK {
		t.Fatalf("queue did not drain: %d", status)
	}
}

func TestForwardRelaysBinaryPayloadAndShape(t *testing.T) {
	var gotBody []byte
	var gotType string
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		gotType = r.Header.Get("Content-Type")
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set(forwardShapeHeader, "2,3")
		_, _ = w.Write([]byte{1, 2, 3, 4, 5, 6})
	}))
	defer worker.Close()
	target, _ := url.Parse(worker.URL)
	gateway := httptest.NewServer(newGateway([]*url.URL{target}, &recordedUsage{}, testLimits))
	defer gateway.Close()
	req, _ := http.NewRequest(http.MethodPost, gateway.URL+forwardPath, bytes.NewReader([]byte{9, 0, 0, 0}))
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if !bytes.Equal(gotBody, []byte{9, 0, 0, 0}) || gotType != "application/octet-stream" {
		t.Fatalf("worker got %v %q", gotBody, gotType)
	}
	if resp.StatusCode != 200 || resp.Header.Get(forwardShapeHeader) != "2,3" || resp.Header.Get("Content-Type") != "application/octet-stream" || !bytes.Equal(body, []byte{1, 2, 3, 4, 5, 6}) {
		t.Fatalf("relayed response: %d %v %v", resp.StatusCode, resp.Header, body)
	}
}

// A forward that outlives the per-dispatch deadline fails (504, so the
// owning /redact fails without a billing event), is counted, and marks the
// worker unhealthy so the container gets replaced.
func TestForwardTimeoutMarksWorkerWedged(t *testing.T) {
	stuck := newFakeWorker(t, time.Hour)
	healthy := newFakeWorker(t, 0)
	limits := limitsWith(2, 512)
	limits.forwardTimeout = 50 * time.Millisecond
	handler := newGateway([]*url.URL{stuck.url(), healthy.url()}, &recordedUsage{}, limits)
	gateway := httptest.NewServer(handler)
	defer gateway.Close()

	health := func() int {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, healthPath, nil))
		return rec.Code
	}
	if health() != http.StatusOK {
		t.Fatal("workers unhealthy before any forward")
	}
	status, elapsed := forwardRequest(t, gateway.URL, "caller")
	if status != http.StatusGatewayTimeout || elapsed > time.Second {
		t.Fatalf("stuck forward: status %d after %v", status, elapsed)
	}
	if health() != http.StatusServiceUnavailable {
		t.Fatal("wedged worker still reported healthy")
	}
	metrics := httptest.NewRecorder()
	handler.ServeHTTP(metrics, httptest.NewRequest(http.MethodGet, metricsPath, nil))
	if !strings.Contains(metrics.Body.String(), "pii_forward_timeouts_total 1\n") {
		t.Fatalf("timeout not counted:\n%s", metrics.Body.String())
	}
}

func TestForwardIsLoopbackOnly(t *testing.T) {
	fw := newFakeWorker(t, 0)
	handler := newGateway([]*url.URL{fw.url()}, &recordedUsage{}, testLimits)
	for _, tc := range []struct {
		remote string
		want   int
	}{
		{"127.0.0.1:4242", http.StatusOK},
		{"[::1]:4242", http.StatusOK},
		{"192.0.2.1:4242", http.StatusForbidden},
		{"172.31.255.1:4242", http.StatusForbidden},
	} {
		req := httptest.NewRequest(http.MethodPost, forwardPath, bytes.NewReader([]byte{1, 0, 0, 0}))
		req.RemoteAddr = tc.remote
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != tc.want {
			t.Fatalf("remote %s: status %d, want %d", tc.remote, rec.Code, tc.want)
		}
	}
	if fw.calls.Load() != 2 {
		t.Fatalf("non-loopback request reached a worker (%d calls)", fw.calls.Load())
	}
}

// The /redact proxy stamps the caller's key hash on the outbound request
// (never the credential) and ignores any hash the client supplies.
func TestRedactCarriesKeyHashToCoordinator(t *testing.T) {
	var seen atomic.Value
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen.Store(r.Header.Get(keyHashHeader))
		w.Header().Set(inputTokensHeader, "1")
		_, _ = io.WriteString(w, `{}`)
	}))
	defer backend.Close()
	target, _ := url.Parse(backend.URL)
	handler := newGateway([]*url.URL{target}, &recordedUsage{}, testLimits)
	req := httptest.NewRequest(http.MethodPost, redactPath, strings.NewReader(`{"text":"hi"}`))
	req.Header.Set("Authorization", "Bearer tk_customer")
	req.Header.Set(keyHashHeader, "spoofed")
	handler.ServeHTTP(httptest.NewRecorder(), req)
	got, _ := seen.Load().(string)
	if got != keyHash("tk_customer") || got == "spoofed" || strings.Contains(got, "tk_customer") {
		t.Fatalf("coordinator saw key hash %q", got)
	}
}

func TestRequestTimeoutScalesWithInputSize(t *testing.T) {
	l := testLimits // stride 768 tokens ≈ 3072 bytes per window, 5 s each, 15 min ceiling
	if got := l.requestTimeout(10); got != requestBaseTimeout+5*time.Second {
		t.Fatalf("tiny request: %v", got)
	}
	if got := l.requestTimeout(100 * 3072); got != requestBaseTimeout+100*5*time.Second {
		t.Fatalf("100-window request: %v", got)
	}
	if got := l.requestTimeout(250 * 3072); got != l.maxRequest { // 30 s + 250 × 5 s > 15 min
		t.Fatalf("250-window request not capped at the ceiling: %v", got)
	}
	if got := l.requestTimeout(maxRequestBytes); got != l.maxRequest {
		t.Fatalf("4 MiB request not capped at the ceiling: %v", got)
	}
}

func TestForwardLimitsFromEnv(t *testing.T) {
	for _, name := range []string{"OPF_MAX_CONCURRENCY", "OPF_MAX_QUEUED_WINDOWS", "OPF_WINDOW_TOKENS", "OPF_WINDOW_OVERLAP", "OPF_WINDOW_TIMEOUT_SECONDS", "OPF_MAX_REQUEST_SECONDS", "OPF_FORWARD_TIMEOUT_SECONDS"} {
		t.Setenv(name, "")
	}
	limits, err := forwardLimitsFromEnv(4)
	want := forwardLimits{maxInflight: 4, maxQueued: defaultMaxQueuedWindows, windowTokens: 1024, windowOverlap: 128, windowTimeout: 5 * time.Second, maxRequest: 15 * time.Minute, forwardTimeout: 30 * time.Second}
	if err != nil || limits != want {
		t.Fatalf("defaults: %+v, %v", limits, err)
	}
	t.Setenv("OPF_MAX_CONCURRENCY", "2")
	t.Setenv("OPF_MAX_QUEUED_WINDOWS", "16")
	t.Setenv("OPF_WINDOW_TOKENS", "2048")
	t.Setenv("OPF_WINDOW_TIMEOUT_SECONDS", "3")
	t.Setenv("OPF_MAX_REQUEST_SECONDS", "600")
	limits, err = forwardLimitsFromEnv(4)
	if err != nil || limits.maxInflight != 8 || limits.maxQueued != 16 || limits.windowTokens != 2048 || limits.windowTimeout != 3*time.Second || limits.maxRequest != 10*time.Minute {
		t.Fatalf("configured: %+v, %v", limits, err)
	}
	t.Setenv("OPF_WINDOW_OVERLAP", "1024")
	if _, err := forwardLimitsFromEnv(4); err == nil {
		t.Fatal("accepted an overlap that leaves no stride")
	}
	t.Setenv("OPF_WINDOW_OVERLAP", "")
	for _, bad := range []string{"0", "-1", "two"} {
		t.Setenv("OPF_MAX_QUEUED_WINDOWS", bad)
		if _, err := forwardLimitsFromEnv(1); err == nil {
			t.Fatalf("accepted OPF_MAX_QUEUED_WINDOWS=%q", bad)
		}
	}
}
