package main

import (
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

var testLimits = forwardLimits{maxInflight: 4, maxQueued: 512}

// fakeWorker serves /forward with a fixed latency and counts concurrency.
type fakeWorker struct {
	server  *httptest.Server
	latency time.Duration
	busy    atomic.Int64
	maxBusy atomic.Int64
	calls   atomic.Int64
}

func newFakeWorker(t *testing.T, latency time.Duration) *fakeWorker {
	t.Helper()
	fw := &fakeWorker{latency: latency}
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
		time.Sleep(fw.latency)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"logprobs":[[0.0]]}`)
	}))
	t.Cleanup(fw.server.Close)
	return fw
}

func (fw *fakeWorker) url() *url.URL {
	u, _ := url.Parse(fw.server.URL)
	return u
}

func forwardRequest(t *testing.T, gateway, key string) (int, time.Duration) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, gateway+forwardPath, strings.NewReader(`{"tokens":[1,2,3]}`))
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
	gateway := httptest.NewServer(newGateway(workers, &recordedUsage{}, forwardLimits{maxInflight: 2, maxQueued: 512}))
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
	gateway := httptest.NewServer(newGateway([]*url.URL{fakes[0].url(), fakes[1].url()}, &recordedUsage{}, forwardLimits{maxInflight: 2, maxQueued: 512}))
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
	gateway := httptest.NewServer(newGateway([]*url.URL{fw.url()}, &recordedUsage{}, forwardLimits{maxInflight: 1, maxQueued: 2}))
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
		req := httptest.NewRequest(http.MethodPost, forwardPath, strings.NewReader(`{"tokens":[1]}`))
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

func TestForwardLimitsFromEnv(t *testing.T) {
	t.Setenv("OPF_MAX_CONCURRENCY", "")
	t.Setenv("OPF_MAX_QUEUED_WINDOWS", "")
	limits, err := forwardLimitsFromEnv(4)
	if err != nil || limits.maxInflight != 4 || limits.maxQueued != defaultMaxQueuedWindows {
		t.Fatalf("defaults: %+v, %v", limits, err)
	}
	t.Setenv("OPF_MAX_CONCURRENCY", "2")
	t.Setenv("OPF_MAX_QUEUED_WINDOWS", "16")
	limits, err = forwardLimitsFromEnv(4)
	if err != nil || limits.maxInflight != 8 || limits.maxQueued != 16 {
		t.Fatalf("configured: %+v, %v", limits, err)
	}
	for _, bad := range []string{"0", "-1", "two"} {
		t.Setenv("OPF_MAX_QUEUED_WINDOWS", bad)
		if _, err := forwardLimitsFromEnv(1); err == nil {
			t.Fatalf("accepted OPF_MAX_QUEUED_WINDOWS=%q", bad)
		}
	}
}
