package main

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// testLimits has the hold rule off; tests that exercise it set horizons.
var testLimits = forwardLimits{maxInflight: 4, maxQueued: 512, windowTokens: 1024, windowOverlap: 128, windowTimeout: 5 * time.Second, maxRequest: 15 * time.Minute, forwardTimeout: 30 * time.Second, shortTokens: 1024, msPerToken: 0.8}

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
	return forwardWindow(t, gateway, key, 1, 1_000_000) // a window of a long request
}

// serveQueue serves a gateway's internal /queue listener for a test, the way
// main binds it to loopback, and returns its base URL.
func serveQueue(t *testing.T, g *gateway) string {
	t.Helper()
	srv := httptest.NewServer(g.internal)
	t.Cleanup(srv.Close)
	return srv.URL
}

// forwardWindow posts one window of `tokens` int32s belonging to a request
// of requestTokens tokens in total.
func forwardWindow(t *testing.T, gateway, key string, tokens, requestTokens int) (int, time.Duration) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, gateway+queuePath, bytes.NewReader(make([]byte, 4*tokens)))
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set(keyHashHeader, key)
	req.Header.Set(requestTokensHeader, strconv.Itoa(requestTokens))
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
	queue := serveQueue(t, newGateway(workers, &recordedUsage{}, limitsWith(2, 512)))

	var wg sync.WaitGroup
	for range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if status, _ := forwardRequest(t, queue, "big-caller"); status != http.StatusOK {
				t.Errorf("big caller window: %d", status)
			}
		}()
	}
	time.Sleep(windowTime / 2) // the big caller's windows are queued and both lanes busy
	status, elapsed := forwardRequest(t, queue, "small-caller")
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
	queue := serveQueue(t, newGateway([]*url.URL{fakes[0].url(), fakes[1].url()}, &recordedUsage{}, limitsWith(2, 512)))

	var wg sync.WaitGroup
	for i := range 12 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if status, _ := forwardRequest(t, queue, "caller"); status != http.StatusOK {
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
	queue := serveQueue(t, newGateway([]*url.URL{fw.url()}, &recordedUsage{}, limitsWith(1, 2)))

	var wg sync.WaitGroup
	var ok, shed atomic.Int64
	for range 6 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			switch status, _ := forwardRequest(t, queue, "caller"); status {
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
	if status, _ := forwardRequest(t, queue, "caller"); status != http.StatusOK {
		t.Fatalf("queue did not drain: %d", status)
	}
}

func TestForwardRelaysBinaryPayloadShapeAndPosition(t *testing.T) {
	var gotBody []byte
	var gotType, gotOffset string
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		gotType = r.Header.Get("Content-Type")
		gotOffset = r.Header.Get(positionOffsetHeader)
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set(forwardShapeHeader, "2,3")
		_, _ = w.Write([]byte{1, 2, 3, 4, 5, 6})
	}))
	defer worker.Close()
	target, _ := url.Parse(worker.URL)
	queue := serveQueue(t, newGateway([]*url.URL{target}, &recordedUsage{}, testLimits))
	req, _ := http.NewRequest(http.MethodPost, queue+queuePath, bytes.NewReader([]byte{9, 0, 0, 0}))
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set(positionOffsetHeader, "4096") // the window's absolute position: exactness depends on it
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if !bytes.Equal(gotBody, []byte{9, 0, 0, 0}) || gotType != "application/octet-stream" || gotOffset != "4096" {
		t.Fatalf("worker got %v %q offset %q", gotBody, gotType, gotOffset)
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
	queue := serveQueue(t, handler)

	health := func() int {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, healthPath, nil))
		return rec.Code
	}
	if health() != http.StatusOK {
		t.Fatal("workers unhealthy before any forward")
	}
	status, elapsed := forwardRequest(t, queue, "caller")
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

// A short request's window goes ahead of every queued long window.
func TestShortRequestsAreServedFirst(t *testing.T) {
	const windowTime = 40 * time.Millisecond
	fw := newFakeWorker(t, windowTime)
	queue := serveQueue(t, newGateway([]*url.URL{fw.url()}, &recordedUsage{}, limitsWith(1, 512)))
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			forwardWindow(t, queue, "long", 1, 1_000_000)
		}()
	}
	time.Sleep(windowTime / 2) // one running, 19 queued
	status, elapsed := forwardWindow(t, queue, "short", 1, 1)
	wg.Wait()
	if status != http.StatusOK || elapsed > 2*windowTime {
		t.Fatalf("short window: status %d after %v behind 19 long windows", status, elapsed)
	}
}

// holdScenario runs a stream of long windows over 4 lanes and measures how
// long a short window arriving at the given offsets waits.
func holdScenario(t *testing.T, limits forwardLimits, arrivals []time.Duration, longWindows int) (waits []time.Duration, total time.Duration, fakes []*fakeWorker) {
	t.Helper()
	const windowTime = 200 * time.Millisecond
	const tokens = 100 // × msPerToken 2 = 200 ms predicted, matching the fake
	limits.msPerToken = 2
	var targets []*url.URL
	for range 4 {
		fw := newFakeWorker(t, windowTime)
		fakes = append(fakes, fw)
		targets = append(targets, fw.url())
	}
	queue := serveQueue(t, newGateway(targets, &recordedUsage{}, limits))
	start := time.Now()
	var wg sync.WaitGroup
	for range longWindows {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if status, _ := forwardWindow(t, queue, "long", tokens, 1_000_000); status != http.StatusOK {
				t.Errorf("long window: %d", status)
			}
		}()
	}
	waits = make([]time.Duration, len(arrivals))
	for i, at := range arrivals {
		wg.Add(1)
		go func(i int, at time.Duration) {
			defer wg.Done()
			time.Sleep(at)
			status, elapsed := forwardWindow(t, queue, "short", tokens, tokens)
			if status != http.StatusOK {
				t.Errorf("short window: %d", status)
			}
			waits[i] = elapsed - windowTime
		}(i, at)
	}
	wg.Wait()
	return waits, time.Since(start), fakes
}

// With the hold rule, a short window arriving at a random moment during a
// stream of long windows waits at most about a horizon; without it, it can
// wait a whole window. The lab measures the capacity cost via
// pii_forward_hold_seconds_total.
func TestHoldRuleBoundsShortRequestWait(t *testing.T) {
	const horizon = 60 * time.Millisecond
	const tick = 60 * time.Millisecond // scheduling + HTTP slack under -race
	// Arrivals spread across one 200 ms window cycle (the lanes start in
	// lockstep at ~0 ms), so without the rule at least one of them lands
	// just after a cycle boundary and waits most of a window.
	arrivals := []time.Duration{230 * time.Millisecond, 300 * time.Millisecond, 370 * time.Millisecond}
	limits := limitsWith(4, 512)
	limits.horizons = []time.Duration{horizon}
	waits, _, _ := holdScenario(t, limits, arrivals, 40)
	for i, wait := range waits {
		if wait > horizon+tick {
			t.Fatalf("short window %d (at %v) waited %v with hold rule (horizon %v)", i, arrivals[i], wait, horizon)
		}
	}
	limits.horizons = nil
	waits, _, _ = holdScenario(t, limits, arrivals, 40)
	var worst time.Duration
	for _, wait := range waits {
		worst = max(worst, wait)
	}
	if worst < horizon+tick {
		t.Fatalf("without the hold rule every short window still waited under %v (worst %v); the rule's test is not discriminating", horizon+tick, worst)
	}
}

// With no short traffic the rule must not leave a lane permanently idle:
// every worker takes windows and throughput stays near the lane count.
func TestHoldRuleKeepsAllLanesBusyWithoutShortTraffic(t *testing.T) {
	limits := limitsWith(4, 512)
	limits.horizons = []time.Duration{60 * time.Millisecond}
	_, total, fakes := holdScenario(t, limits, nil, 24)
	for i, fw := range fakes {
		if fw.calls.Load() == 0 {
			t.Fatalf("worker %d never ran a window", i)
		}
	}
	// 24 windows × 200 ms over 4 lanes is 1.2 s flat out; the rule's cost is
	// bounded by one lane idling (window − horizon) per cycle.
	if total > 2200*time.Millisecond {
		t.Fatalf("24 windows took %v; a lane is idling", total)
	}
}

// Armed mode: with no short request seen recently the rule is off and the
// /queue is not on the shim-facing handler at all; it lives on the
// gateway's internal listener (loopback-only in main).
func TestQueueIsNotOnThePublicHandler(t *testing.T) {
	fw := newFakeWorker(t, 0)
	g := newGateway([]*url.URL{fw.url()}, &recordedUsage{}, testLimits)
	req := httptest.NewRequest(http.MethodPost, queuePath, bytes.NewReader([]byte{1, 0, 0, 0}))
	req.Header.Set(keyHashHeader, "k")
	rec := httptest.NewRecorder()
	g.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound || fw.calls.Load() != 0 {
		t.Fatalf("public /queue: status %d, %d worker calls", rec.Code, fw.calls.Load())
	}
	rec = httptest.NewRecorder()
	g.internal.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, healthPath, nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("internal listener serves the API: %d", rec.Code)
	}
	if status, _ := forwardWindow(t, serveQueue(t, g), "k", 1, 1); status != http.StatusOK {
		t.Fatalf("internal /queue: %d", status)
	}
}

// The /redact proxy stamps the caller's key hash on the outbound request
// (never the credential) and strips every internal X-Tinfoil-* header the
// client supplies.
func TestRedactCarriesKeyHashToCoordinatorAndStripsInternalHeaders(t *testing.T) {
	var seen atomic.Value
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen.Store(r.Header.Clone())
		w.Header().Set(inputTokensHeader, "1")
		_, _ = io.WriteString(w, `{}`)
	}))
	defer backend.Close()
	target, _ := url.Parse(backend.URL)
	handler := newGateway([]*url.URL{target}, &recordedUsage{}, testLimits)
	req := httptest.NewRequest(http.MethodPost, redactPath, strings.NewReader(`{"text":"hi"}`))
	req.Header.Set("Authorization", "Bearer tk_customer")
	internal := []string{keyHashHeader, requestIDHeader, inputTokensHeader, requestTokensHeader, positionOffsetHeader, forwardShapeHeader}
	for _, h := range internal {
		req.Header.Set(h, "spoofed")
	}
	handler.ServeHTTP(httptest.NewRecorder(), req)
	got, _ := seen.Load().(http.Header)
	if hash := got.Get(keyHashHeader); hash != keyHash("tk_customer") || strings.Contains(hash, "tk_customer") {
		t.Fatalf("coordinator saw key hash %q", hash)
	}
	if id := got.Get(requestIDHeader); id == "" || id == "spoofed" {
		t.Fatalf("request id %q", id)
	}
	for _, h := range internal[2:] {
		if v := got.Get(h); v != "" {
			t.Fatalf("client-supplied %s reached the coordinator: %q", h, v)
		}
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
	for _, name := range []string{"OPF_MAX_CONCURRENCY", "OPF_MAX_QUEUED_WINDOWS", "OPF_WINDOW_TOKENS", "OPF_WINDOW_OVERLAP", "OPF_WINDOW_TIMEOUT_SECONDS", "OPF_MAX_REQUEST_SECONDS", "OPF_FORWARD_TIMEOUT_SECONDS", "OPF_SHORT_REQUEST_TOKENS", "OPF_SHORT_LANE_ARMED_SECONDS", "OPF_LONG_GRACE_SECONDS", "OPF_MS_PER_TOKEN"} {
		t.Setenv(name, "")
	}
	t.Setenv("OPF_SHORT_LANE_HORIZONS", "0.5,1.5") // the default, restored after the test (os.Unsetenv would not be)
	limits, err := forwardLimitsFromEnv(4)
	want := forwardLimits{maxInflight: 4, maxQueued: defaultMaxQueuedWindows, windowTokens: 6144, windowOverlap: 1024, windowTimeout: 5 * time.Second, maxRequest: 15 * time.Minute, forwardTimeout: 30 * time.Second, shortTokens: 6144, horizons: []time.Duration{500 * time.Millisecond, 1500 * time.Millisecond}, armed: 30 * time.Second, longGrace: 5 * time.Second, msPerToken: 0.5}
	if err != nil || fmt.Sprint(limits) != fmt.Sprint(want) {
		t.Fatalf("defaults: %+v, %v", limits, err)
	}
	t.Setenv("OPF_SHORT_LANE_HORIZONS", "0")
	t.Setenv("OPF_SHORT_LANE_ARMED_SECONDS", "600")
	t.Setenv("OPF_SHORT_REQUEST_TOKENS", "2048")
	t.Setenv("OPF_LONG_GRACE_SECONDS", "0")
	t.Setenv("OPF_MS_PER_TOKEN", "1.25")
	limits, err = forwardLimitsFromEnv(4)
	if err != nil || len(limits.horizons) != 0 || limits.armed != 10*time.Minute || limits.shortTokens != 2048 || limits.longGrace != 0 || limits.msPerToken != 1.25 {
		t.Fatalf("hold rule config: %+v, %v", limits, err)
	}
	t.Setenv("OPF_SHORT_LANE_HORIZONS", "3,1.5")
	if _, err := forwardLimitsFromEnv(4); err == nil {
		t.Fatal("accepted decreasing horizons")
	}
	t.Setenv("OPF_SHORT_LANE_HORIZONS", "")
	// Retired knobs are refused rather than ignored, so an old config
	// cannot silently reintroduce them.
	for _, retired := range []string{"OPF_MAX_CONCURRENCY", "OPF_FORWARD_FANOUT"} {
		t.Setenv(retired, "2")
		if _, err := forwardLimitsFromEnv(4); err == nil {
			t.Fatalf("accepted retired setting %s", retired)
		}
		t.Setenv(retired, "")
	}
	t.Setenv("OPF_MAX_QUEUED_WINDOWS", "16")
	t.Setenv("OPF_WINDOW_TOKENS", "4096")
	t.Setenv("OPF_WINDOW_TIMEOUT_SECONDS", "3")
	t.Setenv("OPF_MAX_REQUEST_SECONDS", "600")
	limits, err = forwardLimitsFromEnv(4)
	if err != nil || limits.maxInflight != 4 || limits.maxQueued != 16 || limits.windowTokens != 4096 || limits.windowTimeout != 3*time.Second || limits.maxRequest != 10*time.Minute {
		t.Fatalf("configured: %+v, %v", limits, err)
	}
	t.Setenv("OPF_WINDOW_OVERLAP", "2048")
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

func TestQueueURLForWorkersDefaultsToGateway(t *testing.T) {
	t.Setenv("OPF_QUEUE_URL", "")
	if got := queueURLForWorkers(); got != "http://127.0.0.1:8000/queue" {
		t.Fatalf("default forward url: %q", got)
	}
	t.Setenv("OPF_QUEUE_URL", "http://127.0.0.1:9999/queue")
	if got := queueURLForWorkers(); got != "http://127.0.0.1:9999/queue" {
		t.Fatalf("override forward url: %q", got)
	}
}

// A pure scheduler test: cancelling a request drops only its queued windows
// (both classes), wakes their waiters with a nil lane, leaves windows already
// on a lane and other requests' windows alone, and refills the freed slots.
func TestSchedulerCancelRequestDropsOnlyThatRequestsQueue(t *testing.T) {
	workers := []*worker{{target: &url.URL{Host: "w0"}}, {target: &url.URL{Host: "w1"}}}
	s := newForwardScheduler(workers, limitsWith(2, 64))
	// Two lanes: the first two windows of request A run at once.
	a1, _ := s.enqueue("key", "A", 100, false)
	a2, _ := s.enqueue("key", "A", 100, false)
	a3, _ := s.enqueue("key", "A", 100, false)   // queued (long)
	aShort, _ := s.enqueue("key", "A", 10, true) // queued (short class)
	b1, _ := s.enqueue("key", "B", 100, false)   // same caller, other request
	c1, _ := s.enqueue("other", "C", 100, false) // other caller
	for _, running := range []*forwardItem{a1, a2} {
		if <-running.ready == nil {
			t.Fatal("running window lost its lane")
		}
	}
	if s.queued != 4 || s.inflight != 2 {
		t.Fatalf("queued %d inflight %d", s.queued, s.inflight)
	}
	if dropped := s.cancelRequest("A"); dropped != 2 {
		t.Fatalf("dropped %d windows of A, want 2", dropped)
	}
	for _, cancelled := range []*forwardItem{a3, aShort} {
		select {
		case lane := <-cancelled.ready:
			if lane != nil {
				t.Fatal("cancelled window was given a lane")
			}
		default:
			t.Fatal("cancelled window was not woken")
		}
	}
	if s.queued != 2 || s.inflight != 2 {
		t.Fatalf("after cancel: queued %d inflight %d", s.queued, s.inflight)
	}
	// Releasing A's running lanes dispatches B then C, in caller round-robin.
	s.release(s.lanes[0], true)
	s.release(s.lanes[1], true)
	for _, next := range []*forwardItem{b1, c1} {
		select {
		case lane := <-next.ready:
			if lane == nil {
				t.Fatal("surviving window was cancelled")
			}
		default:
			t.Fatal("surviving window not dispatched after lanes freed")
		}
	}
	if s.cancelRequest("A") != 0 || s.cancelRequest("") != 0 {
		t.Fatal("cancelling an absent request dropped something")
	}
}

// Within the short class callers are served round-robin too: one caller
// streaming many short requests cannot starve another caller's short one.
func TestSchedulerShortClassIsFairAcrossCallers(t *testing.T) {
	workers := []*worker{{target: &url.URL{Host: "w0"}}}
	s := newForwardScheduler(workers, limitsWith(1, 64))
	first, _ := s.enqueue("chatty", "r0", 10, true) // takes the only lane
	<-first.ready
	var chatty []*forwardItem
	for i := range 5 {
		item, _ := s.enqueue("chatty", "r"+strconv.Itoa(i+1), 10, true)
		chatty = append(chatty, item)
	}
	quiet, _ := s.enqueue("quiet", "q", 10, true)
	s.release(s.lanes[0], true) // next dispatch: chatty[0] (its turn), then quiet
	<-chatty[0].ready
	s.release(s.lanes[0], true)
	select {
	case <-quiet.ready:
	default:
		t.Fatal("the quiet caller's short window waited behind the chatty caller's backlog")
	}
}

// /health answers from state, not by probing: a worker that no longer
// answers anything (wedged) or has not booted yet makes it 503 at once.
func TestHealthAnswersFromStateWithoutProbing(t *testing.T) {
	hung := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done() // never answers
	}))
	defer hung.Close()
	target, _ := url.Parse(hung.URL)
	g := buildGateway([]*url.URL{target}, &recordedUsage{}, limitsWith(1, 512), "")
	health := func() (int, string, time.Duration) {
		rec := httptest.NewRecorder()
		start := time.Now()
		g.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, healthPath, nil))
		return rec.Code, strings.TrimSpace(rec.Body.String()), time.Since(start)
	}
	if code, _, took := health(); code != http.StatusOK || took > 100*time.Millisecond {
		t.Fatalf("ready worker: %d in %v (must not probe)", code, took)
	}
	g.workers[0].wedged.Store(true)
	if code, body, took := health(); code != http.StatusServiceUnavailable || body != "worker 0 being replaced" || took > 100*time.Millisecond {
		t.Fatalf("wedged worker: %d %q in %v", code, body, took)
	}
	g.workers[0].wedged.Store(false)
	g.workers[0].ready.Store(false)
	if code, body, _ := health(); code != http.StatusServiceUnavailable || body != "worker 0 starting" {
		t.Fatalf("booting worker: %d %q", code, body)
	}
}

// The floor under the long class: short windows go first, but a long window
// is dispatched at least every longGrace even under a stream of short calls
// that would otherwise keep every lane forever.
func TestLongWindowsGetALaneEveryGrace(t *testing.T) {
	workers := []*worker{{target: &url.URL{Host: "w0"}}}
	limits := limitsWith(1, 64)
	limits.longGrace = time.Second
	s := newForwardScheduler(workers, limits)
	clock := time.Unix(1_700_000_000, 0)
	s.now = func() time.Time { return clock }
	s.lastLong = clock

	first, _ := s.enqueue("chat", "c0", 10, true) // takes the only lane
	<-first.ready
	doc, _ := s.enqueue("doc", "d", 6144, false)
	dispatched := func(item *forwardItem) bool {
		select {
		case <-item.ready:
			return true
		default:
			return false
		}
	}
	// A short stream: each release dispatches the next short, never the doc,
	// while the grace has not run out.
	for i := range 5 {
		clock = clock.Add(150 * time.Millisecond)
		next, _ := s.enqueue("chat", "c"+strconv.Itoa(i+1), 10, true)
		s.release(s.lanes[0], true)
		if !dispatched(next) || dispatched(doc) {
			t.Fatalf("release %d: short %v doc %v", i, dispatched(next), dispatched(doc))
		}
	}
	// Past the grace, the next lane goes to the doc even though a short waits.
	clock = clock.Add(400 * time.Millisecond) // 1.15 s since lastLong
	pending, _ := s.enqueue("chat", "c9", 10, true)
	s.release(s.lanes[0], true)
	if !dispatched(doc) || dispatched(pending) {
		t.Fatal("the long window did not get its lane after the grace")
	}
	// The floor's clock restarted: the short goes next.
	s.release(s.lanes[0], true)
	if !dispatched(pending) {
		t.Fatal("short not dispatched after the floor fired")
	}
}

// With the floor off, short windows win every time (the pre-floor rule).
func TestLongWindowsStarveWithoutGrace(t *testing.T) {
	workers := []*worker{{target: &url.URL{Host: "w0"}}}
	limits := limitsWith(1, 64)
	limits.longGrace = 0
	s := newForwardScheduler(workers, limits)
	clock := time.Unix(1_700_000_000, 0)
	s.now = func() time.Time { return clock }
	first, _ := s.enqueue("chat", "c0", 10, true)
	<-first.ready
	doc, _ := s.enqueue("doc", "d", 6144, false)
	for i := range 5 {
		clock = clock.Add(10 * time.Second)
		next, _ := s.enqueue("chat", "c"+strconv.Itoa(i+1), 10, true)
		s.release(s.lanes[0], true)
		select {
		case <-doc.ready:
			t.Fatal("long window dispatched with the floor off")
		case <-next.ready:
		}
	}
}
