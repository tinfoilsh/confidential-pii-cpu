package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	usage "github.com/tinfoilsh/usage-reporting-go"
	usageclient "github.com/tinfoilsh/usage-reporting-go/client"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := run(ctx); err != nil {
		slog.Error("privacy filter stopped", "error", err)
		os.Exit(1)
	}
}

func reportingConfig() (usageclient.Config, error) {
	secret := strings.TrimSpace(os.Getenv("USAGE_REPORTER_SECRET"))
	if secret == "" {
		return usageclient.Config{}, errors.New("USAGE_REPORTER_SECRET is required")
	}
	base := os.Getenv("CONTROL_PLANE_URL")
	if base == "" {
		base = controlplaneURL
	}
	endpoint, err := url.Parse(base)
	if err != nil || endpoint.Scheme != "https" || endpoint.Host == "" || endpoint.User != nil || endpoint.RawQuery != "" || endpoint.Fragment != "" {
		return usageclient.Config{}, errors.New("CONTROL_PLANE_URL must be an HTTPS URL without credentials, query, or fragment")
	}
	return usageclient.Config{Endpoint: endpoint.JoinPath(usage.IngestionPath).String(), ReporterID: reporterID, Secret: secret}, nil
}

// workerCount reads OPF_WORKERS. Each worker is a separate inference process
// with its own interpreter and thread pool, so requests dispatched to
// different workers do not contend on a shared lock or torch pool.
func workerCount() (int, error) {
	raw := strings.TrimSpace(os.Getenv("OPF_WORKERS"))
	if raw == "" {
		return 1, nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 || n > maxWorkers {
		return 0, fmt.Errorf("OPF_WORKERS must be an integer between 1 and %d (got %q)", maxWorkers, raw)
	}
	return n, nil
}

// intEnv reads an integer environment variable that must be at least minimum.
func intEnv(name string, fallback, minimum int) (int, error) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback, nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < minimum {
		return 0, fmt.Errorf("%s must be an integer >= %d (got %q)", name, minimum, raw)
	}
	return n, nil
}

// forwardLimitsFromEnv sizes the forward scheduler and request deadlines:
// workers × the per-worker OPF_MAX_CONCURRENCY lanes may run at once,
// OPF_MAX_QUEUED_WINDOWS windows may wait for one, and a request is allowed
// OPF_WINDOW_TIMEOUT_SECONDS per window (estimated with OPF_WINDOW_TOKENS /
// OPF_WINDOW_OVERLAP, the geometry the workers use) up to
// OPF_MAX_REQUEST_SECONDS. Each dispatched forward must finish within
// OPF_FORWARD_TIMEOUT_SECONDS or its worker is considered wedged. Requests
// of at most OPF_SHORT_REQUEST_TOKENS are short (priority); the hold rule
// is OPF_SHORT_LANE_HORIZONS / OPF_SHORT_LANE_ARMED_SECONDS with the ETA
// model seeded by OPF_MS_PER_TOKEN.
func forwardLimitsFromEnv(workers int) (forwardLimits, error) {
	var limits forwardLimits
	var concurrency, windowTimeout, maxRequest, forwardTimeout int
	for _, v := range []struct {
		name              string
		fallback, minimum int
		into              *int
	}{
		{"OPF_MAX_CONCURRENCY", defaultMaxConcurrency, 1, &concurrency},
		{"OPF_MAX_QUEUED_WINDOWS", defaultMaxQueuedWindows, 1, &limits.maxQueued},
		{"OPF_WINDOW_TOKENS", defaultWindowTokens, 1, &limits.windowTokens},
		{"OPF_WINDOW_OVERLAP", defaultWindowOverlap, 0, &limits.windowOverlap},
		{"OPF_WINDOW_TIMEOUT_SECONDS", int(defaultWindowTimeout / time.Second), 1, &windowTimeout},
		{"OPF_MAX_REQUEST_SECONDS", int(defaultMaxRequest / time.Second), 1, &maxRequest},
		{"OPF_FORWARD_TIMEOUT_SECONDS", int(defaultForwardTimeout / time.Second), 1, &forwardTimeout},
	} {
		n, err := intEnv(v.name, v.fallback, v.minimum)
		if err != nil {
			return forwardLimits{}, err
		}
		*v.into = n
	}
	if 2*limits.windowOverlap >= limits.windowTokens {
		return forwardLimits{}, fmt.Errorf("OPF_WINDOW_TOKENS (%d) must exceed twice OPF_WINDOW_OVERLAP (%d)", limits.windowTokens, limits.windowOverlap)
	}
	shortTokens, err := intEnv("OPF_SHORT_REQUEST_TOKENS", limits.windowTokens, 1)
	if err != nil {
		return forwardLimits{}, err
	}
	limits.shortTokens = shortTokens
	limits.maxInflight = workers * concurrency
	limits.windowTimeout = time.Duration(windowTimeout) * time.Second
	limits.maxRequest = time.Duration(maxRequest) * time.Second
	limits.forwardTimeout = time.Duration(forwardTimeout) * time.Second
	if limits.horizons, err = secondsListEnv("OPF_SHORT_LANE_HORIZONS", defaultShortLaneHorizon); err != nil {
		return forwardLimits{}, err
	}
	armed, err := secondsEnv("OPF_SHORT_LANE_ARMED_SECONDS", defaultShortLaneArmed)
	if err != nil {
		return forwardLimits{}, err
	}
	limits.armed = armed
	if limits.msPerToken, err = floatEnv("OPF_MS_PER_TOKEN", defaultMsPerToken); err != nil {
		return forwardLimits{}, err
	}
	return limits, nil
}

// floatEnv reads a positive floating-point environment variable.
func floatEnv(name string, fallback float64) (float64, error) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback, nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("%s must be a positive number (got %q)", name, raw)
	}
	return v, nil
}

// secondsEnv reads a non-negative duration in seconds.
func secondsEnv(name string, fallback time.Duration) (time.Duration, error) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback, nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || v < 0 {
		return 0, fmt.Errorf("%s must be a non-negative number of seconds (got %q)", name, raw)
	}
	return time.Duration(v * float64(time.Second)), nil
}

// secondsListEnv reads a comma-separated list of seconds. Empty or "0"
// yields no entries (the hold rule off); the list must be non-decreasing
// since entry k applies to the k-th soonest-freeing lane.
func secondsListEnv(name, fallback string) ([]time.Duration, error) {
	raw, set := os.LookupEnv(name)
	if !set {
		raw = fallback
	}
	raw = strings.TrimSpace(raw)
	if raw == "" || raw == "0" {
		return nil, nil
	}
	var out []time.Duration
	for _, field := range strings.Split(raw, ",") {
		v, err := strconv.ParseFloat(strings.TrimSpace(field), 64)
		if err != nil || v < 0 {
			return nil, fmt.Errorf("%s must be comma-separated seconds (got %q)", name, raw)
		}
		d := time.Duration(v * float64(time.Second))
		if len(out) > 0 && d < out[len(out)-1] {
			return nil, fmt.Errorf("%s must be non-decreasing (got %q)", name, raw)
		}
		out = append(out, d)
	}
	return out, nil
}

func run(ctx context.Context) error {
	cfg, err := reportingConfig()
	if err != nil {
		return err
	}
	workers, err := workerCount()
	if err != nil {
		return err
	}
	limits, err := forwardLimitsFromEnv(workers)
	if err != nil {
		return err
	}
	reporter := usageclient.New(cfg)
	defer func() {
		flushCtx, cancel := context.WithTimeout(context.Background(), reportFlushTimeout)
		defer cancel()
		reporter.Stop(flushCtx)
	}()

	backendCtx, stopBackend := context.WithCancel(context.Background())
	defer stopBackend()

	backendDone := make(chan error, workers)
	unhealthy := make(chan error, workers)
	targets := make([]*url.URL, workers)
	backends := make([]*exec.Cmd, workers)
	for i := range workers {
		port := strconv.Itoa(inferenceBasePort + i)
		backendURL := "http://" + inferenceHost + ":" + port
		backend := exec.CommandContext(backendCtx, "uvicorn", "server:app", "--host", inferenceHost, "--port", port, "--no-access-log", "--timeout-keep-alive", strconv.Itoa(int(workerKeepAlive.Seconds())))
		backend.Stdout, backend.Stderr = os.Stdout, os.Stderr
		backend.Cancel = func() error { return backend.Process.Signal(syscall.SIGTERM) }
		backend.WaitDelay = shutdownTimeout
		if err := backend.Start(); err != nil {
			stopBackend()
			for _, started := range backends[:i] {
				_ = started.Wait()
			}
			return fmt.Errorf("start inference worker %d: %w", i, err)
		}
		backends[i] = backend
		go func(worker int, cmd *exec.Cmd) {
			err := cmd.Wait()
			backendDone <- fmt.Errorf("inference worker %d exited unexpectedly: %v", worker, err)
		}(i, backend)
		targets[i], _ = url.Parse(backendURL)
		// Exiting on a wedged worker lets Docker's restart policy replace the
		// whole set; an unhealthy state alone never triggers a restart.
		go func(worker int, failed <-chan error) {
			if err, ok := <-failed; ok {
				unhealthy <- fmt.Errorf("inference worker %d unhealthy: %w", worker, err)
			}
		}(i, watchHealth(backendCtx, backendURL+healthPath, healthProbeInterval))
	}

	server := &http.Server{Addr: listenAddress, Handler: newGateway(targets, reporter, limits), ReadHeaderTimeout: headerTimeout, ReadTimeout: limits.maxRequest, IdleTimeout: limits.maxRequest}
	serverDone := make(chan error, 1)
	go func() { serverDone <- server.ListenAndServe() }()

	var result error
	backendsExited := 0
	select {
	case <-ctx.Done():
	case err := <-backendDone:
		backendsExited = 1
		result = err
	case err := <-unhealthy:
		result = err
	case err := <-serverDone:
		if !errors.Is(err, http.ErrServerClosed) {
			result = err
		}
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		_ = server.Close()
		result = errors.Join(result, err)
	}
	stopBackend()
	for ; backendsExited < workers; backendsExited++ {
		<-backendDone
	}
	return result
}
