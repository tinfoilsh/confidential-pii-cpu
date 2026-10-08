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

func run(ctx context.Context) error {
	cfg, err := reportingConfig()
	if err != nil {
		return err
	}
	workers, err := workerCount()
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
		backend := exec.CommandContext(backendCtx, "uvicorn", "server:app", "--host", inferenceHost, "--port", port, "--no-access-log")
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

	server := &http.Server{Addr: listenAddress, Handler: newGateway(targets, reporter), ReadHeaderTimeout: headerTimeout, ReadTimeout: inferenceTimeout, IdleTimeout: inferenceTimeout}
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
