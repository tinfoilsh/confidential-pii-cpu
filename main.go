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

func run(ctx context.Context) error {
	cfg, err := reportingConfig()
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
	backend := exec.CommandContext(backendCtx, "uvicorn", "server:app", "--host", inferenceHost, "--port", inferencePort, "--no-access-log")
	backend.Stdout, backend.Stderr = os.Stdout, os.Stderr
	backend.Cancel = func() error { return backend.Process.Signal(syscall.SIGTERM) }
	backend.WaitDelay = shutdownTimeout
	if err := backend.Start(); err != nil {
		return fmt.Errorf("start inference: %w", err)
	}
	backendDone := make(chan error, 1)
	go func() { backendDone <- backend.Wait() }()
	target, _ := url.Parse(inferenceURL)
	unhealthy := watchHealth(backendCtx, inferenceURL+healthPath, healthProbeInterval)
	server := &http.Server{Addr: listenAddress, Handler: newGateway(target, reporter), ReadHeaderTimeout: headerTimeout, ReadTimeout: inferenceTimeout, IdleTimeout: inferenceTimeout}
	serverDone := make(chan error, 1)
	go func() { serverDone <- server.ListenAndServe() }()
	var result error
	backendExited := false
	select {
	case <-ctx.Done():
	case err := <-backendDone:
		backendExited = true
		result = fmt.Errorf("inference exited unexpectedly: %v", err)
	case err := <-unhealthy:
		// Exiting lets Docker's restart policy replace the wedged process;
		// an unhealthy state alone never triggers a restart.
		result = fmt.Errorf("inference unhealthy: %w", err)
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
	if !backendExited {
		<-backendDone
	}
	return result
}
