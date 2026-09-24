package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestWatchHealthExitsOnlyAfterReadyThenRepeatedFailure(t *testing.T) {
	const interval = 10 * time.Millisecond
	var status atomic.Int32
	status.Store(http.StatusServiceUnavailable)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(int(status.Load()))
	}))
	defer backend.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	unhealthy := watchHealth(ctx, backend.URL+healthPath, interval)

	// Failing before the model has ever loaded is startup, not a wedge.
	select {
	case err := <-unhealthy:
		t.Fatalf("reported unhealthy during startup: %v", err)
	case <-time.After(20 * interval):
	}

	status.Store(http.StatusOK)
	time.Sleep(5 * interval)

	// A single failure recovers without exiting.
	status.Store(http.StatusServiceUnavailable)
	time.Sleep(interval + interval/2)
	status.Store(http.StatusOK)
	select {
	case err := <-unhealthy:
		t.Fatalf("reported unhealthy after one failed probe: %v", err)
	case <-time.After(5 * interval):
	}

	status.Store(http.StatusServiceUnavailable)
	select {
	case err := <-unhealthy:
		if err == nil {
			t.Fatal("unhealthy signal carried no error")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("sustained failures did not trigger exit")
	}
}
