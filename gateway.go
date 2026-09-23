package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"

	"github.com/google/uuid"
	usage "github.com/tinfoilsh/usage-reporting-go"
	usageclient "github.com/tinfoilsh/usage-reporting-go/client"
)

type eventReporter interface {
	AddEvent(usage.Event)
	Stats() usageclient.Stats
}

// The shim authenticates the bearer credential before this handler is reached.
// The inference server is private; only this handler emits billing events.
func newGateway(target *url.URL, reporter eventReporter) http.Handler {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = inferenceTimeout
	transport.DisableCompression = true
	proxy := &httputil.ReverseProxy{
		Transport: transport,
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(target)
			r.Out.Header.Del("Authorization")
			r.Out.Header.Del(usage.HeaderContext)
			r.Out.Header.Del(usage.HeaderUsageContextSignature)
			r.Out.Header.Del(billableRequestsHeader)
			r.Out.Header.Del("Accept-Encoding")
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			slog.Error("privacy filter upstream unavailable")
			http.Error(w, "privacy filter unavailable", http.StatusBadGateway)
		},
	}
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
		ctx, cancel := context.WithTimeout(r.Context(), inferenceTimeout)
		defer cancel()
		requestProxy := *proxy
		requestProxy.ModifyResponse = func(resp *http.Response) error {
			resp.Header.Del(billableRequestsHeader)
			if resp.StatusCode == http.StatusOK {
				reporter.AddEvent(usage.Event{
					RequestID:        uuid.NewString(),
					APIKey:           parts[1],
					Operation:        usage.Operation{Service: usage.ServicePIIFilter, Name: usage.OperationPIIFilterRedact},
					CustomerRequests: 1,
				})
				resp.Header.Set(billableRequestsHeader, "1")
			}
			return nil
		}
		requestProxy.ServeHTTP(w, r.WithContext(ctx))
	})
	mux.Handle("GET "+healthPath, proxy)
	mux.HandleFunc("GET "+metricsPath, func(w http.ResponseWriter, r *http.Request) {
		// /metrics is admin-authenticated by the shim, just like Python's metrics.
		metricsProxy := *proxy
		metricsProxy.ModifyResponse = func(resp *http.Response) error {
			if resp.StatusCode != http.StatusOK {
				return nil
			}
			stats := reporter.Stats()
			suffix := fmt.Sprintf("\npii_usage_enqueued_total %d\npii_usage_delivered_total %d\npii_usage_failed_total %d\npii_usage_dropped_total %d\n",
				stats.Enqueued, stats.DeliveredEvents, stats.FailedEvents, stats.DroppedDisabled+stats.DroppedBufferFull)
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
