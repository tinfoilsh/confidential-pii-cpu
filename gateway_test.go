package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
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
				if r.URL.Path == redactPath {
					body, _ := io.ReadAll(r.Body)
					if string(body) != query {
						t.Errorf("body changed: %s", body)
					}
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.upstreamStatus)
				_, _ = io.WriteString(w, responseBody)
			}))
			defer backend.Close()
			target, _ := url.Parse(backend.URL)
			handler := newGateway(target, reporter)
			request := httptest.NewRequest(tc.method, tc.path, strings.NewReader(query))
			if tc.auth != "" {
				request.Header.Set("Authorization", tc.auth)
			}
			request.Header.Set(billableRequestsHeader, "0")
			request.Header.Set(usage.HeaderContext, "attacker-requests-free-inference")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != tc.wantStatus {
				t.Fatalf("status %d, want %d", response.Code, tc.wantStatus)
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
			if event.APIKey != strings.TrimPrefix(tc.auth, "Bearer ") || event.CustomerRequests != 1 || event.Operation.Service != usage.ServicePIIFilter || event.Operation.Name != usage.OperationPIIFilterRedact {
				t.Fatalf("incorrect attribution: %+v", event)
			}
			if event.RequestID == "" || event.EventID == "" || response.Header().Get(billableRequestsHeader) != "1" {
				t.Fatal("missing event identity or receipt")
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
		_, _ = io.WriteString(w, `{}`)
	}))
	defer backend.Close()
	target, _ := url.Parse(backend.URL)
	reporter := &recordedUsage{}
	handler := newGateway(target, reporter)
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
