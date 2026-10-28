package main

import "time"

const (
	listenAddress          = ":8001"
	inferenceHost          = "127.0.0.1"
	inferenceBasePort      = 8002
	maxWorkers             = 8
	controlplaneURL        = "https://api.tinfoil.sh"
	reporterID             = "pii-filter"
	redactPath             = "/redact"
	forwardPath            = "/forward"
	healthPath             = "/health"
	metricsPath            = "/metrics"
	billableRequestsHeader = "X-Tinfoil-Billable-Requests"
	// keyHashHeader carries a hash of the caller's credential from the /redact
	// proxy to the coordinating worker, which echoes it on every /forward it
	// fans out, so the forward scheduler can queue windows per caller.
	keyHashHeader = "X-Tinfoil-Key-Hash"
	// inputTokensHeader carries the request's input token count from the
	// coordinator back to the gateway, which meters it and strips it.
	inputTokensHeader = "X-Tinfoil-Input-Tokens"
	// forwardShapeHeader is the worker's "rows,classes" shape of a /forward
	// response body, relayed to the coordinator untouched.
	forwardShapeHeader      = "X-Tinfoil-Shape"
	maxRequestBytes         = 4 << 20
	defaultMaxConcurrency   = 1
	defaultMaxQueuedWindows = 4096
	headerTimeout           = 10 * time.Second
	// A /redact request may run for requestBaseTimeout plus
	// OPF_WINDOW_TIMEOUT_SECONDS per window, up to OPF_MAX_REQUEST_SECONDS;
	// the latter also bounds a queued window's wait and the server's
	// read/idle timeouts.
	requestBaseTimeout     = 30 * time.Second
	defaultWindowTimeout   = 5 * time.Second
	defaultForwardTimeout  = 30 * time.Second
	defaultMaxRequest      = 15 * time.Minute
	defaultWindowTokens    = 1024
	defaultWindowOverlap   = 128
	shutdownTimeout        = 30 * time.Second
	reportFlushTimeout     = 10 * time.Second
	healthProbeInterval    = 30 * time.Second
	healthProbeTimeout     = 5 * time.Second
	healthFailureThreshold = 3
)
