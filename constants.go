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
	inputTokensHeader       = "X-Tinfoil-Input-Tokens"
	maxRequestBytes         = 4 << 20
	defaultMaxConcurrency   = 1
	defaultMaxQueuedWindows = 512
	headerTimeout           = 10 * time.Second
	inferenceTimeout        = 2 * time.Minute
	shutdownTimeout         = 30 * time.Second
	reportFlushTimeout      = 10 * time.Second
	healthProbeInterval     = 30 * time.Second
	healthProbeTimeout      = 5 * time.Second
	healthFailureThreshold  = 3
)
