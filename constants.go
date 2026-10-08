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
	healthPath             = "/health"
	metricsPath            = "/metrics"
	billableRequestsHeader = "X-Tinfoil-Billable-Requests"
	maxRequestBytes        = 4 << 20
	headerTimeout          = 10 * time.Second
	inferenceTimeout       = 2 * time.Minute
	shutdownTimeout        = 30 * time.Second
	reportFlushTimeout     = 10 * time.Second
	healthProbeInterval    = 30 * time.Second
	healthProbeTimeout     = 5 * time.Second
	healthFailureThreshold = 3
)
