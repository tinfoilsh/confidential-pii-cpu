package main

import "time"

const (
	listenAddress = ":8001"
	inferenceHost = "127.0.0.1"
	// Signed usage context from a parent service: tolerated clock skew and
	// the deepest nesting accepted (router -> websearch -> privacy filter).
	usageContextMaxSkew  = 10 * time.Minute
	usageContextMaxDepth = 4
	// Catalog model name the usage event is billed under.
	billingModel           = "pii-filter"
	inferenceBasePort      = 8002
	maxWorkers             = 8
	controlplaneURL        = "https://api.tinfoil.sh"
	reporterID             = "pii-filter"
	redactPath             = "/redact"
	forwardPath            = "/forward"
	healthPath             = "/health"
	metricsPath            = "/metrics"
	billableRequestsHeader = "X-Tinfoil-Billable-Requests"
	// Billing receipt detail: the input tokens metered for this request, so
	// a calling service (websearch) can carry the per-token part of the fee
	// up to the router's cost trailer alongside the request count.
	billableInputTokensHeader = "X-Tinfoil-Billable-Input-Tokens"
	// keyHashHeader carries a hash of the caller's credential from the /redact
	// proxy to the coordinating worker, which echoes it on every /forward it
	// fans out, so the forward scheduler can queue windows per caller.
	keyHashHeader = "X-Tinfoil-Key-Hash"
	// inputTokensHeader carries the request's input token count from the
	// coordinator back to the gateway, which meters it and strips it.
	inputTokensHeader = "X-Tinfoil-Input-Tokens"
	// requestTokensHeader carries the owning request's total token count on
	// each /forward so the scheduler can tell short requests from long ones.
	requestTokensHeader = "X-Tinfoil-Request-Tokens"
	// The /redact request a window belongs to: minted by the gateway, handed
	// to the coordinator, echoed on every /forward, used to cancel the
	// request's queued windows when the /redact ends.
	requestIDHeader = "X-Tinfoil-Request-ID"
	// positionOffsetHeader carries a window's absolute token offset on each
	// /forward, relayed to the worker so the window runs at its true
	// positions (bit-for-bit the same rotary angles as an unchunked pass).
	positionOffsetHeader = "X-Tinfoil-Position-Offset"
	// forwardShapeHeader is the worker's "rows,classes" shape of a /forward
	// response body, relayed to the coordinator untouched.
	forwardShapeHeader = "X-Tinfoil-Shape"
	maxRequestBytes    = 4 << 20
	// rateSampleMinTokens: only forwards at least this long update the
	// ms-per-token estimate behind lane ETAs (see forwardScheduler.release).
	rateSampleMinTokens     = 1024
	defaultMaxQueuedWindows = 4096
	headerTimeout           = 10 * time.Second
	// A /redact request may run for requestBaseTimeout plus
	// OPF_WINDOW_TIMEOUT_SECONDS per window, up to OPF_MAX_REQUEST_SECONDS;
	// the latter also bounds a queued window's wait and the server's
	// read/idle timeouts.
	requestBaseTimeout    = 30 * time.Second
	defaultWindowTimeout  = 5 * time.Second
	defaultForwardTimeout = 30 * time.Second
	defaultMaxRequest     = 15 * time.Minute
	// Window geometry: the model's receptive field is layers × context
	// (8 × 128 = 1024 for the pinned checkpoint), so the overlap must be at
	// least that for chunked scores to equal one long pass. The workers
	// enforce the floor from the checkpoint config at startup.
	defaultWindowTokens  = 6144
	defaultWindowOverlap = 1024
	// Lab-measured on TDX (tf-test benchmarks/safeguards/opf/chunking):
	// horizons 0.5,1.5 keep prod-shaped short calls under ~1 s p99 behind a
	// 50k-token document at no document cost; 0.5 ms/token at 4 × 8 threads.
	defaultShortLaneHorizon = "0.5,1.5"
	defaultShortLaneArmed   = 30 * time.Second
	defaultMsPerToken       = 0.5
	shutdownTimeout         = 30 * time.Second
	reportFlushTimeout      = 10 * time.Second
	healthProbeInterval     = 30 * time.Second
	healthProbeTimeout      = 5 * time.Second
	// Keep-alive hygiene on the loopback hops. uvicorn closes an idle
	// keep-alive connection after workerKeepAlive; the gateway must drop its
	// idle connections well before that, or a POST reused on a connection the
	// worker just closed fails with EOF (Go only auto-retries idempotent
	// requests) and surfaces as a sporadic 502 with nothing in the worker's
	// logs. Seen in the lab at ~1 in 400 requests under Poisson traffic.
	workerKeepAlive        = 120 * time.Second
	workerIdleConnTimeout  = 30 * time.Second
	healthFailureThreshold = 3
)
