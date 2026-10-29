package main

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"sort"
	"strings"
	"sync"
	"time"
)

// forwardLimits bounds the forward scheduler and sizes request deadlines.
type forwardLimits struct {
	// maxInflight is the number of model forward passes the workers can run
	// at once (one per worker): the lane count.
	maxInflight int
	// maxQueued is how many windows may wait for a lane before new ones are
	// refused.
	maxQueued int
	// windowTokens/windowOverlap are the workers' window geometry, used to
	// estimate a request's window count for its deadline.
	windowTokens  int
	windowOverlap int
	// A /redact request gets requestBaseTimeout plus windowTimeout per
	// estimated window, capped at maxRequest.
	windowTimeout time.Duration
	maxRequest    time.Duration
	// forwardTimeout is the sharp deadline on one gateway→worker forward
	// (~40× a window): the wedge detector. Exceeding it fails the owning
	// /redact and marks the worker unhealthy so the container is replaced.
	forwardTimeout time.Duration
	// shortTokens: requests at or under this many tokens are "short": their
	// windows go to the front of the queue and skip the hold rule.
	shortTokens int
	// horizons drive the predicted-short-lane rule. After a long window is
	// dispatched, the k-th soonest-freeing OTHER lane must be predicted to
	// free within horizons[k]; otherwise the window is held until a lane
	// frees or the prediction comes within range. Empty disables the rule.
	horizons []time.Duration
	// armed > 0 applies the hold rule only while a short request has been
	// seen within this long, so an idle night runs flat out.
	armed time.Duration
	// msPerToken seeds the EWMA used to predict when busy lanes free.
	msPerToken float64
}

var errForwardQueueFull = errors.New("forward queue full")

// lane is one forward slot on one worker.
type lane struct {
	worker *worker
	busy   bool
	start  time.Time
	tokens int
}

// forwardItem is one window waiting for a lane. ready receives the lane it
// was assigned; the sender never blocks because the channel holds one.
type forwardItem struct {
	key        string
	request    string // the /redact request this window belongs to
	tokens     int
	short      bool
	ready      chan *lane // receives the lane, or nil when the request was cancelled
	enqueuedAt time.Time
}

// callerQueue holds waiting windows grouped by caller and served round-robin
// across callers, FIFO within one: a caller's windows only ever wait behind
// its own. Both the short and the long class use one.
type callerQueue struct {
	queues map[string][]*forwardItem
	keys   []string // callers with queued windows, round-robin order
	next   int      // index into keys of the caller served next
	size   int
}

func newCallerQueue() *callerQueue {
	return &callerQueue{queues: make(map[string][]*forwardItem)}
}

func (q *callerQueue) push(item *forwardItem) {
	if _, pending := q.queues[item.key]; !pending {
		q.keys = append(q.keys, item.key)
	}
	q.queues[item.key] = append(q.queues[item.key], item)
	q.size++
}

// peek returns the window to dispatch next and the function that removes it.
func (q *callerQueue) peek() (*forwardItem, func()) {
	key := q.keys[q.next]
	queue := q.queues[key]
	item := queue[0]
	return item, func() {
		q.size--
		if len(queue) == 1 {
			// Removing the current key shifts the next caller into this slot,
			// so the cursor stays put.
			q.dropKey(key)
			return
		}
		q.queues[key] = queue[1:]
		q.next = (q.next + 1) % len(q.keys)
	}
}

// remove withdraws every window matching keep==false and returns them.
func (q *callerQueue) remove(drop func(*forwardItem) bool) []*forwardItem {
	var removed []*forwardItem
	for _, key := range append([]string(nil), q.keys...) {
		kept := q.queues[key][:0]
		for _, item := range q.queues[key] {
			if drop(item) {
				removed = append(removed, item)
			} else {
				kept = append(kept, item)
			}
		}
		if len(kept) == 0 {
			q.dropKey(key)
		} else {
			q.queues[key] = kept
		}
	}
	q.size -= len(removed)
	return removed
}

func (q *callerQueue) dropKey(key string) {
	delete(q.queues, key)
	for i, k := range q.keys {
		if k == key {
			q.keys = append(q.keys[:i], q.keys[i+1:]...)
			if i < q.next {
				q.next--
			}
			break
		}
	}
	if len(q.keys) == 0 || q.next >= len(q.keys) {
		q.next = 0
	}
}

// shortWaitBuckets are the upper bounds (seconds) of the short-window
// queue-wait histogram exported on /metrics: how long a short request's
// window sat in the queue before a lane took it. This is the number the
// hold rule is meant to bound, separated from everything else in a
// request's latency.
var shortWaitBuckets = []float64{0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10}

// forwardScheduler hands windows to lanes. Short requests' windows are
// served before long ones; within each class callers (the hash the /redact
// proxy stamps on its request) are served round-robin, FIFO per caller, and
// each window goes to the worker with the fewest forwards in flight. A
// coordinator posts every window of a request at once; when the request
// ends (answered, timed out, or the client gone) the gateway cancels its
// queued windows here. The hold rule (see forwardLimits) keeps a lane
// predicted to free soon so a short request arriving mid-way through long
// documents waits about a horizon, not a whole window.
type forwardScheduler struct {
	mu       sync.Mutex
	lanes    []*lane
	workers  []*worker
	limits   forwardLimits
	short    *callerQueue
	long     *callerQueue
	queued   int
	inflight int

	msPerToken float64
	lastShort  time.Time
	// holdSince/holdLanes account lane-seconds left idle by the hold rule.
	holdSince   time.Time
	holdLanes   int
	holdSeconds float64
	// short-window queue wait histogram (see shortWaitBuckets): counts per
	// bucket (cumulative, Prometheus style), total count and sum.
	shortWaitCounts [9]int64
	shortWaitSum    float64
	shortWaitCount  int64
	holdTimer       *time.Timer
	now             func() time.Time
}

func newForwardScheduler(workers []*worker, limits forwardLimits) *forwardScheduler {
	if limits.maxInflight < 1 {
		limits.maxInflight = 1
	}
	if limits.msPerToken <= 0 {
		limits.msPerToken = defaultMsPerToken
	}
	s := &forwardScheduler{
		workers:    workers,
		limits:     limits,
		short:      newCallerQueue(),
		long:       newCallerQueue(),
		msPerToken: limits.msPerToken,
		now:        time.Now,
	}
	for i := range limits.maxInflight {
		s.lanes = append(s.lanes, &lane{worker: workers[i%len(workers)]})
	}
	return s
}

// enqueue registers a window of request for caller key. tokens is the
// window's length (for the ETA model); short marks a window of a short
// request. It fails when the queue budget is exhausted: nothing queued now
// could finish inside the request ceiling anyway, so refusing early is
// kinder than timing out later.
func (s *forwardScheduler) enqueue(key, request string, tokens int, short bool) (*forwardItem, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.queued >= s.limits.maxQueued {
		return nil, errForwardQueueFull
	}
	item := &forwardItem{key: key, request: request, tokens: tokens, short: short, ready: make(chan *lane, 1), enqueuedAt: s.now()}
	if short {
		s.lastShort = s.now()
		s.short.push(item)
	} else {
		s.long.push(item)
	}
	s.queued++
	s.dispatchLocked()
	return item, nil
}

// cancel withdraws a window that is still waiting. It reports false when the
// window was already assigned a lane, in which case the caller must take the
// lane from item.ready and release it.
func (s *forwardScheduler) cancel(item *forwardItem) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	class := s.long
	if item.short {
		class = s.short
	}
	removed := class.remove(func(queued *forwardItem) bool { return queued == item })
	s.queued -= len(removed)
	return len(removed) == 1
}

// cancelRequest withdraws every queued window of a request whose /redact has
// ended, waking their /forward handlers with a nil lane. Windows already on
// a lane finish; nothing else of the request runs. Returns how many were
// dropped.
func (s *forwardScheduler) cancelRequest(request string) int {
	if request == "" {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	drop := func(item *forwardItem) bool { return item.request == request }
	removed := append(s.short.remove(drop), s.long.remove(drop)...)
	s.queued -= len(removed)
	for _, item := range removed {
		item.ready <- nil
	}
	if len(removed) > 0 {
		s.dispatchLocked()
	}
	return len(removed)
}

// release returns a lane, feeds the ETA model with the observed speed when
// the forward completed normally, and dispatches whatever is waiting.
func (s *forwardScheduler) release(l *lane, completed bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Learn the per-token rate only from windows large enough that the
	// forward's fixed overhead is negligible: a 30-token chat call measures
	// ~4 ms/token against ~0.4 for a full window, and one such sample every
	// half second would inflate every lane's ETA 10x and make the hold rule
	// hold constantly (measured: ~50% lane idle under a doc burst).
	if completed && l.tokens >= rateSampleMinTokens {
		observed := float64(s.now().Sub(l.start).Milliseconds()) / float64(l.tokens)
		s.msPerToken = 0.8*s.msPerToken + 0.2*observed
	}
	l.busy = false
	l.tokens = 0
	l.worker.forwards--
	s.inflight--
	s.dispatchLocked()
}

type forwardStats struct {
	queued, inflight int
	holdSeconds      float64
	msPerToken       float64
	shortWaitCounts  [9]int64
	shortWaitSum     float64
	shortWaitCount   int64
}

func (s *forwardScheduler) observeShortWaitLocked(seconds float64) {
	for i, upper := range shortWaitBuckets {
		if seconds <= upper {
			s.shortWaitCounts[i]++
		}
	}
	s.shortWaitCounts[len(shortWaitBuckets)]++ // +Inf
	s.shortWaitSum += seconds
	s.shortWaitCount++
}

// shortWaitMetrics renders the histogram in Prometheus text format.
func (fs forwardStats) shortWaitMetrics() string {
	var b strings.Builder
	for i, upper := range shortWaitBuckets {
		fmt.Fprintf(&b, "pii_forward_short_wait_seconds_bucket{le=\"%g\"} %d\n", upper, fs.shortWaitCounts[i])
	}
	fmt.Fprintf(&b, "pii_forward_short_wait_seconds_bucket{le=\"+Inf\"} %d\n", fs.shortWaitCounts[len(shortWaitBuckets)])
	fmt.Fprintf(&b, "pii_forward_short_wait_seconds_sum %.4f\npii_forward_short_wait_seconds_count %d\n", fs.shortWaitSum, fs.shortWaitCount)
	return b.String()
}

func (s *forwardScheduler) stats() forwardStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.settleHoldLocked()
	return forwardStats{queued: s.queued, inflight: s.inflight, holdSeconds: s.holdSeconds, msPerToken: s.msPerToken,
		shortWaitCounts: s.shortWaitCounts, shortWaitSum: s.shortWaitSum, shortWaitCount: s.shortWaitCount}
}

// settleHoldLocked books the idle lane-seconds accrued since the hold began.
func (s *forwardScheduler) settleHoldLocked() {
	if s.holdLanes > 0 {
		s.holdSeconds += s.now().Sub(s.holdSince).Seconds() * float64(s.holdLanes)
		s.holdSince = s.now()
	}
}

func (s *forwardScheduler) dispatchLocked() {
	s.settleHoldLocked()
	s.holdLanes = 0
	if s.holdTimer != nil {
		s.holdTimer.Stop()
		s.holdTimer = nil
	}
	for s.inflight < len(s.lanes) && s.queued > 0 {
		item, pop := s.nextLocked()
		l := s.freeLaneLocked()
		if !item.short && s.holdActiveLocked() {
			if wait := s.holdDelayLocked(l); wait > 0 {
				s.holdSince = s.now()
				s.holdLanes = len(s.lanes) - s.inflight
				s.holdTimer = time.AfterFunc(wait, func() {
					s.mu.Lock()
					defer s.mu.Unlock()
					s.dispatchLocked()
				})
				return
			}
		}
		pop()
		s.queued--
		s.inflight++
		if item.short {
			s.observeShortWaitLocked(s.now().Sub(item.enqueuedAt).Seconds())
		}
		l.busy, l.start, l.tokens = true, s.now(), item.tokens
		l.worker.forwards++
		item.ready <- l
	}
}

// nextLocked peeks the window to dispatch next and returns the function that
// removes it from its queue.
func (s *forwardScheduler) nextLocked() (*forwardItem, func()) {
	if s.short.size > 0 {
		return s.short.peek()
	}
	return s.long.peek()
}

func (s *forwardScheduler) holdActiveLocked() bool {
	if len(s.limits.horizons) == 0 {
		return false
	}
	if s.limits.armed > 0 && (s.lastShort.IsZero() || s.now().Sub(s.lastShort) > s.limits.armed) {
		return false
	}
	return true
}

// holdDelayLocked evaluates the hold rule for a long window about to take
// lane chosen. It returns 0 when the dispatch may proceed, otherwise how
// long until the predictions could satisfy the rule (a lane release will
// re-evaluate sooner).
func (s *forwardScheduler) holdDelayLocked(chosen *lane) time.Duration {
	now := s.now()
	var etas []time.Duration // predicted time until each other lane frees
	for _, l := range s.lanes {
		if l == chosen {
			continue
		}
		etas = append(etas, s.etaLocked(l, now))
	}
	sort.Slice(etas, func(i, j int) bool { return etas[i] < etas[j] })
	var wait time.Duration
	for k, horizon := range s.limits.horizons {
		if k >= len(etas) {
			break
		}
		if over := etas[k] - horizon; over > wait {
			wait = over
		}
	}
	return wait
}

func (s *forwardScheduler) etaLocked(l *lane, now time.Time) time.Duration {
	if !l.busy {
		return 0
	}
	predicted := l.start.Add(time.Duration(float64(l.tokens) * s.msPerToken * float64(time.Millisecond)))
	if remaining := predicted.Sub(now); remaining > 0 {
		return remaining
	}
	return 0
}

// freeLaneLocked picks a free lane on the worker with the fewest forwards
// in flight. Lanes are spread evenly over workers, so whenever the global
// cap allows a dispatch the least-busy worker has a free lane.
func (s *forwardScheduler) freeLaneLocked() *lane {
	var best *lane
	for _, l := range s.lanes {
		if l.busy {
			continue
		}
		if best == nil || l.worker.forwards < best.worker.forwards {
			best = l
		}
	}
	return best
}

// requestTimeout scales the /redact deadline with the input size, estimated
// from the body length before the coordinator has tokenized it. The
// estimate (one token per four bytes) overshoots for prose, which is the
// intended direction for a timeout; the ceiling still bounds it.
func (l forwardLimits) requestTimeout(bodyBytes int) time.Duration {
	stride := l.windowTokens - 2*l.windowOverlap
	if stride < 1 {
		stride = 1
	}
	tokens := (bodyBytes + 3) / 4
	windows := (tokens + stride - 1) / stride
	if windows < 1 {
		windows = 1
	}
	timeout := requestBaseTimeout + time.Duration(windows)*l.windowTimeout
	if timeout > l.maxRequest {
		return l.maxRequest
	}
	return timeout
}

// keyHash identifies a caller to the scheduler without ever sending the
// credential to the inference processes.
func keyHash(credential string) string {
	sum := sha256.Sum256([]byte(credential))
	return hex.EncodeToString(sum[:16])
}

// isLoopback reports whether a request came from this host. /forward is an
// internal fan-out path; only the coordinating workers may use it.
func isLoopback(remoteAddr string) bool {
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		host = remoteAddr
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
