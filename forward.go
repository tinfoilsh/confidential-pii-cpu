package main

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net"
	"sync"
)

// forwardLimits bounds the forward scheduler. maxInflight is the number of
// model forward passes the workers can run at once (workers × their
// per-process concurrency); maxQueued is how many windows may wait for a
// lane before new ones are refused.
type forwardLimits struct {
	maxInflight int
	maxQueued   int
}

var errForwardQueueFull = errors.New("forward queue full")

// forwardItem is one window waiting for a worker. ready receives the worker
// it was assigned to; the sender never blocks because the channel holds one.
type forwardItem struct {
	key   string
	ready chan *worker
}

// forwardScheduler hands windows to workers fairly across callers. Every
// caller (keyed by the hash the /redact proxy stamps on its request) has a
// FIFO queue; the scheduler round-robins across callers with pending work
// and sends each window to the worker with the fewest forwards in flight.
// A caller that fans out a 40k-token document therefore interleaves with a
// caller sending one short request instead of holding every lane.
type forwardScheduler struct {
	mu       sync.Mutex
	workers  []*worker
	queues   map[string][]*forwardItem
	keys     []string // callers with queued windows, in round-robin order
	next     int      // index into keys of the caller served next
	queued   int
	inflight int
	limits   forwardLimits
}

func newForwardScheduler(workers []*worker, limits forwardLimits) *forwardScheduler {
	if limits.maxInflight < 1 {
		limits.maxInflight = 1
	}
	return &forwardScheduler{workers: workers, queues: make(map[string][]*forwardItem), limits: limits}
}

// enqueue registers a window for the given caller. It fails when the queue
// budget is exhausted: nothing queued now could finish inside the inference
// timeout anyway, so refusing early is kinder than timing out later.
func (s *forwardScheduler) enqueue(key string) (*forwardItem, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.queued >= s.limits.maxQueued {
		return nil, errForwardQueueFull
	}
	item := &forwardItem{key: key, ready: make(chan *worker, 1)}
	if _, pending := s.queues[key]; !pending {
		s.keys = append(s.keys, key)
	}
	s.queues[key] = append(s.queues[key], item)
	s.queued++
	s.dispatchLocked()
	return item, nil
}

// cancel withdraws a window that is still waiting. It reports false when the
// window was already assigned a worker, in which case the caller must take
// the worker from item.ready and release it.
func (s *forwardScheduler) cancel(item *forwardItem) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	queue := s.queues[item.key]
	for i, queuedItem := range queue {
		if queuedItem != item {
			continue
		}
		queue = append(queue[:i], queue[i+1:]...)
		s.queued--
		if len(queue) == 0 {
			s.dropKeyLocked(item.key)
		} else {
			s.queues[item.key] = queue
		}
		return true
	}
	return false
}

// release returns a worker lane and dispatches whatever is waiting.
func (s *forwardScheduler) release(w *worker) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inflight--
	w.forwards--
	s.dispatchLocked()
}

func (s *forwardScheduler) stats() (queued, inflight int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.queued, s.inflight
}

func (s *forwardScheduler) dispatchLocked() {
	for s.inflight < s.limits.maxInflight && s.queued > 0 {
		key := s.keys[s.next]
		queue := s.queues[key]
		item := queue[0]
		if len(queue) == 1 {
			// Removing the current key shifts the next caller into this slot,
			// so the cursor stays put.
			s.dropKeyLocked(key)
		} else {
			s.queues[key] = queue[1:]
			s.next = (s.next + 1) % len(s.keys)
		}
		s.queued--
		s.inflight++
		w := s.leastBusyLocked()
		w.forwards++
		item.ready <- w
	}
}

func (s *forwardScheduler) dropKeyLocked(key string) {
	delete(s.queues, key)
	for i, k := range s.keys {
		if k == key {
			s.keys = append(s.keys[:i], s.keys[i+1:]...)
			if i < s.next {
				s.next--
			}
			break
		}
	}
	if len(s.keys) == 0 {
		s.next = 0
	} else {
		s.next %= len(s.keys)
	}
}

// leastBusyLocked picks the worker with the fewest forwards in flight. With
// maxInflight = len(workers) × per-worker concurrency, the least-busy worker
// always has a free lane whenever the global cap allows a dispatch.
func (s *forwardScheduler) leastBusyLocked() *worker {
	best := s.workers[0]
	for _, w := range s.workers[1:] {
		if w.forwards < best.forwards {
			best = w
		}
	}
	return best
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
