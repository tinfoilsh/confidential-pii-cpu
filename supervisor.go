package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
	"time"
)

// workerSupervisor runs the inference worker processes and replaces one
// that wedges, fails its health probes, or dies. Escalation: a worker is
// restarted in place (its siblings keep serving, its queued windows go
// elsewhere); a worker that cannot be brought back, or that needs more than
// maxWorkerRestarts restarts within workerRestartWindow, is fatal and the
// whole process exits so the container's restart policy replaces the set.
type workerSupervisor struct {
	ctx     context.Context
	workers []*worker
	// fatal receives the first process-level failure; run() exits on it.
	fatal chan error
	// onReady is called after a worker (re)starts and answers /health, so
	// the scheduler resumes dispatching to it.
	onReady func(worker int)

	// command builds the process for worker i (uvicorn in production; tests
	// substitute a stand-in) and basePort is the first worker's port.
	command  func(ctx context.Context, i int, port string) *exec.Cmd
	basePort int
	interval time.Duration
	// bootTimeout bounds how long a replaced worker may take to answer
	// /health; tests shorten it.
	bootTimeout time.Duration

	mu       sync.Mutex
	procs    []*workerProc
	restarts [][]time.Time
	stopping bool
}

type workerProc struct {
	cmd         *exec.Cmd
	cancelWatch context.CancelFunc
	exited      chan struct{}
	// ready is closed once the process has answered /health (or boot gave
	// up: bootErr is then set). The health watcher owns boot; replace waits
	// on it.
	ready   chan struct{}
	bootErr error
	// expected marks an exit the supervisor caused (restart or shutdown).
	expected bool
}

func newWorkerSupervisor(ctx context.Context, workers []*worker) *workerSupervisor {
	return &workerSupervisor{
		ctx:      ctx,
		workers:  workers,
		fatal:    make(chan error, 1),
		command:  uvicornCommand,
		basePort: inferenceBasePort,
		procs:    make([]*workerProc, len(workers)),
		restarts: make([][]time.Time, len(workers)),
	}
}

func uvicornCommand(ctx context.Context, _ int, port string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, "uvicorn", "server:app", "--host", inferenceHost, "--port", port, "--no-access-log", "--timeout-keep-alive", strconv.Itoa(int(workerKeepAlive.Seconds())))
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	// Workers post their windows to this gateway's /queue unless the
	// deployment points them elsewhere (unset in the worker means
	// in-process, single-worker development).
	cmd.Env = append(os.Environ(), "OPF_QUEUE_URL="+queueURLForWorkers())
	return cmd
}

func (s *workerSupervisor) port(i int) string { return strconv.Itoa(s.basePort + i) }

func (s *workerSupervisor) url(i int) string { return "http://" + inferenceHost + ":" + s.port(i) }

// startAll spawns every worker. Failure to start any is returned directly;
// the caller tears down.
func (s *workerSupervisor) startAll() error {
	for i := range s.workers {
		if err := s.spawn(i); err != nil {
			return fmt.Errorf("start inference worker %d: %w", i, err)
		}
	}
	return nil
}

// spawn starts worker i and its health watcher. Caller holds no lock.
func (s *workerSupervisor) spawn(i int) error {
	cmd := s.command(s.ctx, i, s.port(i))
	cmd.Cancel = func() error { return cmd.Process.Signal(syscall.SIGTERM) }
	cmd.WaitDelay = shutdownTimeout
	if err := cmd.Start(); err != nil {
		return err
	}
	watchCtx, cancelWatch := context.WithCancel(s.ctx)
	proc := &workerProc{cmd: cmd, cancelWatch: cancelWatch, exited: make(chan struct{}), ready: make(chan struct{})}
	s.mu.Lock()
	s.procs[i] = proc
	s.mu.Unlock()
	go func() {
		err := cmd.Wait()
		close(proc.exited)
		s.mu.Lock()
		expected := proc.expected || s.stopping
		s.mu.Unlock()
		if !expected {
			// A crash is replaced like a wedge, under the same escalation.
			s.replace(i, fmt.Errorf("exited unexpectedly: %v", err))
		}
	}()
	go func() {
		// Probes count only once the worker has answered /health once:
		// loading the model takes longer than healthFailureThreshold probes
		// and a worker killed mid-load would only load again.
		proc.bootErr = s.awaitReady(watchCtx, i)
		close(proc.ready)
		if proc.bootErr != nil {
			return // replace (or shutdown) handles it
		}
		if err, ok := <-watchHealth(watchCtx, s.url(i)+healthPath, s.probeInterval(), true); ok {
			s.replace(i, fmt.Errorf("health probes failed: %w", err))
		}
	}()
	return nil
}

// wedged is the gateway's signal that a forward on worker i exceeded the
// per-window deadline: the process is stuck (most likely inside the model,
// holding the interpreter) and is replaced.
func (s *workerSupervisor) wedged(i int) {
	s.replace(i, errors.New("forward timed out"))
}

// replace kills worker i and spawns a fresh process on its port, then waits
// for it to answer /health before letting the scheduler use it again.
// Runs at most once at a time per worker; a replacement already in progress
// absorbs further signals.
func (s *workerSupervisor) replace(i int, reason error) {
	s.mu.Lock()
	if s.stopping {
		s.mu.Unlock()
		return
	}
	proc := s.procs[i]
	if proc == nil || proc.expected {
		s.mu.Unlock()
		return
	}
	proc.expected = true
	now := time.Now()
	recent := s.restarts[i][:0]
	for _, t := range s.restarts[i] {
		if now.Sub(t) < workerRestartWindow {
			recent = append(recent, t)
		}
	}
	s.restarts[i] = append(recent, now)
	tooMany := len(s.restarts[i]) > maxWorkerRestarts
	s.mu.Unlock()

	s.workers[i].wedged.Store(true)
	if tooMany {
		s.fail(fmt.Errorf("inference worker %d replaced %d times within %s (last: %v); giving up", i, len(s.restarts[i]), workerRestartWindow, reason))
		return
	}
	slog.Error("replacing inference worker", "worker", i, "reason", reason, "restart", len(s.restarts[i]))
	proc.cancelWatch()
	// A wedged forward holds the interpreter, so SIGTERM (handled by the
	// event loop) may never run: kill outright.
	_ = proc.cmd.Process.Kill()
	select {
	case <-proc.exited:
	case <-time.After(shutdownTimeout):
		s.fail(fmt.Errorf("inference worker %d did not exit after kill", i))
		return
	}
	if err := s.spawn(i); err != nil {
		s.fail(fmt.Errorf("respawn inference worker %d: %w", i, err))
		return
	}
	if err := s.booted(i); err != nil {
		s.fail(fmt.Errorf("after restart: %w", err))
		return
	}
	s.workers[i].wedged.Store(false)
	slog.Info("inference worker replaced", "worker", i)
	if s.onReady != nil {
		s.onReady(i)
	}
}

// booted waits for worker i's current process to answer /health, or for
// its boot to give up.
func (s *workerSupervisor) booted(i int) error {
	s.mu.Lock()
	proc := s.procs[i]
	s.mu.Unlock()
	<-proc.ready
	return proc.bootErr
}

// probeInterval is the health-watch cadence; tests shorten it.
func (s *workerSupervisor) probeInterval() time.Duration {
	if s.interval > 0 {
		return s.interval
	}
	return healthProbeInterval
}

func (s *workerSupervisor) awaitReady(ctx context.Context, i int) error {
	client := &http.Client{Timeout: healthProbeTimeout}
	bootTimeout := s.bootTimeout
	if bootTimeout == 0 {
		bootTimeout = workerBootTimeout
	}
	deadline := time.Now().Add(bootTimeout)
	var last error
	for time.Now().Before(deadline) {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if last = probe(ctx, client, s.url(i)+healthPath); last == nil {
			return nil
		}
		time.Sleep(time.Second)
	}
	return fmt.Errorf("inference worker %d did not become healthy within %s: %w", i, bootTimeout, last)
}

func (s *workerSupervisor) fail(err error) {
	select {
	case s.fatal <- err:
	default:
	}
}

// stop marks shutdown so worker exits are expected, and waits for them.
func (s *workerSupervisor) stop() {
	s.mu.Lock()
	s.stopping = true
	procs := append([]*workerProc(nil), s.procs...)
	s.mu.Unlock()
	for _, proc := range procs {
		if proc != nil {
			<-proc.exited
		}
	}
}
