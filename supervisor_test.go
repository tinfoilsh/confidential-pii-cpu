package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// TestMain lets the test binary double as a stand-in worker process: with
// PII_FAKE_WORKER_PORT set it serves /health on that port (200, or 503 when
// the file named by PII_FAKE_WORKER_SICK exists) until killed.
func TestMain(m *testing.M) {
	if port := os.Getenv("PII_FAKE_WORKER_PORT"); port != "" {
		sick := os.Getenv("PII_FAKE_WORKER_SICK")
		http.HandleFunc(healthPath, func(w http.ResponseWriter, r *http.Request) {
			if sick != "" {
				if _, err := os.Stat(sick); err == nil {
					http.Error(w, "sick", http.StatusServiceUnavailable)
					return
				}
			}
			_, _ = w.Write([]byte("ok"))
		})
		if err := http.ListenAndServe(net.JoinHostPort(inferenceHost, port), nil); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	os.Exit(m.Run())
}

func freePortBase(t *testing.T, n int) int {
	t.Helper()
	for base := 20000; base < 60000; base += 100 {
		ok := true
		for i := range n {
			l, err := net.Listen("tcp", fmt.Sprintf("%s:%d", inferenceHost, base+i))
			if err != nil {
				ok = false
				break
			}
			l.Close()
		}
		if ok {
			return base
		}
	}
	t.Fatal("no free port range")
	return 0
}

// fakeSupervisor runs the supervisor over stand-in worker processes.
func fakeSupervisor(t *testing.T, n int, sickFile string) (*workerSupervisor, []*worker, context.CancelFunc) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	workers := make([]*worker, n)
	for i := range workers {
		target, _ := url.Parse("http://" + inferenceHost)
		workers[i] = &worker{index: i, target: target}
	}
	s := newWorkerSupervisor(ctx, workers)
	s.basePort = freePortBase(t, n)
	s.interval = 50 * time.Millisecond
	s.bootTimeout = 5 * time.Second
	s.command = func(ctx context.Context, i int, port string) *exec.Cmd {
		cmd := exec.CommandContext(ctx, os.Args[0])
		cmd.Env = append(os.Environ(), "PII_FAKE_WORKER_PORT="+port, "PII_FAKE_WORKER_SICK="+sickFile)
		return cmd
	}
	if err := s.startAll(); err != nil {
		cancel()
		t.Fatal(err)
	}
	for i := range workers {
		if err := s.awaitReady(i); err != nil {
			cancel()
			t.Fatal(err)
		}
	}
	t.Cleanup(func() { cancel(); s.stop() })
	return s, workers, cancel
}

func pidOf(s *workerSupervisor, i int) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.procs[i].cmd.Process.Pid
}

// A wedged worker is replaced in place: new process, siblings untouched,
// the scheduler told once it answers /health; a worker that dies on its own
// is replaced the same way; neither is fatal.
func TestSupervisorReplacesWedgedAndDeadWorkers(t *testing.T) {
	s, workers, _ := fakeSupervisor(t, 2, "")
	ready := make(chan int, 4)
	s.onReady = func(i int) { ready <- i }
	before0, before1 := pidOf(s, 0), pidOf(s, 1)

	s.wedged(0)
	select {
	case i := <-ready:
		if i != 0 {
			t.Fatalf("ready for worker %d", i)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("wedged worker was not replaced")
	}
	if pidOf(s, 0) == before0 || pidOf(s, 1) != before1 || workers[0].wedged.Load() {
		t.Fatalf("replacement: pid0 %d→%d, pid1 %d→%d, wedged=%v", before0, pidOf(s, 0), before1, pidOf(s, 1), workers[0].wedged.Load())
	}

	// Worker 1 dies on its own (a crash): replaced, not fatal.
	proc1 := func() *os.Process { s.mu.Lock(); defer s.mu.Unlock(); return s.procs[1].cmd.Process }()
	_ = proc1.Kill()
	select {
	case i := <-ready:
		if i != 1 {
			t.Fatalf("ready for worker %d", i)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("dead worker was not replaced")
	}
	select {
	case err := <-s.fatal:
		t.Fatalf("replacement escalated to fatal: %v", err)
	default:
	}
}

// Escalation: more than maxWorkerRestarts replacements of one worker inside
// the window is fatal, so the process exits and the container is replaced.
func TestSupervisorGivesUpAfterRepeatedReplacements(t *testing.T) {
	s, _, _ := fakeSupervisor(t, 1, "")
	ready := make(chan int, 8)
	s.onReady = func(i int) { ready <- i }
	for n := 0; n < maxWorkerRestarts; n++ {
		s.wedged(0)
		select {
		case <-ready:
		case <-time.After(10 * time.Second):
			t.Fatalf("replacement %d did not complete", n+1)
		}
	}
	s.wedged(0)
	select {
	case err := <-s.fatal:
		if !strings.Contains(err.Error(), "giving up") {
			t.Fatalf("unexpected fatal: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("repeated wedges did not escalate")
	}
}

// A replacement that never becomes healthy is fatal too.
func TestSupervisorGivesUpWhenReplacementStaysSick(t *testing.T) {
	sick := t.TempDir() + "/sick"
	s, _, _ := fakeSupervisor(t, 1, sick)
	s.bootTimeout = 500 * time.Millisecond
	if err := os.WriteFile(sick, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	s.wedged(0)
	select {
	case err := <-s.fatal:
		if !strings.Contains(err.Error(), "did not become healthy") {
			t.Fatalf("unexpected fatal: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("sick replacement did not escalate")
	}
}

// Failing health probes (not just a wedge) replace the worker as well.
func TestSupervisorReplacesWorkerWhoseProbesFail(t *testing.T) {
	sick := t.TempDir() + "/sick"
	s, workers, _ := fakeSupervisor(t, 1, sick)
	ready := make(chan int, 2)
	s.onReady = func(i int) { ready <- i }
	before := pidOf(s, 0)
	if err := os.WriteFile(sick, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	// Probes fail healthFailureThreshold times, the worker is killed; the
	// replacement must come up healthy, so clear the flag once it is killed.
	go func() {
		for pidOf(s, 0) == before {
			time.Sleep(10 * time.Millisecond)
		}
		_ = os.Remove(sick)
	}()
	select {
	case <-ready:
	case <-time.After(10 * time.Second):
		t.Fatal("worker with failing probes was not replaced")
	}
	if pidOf(s, 0) == before || workers[0].wedged.Load() {
		t.Fatal("worker not replaced cleanly")
	}
}

// Shutdown: worker exits caused by the supervisor are not reported fatal.
func TestSupervisorStopIsNotFatal(t *testing.T) {
	s, _, cancel := fakeSupervisor(t, 2, "")
	cancel()
	s.stop()
	select {
	case err := <-s.fatal:
		t.Fatalf("shutdown reported fatal: %v", err)
	default:
	}
	if !errors.Is(s.ctx.Err(), context.Canceled) {
		t.Fatal("context not cancelled")
	}
}
