//go:build linux

package main

import "golang.org/x/sys/unix"

// pinToCPUs restricts a freshly started worker process to an inclusive CPU
// range. Threads the process creates afterwards inherit the mask.
func pinToCPUs(pid, start, end int) error {
	var set unix.CPUSet
	for cpu := start; cpu <= end; cpu++ {
		set.Set(cpu)
	}
	return unix.SchedSetaffinity(pid, &set)
}
