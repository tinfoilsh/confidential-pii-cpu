//go:build !linux

package main

import "errors"

// pinToCPUs is linux-only; OPF_PIN_WORKERS fails closed elsewhere so a
// misconfigured deployment is caught at startup rather than silently
// running unpinned.
func pinToCPUs(pid, start, end int) error {
	return errors.New("OPF_PIN_WORKERS requires linux")
}
