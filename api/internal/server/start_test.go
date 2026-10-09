package server

import (
	"net"
	"strings"
	"testing"
)

func TestStartReturnsOccupiedPortError(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	// Binding must fail before starting the scheduler.
	s := New(Config{Addr: listener.Addr().String()}, nil, nil, nil)
	if err := s.Start(); err == nil || !strings.Contains(err.Error(), "listen on") {
		t.Fatalf("wanted actionable bind error, got %v", err)
	}
}
