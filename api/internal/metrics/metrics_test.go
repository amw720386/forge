package metrics

import (
	"testing"
	"time"
)

func TestStageTotalsAndSnapshotIsolation(t *testing.T) {
	c := New()
	c.ObserveStage("queue_wait", 250*time.Microsecond)
	c.ObserveStage("queue_wait", 750*time.Microsecond)
	s := c.Snapshot()
	v := s.Stages["queue_wait"]
	if v.Count != 2 || v.TotalNs != 1000000 || v.MaxNs != 750000 {
		t.Fatalf("bad stage: %+v", v)
	}
	delete(s.Stages, "queue_wait")
	if c.Snapshot().Stages["queue_wait"].Count != 2 {
		t.Fatal("snapshot aliases collector")
	}
	c.ObserveQueueWait(250 * time.Microsecond)
	if c.Snapshot().AvgQueueWaitMs != 0.25 {
		t.Fatal("submillisecond queue wait lost")
	}
}
