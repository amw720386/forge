"""Independent Studio watchdog using the Studio's existing SDK credentials."""
import datetime
import argparse
import time
from lightning_sdk import Studio

# Require an absolute UTC deadline; restarting must not extend the reservation.
parser = argparse.ArgumentParser()
parser.add_argument("--deadline", required=True, help="Absolute ISO-8601 UTC deadline")
parser.add_argument("--org", required=True)
parser.add_argument("--teamspace", default="general")
args = parser.parse_args()
deadline = datetime.datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
if deadline.utcoffset() != datetime.timedelta(0):
    parser.error("deadline must include UTC offset")
studio = Studio("forge-gpu", teamspace=args.teamspace, org=args.org)
print("PHASE13_WATCHDOG_READY deadline=" + deadline.isoformat(), flush=True)
while datetime.datetime.now(datetime.timezone.utc) < deadline:
    time.sleep(5)
for attempt in range(4):
    try:
        studio.stop()
        break
    except Exception:
        print("watchdog stop retry", attempt, flush=True)
        time.sleep(5)
