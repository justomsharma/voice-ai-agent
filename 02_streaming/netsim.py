"""A deliberately bad network, so we can watch what it does to audio.

On localhost, packets arrive in ~0.1 ms, in order, every time. A real
network (Wi-Fi, 4G, the internet) does three things to a stream of packets:

    delay    every packet takes a while: distance, routers, queues
    jitter   that while is different for every packet. If packet 2 is
             delayed 30 ms more than packet 3, packet 3 arrives first
             (reordering).
    loss     some packets never arrive: a full router queue drops them,
             Wi-Fi interference corrupts them

NetSim makes those three decisions for each packet. DelayLine then actually
holds each packet for its delay before putting it on the socket.

We impair on the *send* side because it's simpler than a middle relay
process. To the receiver it makes no difference: a packet that's late or
missing looks exactly the same whoever caused it.
"""

import heapq
import itertools
import random
import threading
import time
from collections.abc import Callable


class NetSim:
    def __init__(self, delay_ms: float = 0.0, jitter_ms: float = 0.0,
                 loss: float = 0.0, seed: int | None = None):
        if not 0.0 <= loss <= 1.0:
            raise ValueError(f"loss must be between 0 and 1 (a fraction), got {loss}")
        if delay_ms < 0 or jitter_ms < 0:
            raise ValueError("delay_ms and jitter_ms must be >= 0")
        self.delay_ms, self.jitter_ms, self.loss = delay_ms, jitter_ms, loss
        # Our own Random instance with an optional seed: the same seed gives
        # the same "bad network", so two runs (e.g. buffer 0 ms vs 200 ms)
        # can be compared fairly with exactly the same losses and delays.
        self._rng = random.Random(seed)

    def decide(self) -> float | None:
        """None = drop this packet. Otherwise: how long to hold it (seconds)."""
        if self._rng.random() < self.loss:
            return None
        # Uniform jitter: each packet gets its own delay somewhere in
        # [delay - jitter, delay + jitter]. Real jitter is lumpier (bursts
        # when a queue fills up), but uniform is easy to reason about: the
        # jitter buffer has to cover a spread of 2 x jitter.
        ms = self.delay_ms + self._rng.uniform(-self.jitter_ms, self.jitter_ms)
        # A packet can't arrive before it was sent.
        return max(0.0, ms) / 1000


class DelayLine:
    """Holds each packet until its due time, then calls send_fn(data).

    One background thread plus a heap (priority queue) ordered by due time.
    Packets leave in *due-time* order, not arrival order. That's how jitter
    turns into reordering.
    """

    def __init__(self, send_fn: Callable[[bytes], None]):
        self._send = send_fn
        # Entries are (due_time, tie, data). The tie counter is there because
        # if two due times are equal, heapq would compare the next field.
        # A unique increasing number settles it (and keeps FIFO order) so it
        # never has to compare the bytes.
        self._heap: list[tuple[float, int, bytes]] = []
        self._tie = itertools.count()
        self._lock = threading.Lock()
        self._closing = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def put(self, delay_s: float, data: bytes) -> None:
        with self._lock:
            heapq.heappush(self._heap, (time.perf_counter() + delay_s, next(self._tie), data))

    def close(self) -> None:
        """Block until every queued packet has been sent, then stop."""
        with self._lock:
            self._closing = True
        self._thread.join()

    def _run(self) -> None:
        while True:
            data = None
            with self._lock:
                if self._heap and self._heap[0][0] <= time.perf_counter():
                    data = heapq.heappop(self._heap)[2]
                elif not self._heap and self._closing:
                    return
            if data is not None:
                # Send outside the lock so put() is never stuck behind a
                # (comparatively slow) socket call.
                self._send(data)
            else:
                # Poll every 1 ms instead of a timed Condition.wait(): on
                # Windows a timed wait can round up to the ~15.6 ms system
                # timer tick, which would add jitter we never asked for.
                # time.sleep uses a high-resolution timer since Python 3.11.
                time.sleep(0.001)
