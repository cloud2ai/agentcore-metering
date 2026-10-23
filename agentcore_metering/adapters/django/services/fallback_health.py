"""Shared cooldown state; real traffic performs recovery without probes."""

import hashlib
import logging
import time

from django.core.cache import cache

logger = logging.getLogger(__name__)


class ModelAvailability:
    def __init__(self, root, origin, rows, policy):
        self.rows = {str(row.uuid): row for row in rows}
        self.policy = policy
        version = f"{root.uuid}:{root.updated_at}:{origin.updated_at}"
        self.prefix = (
            "agentcore:fallback:"
            + hashlib.sha256(
                version.encode(),
            ).hexdigest()[:32]
        )

    def key(self, target):
        row = self.rows[target.config_uuid]
        version = hashlib.sha256(
            str(row.updated_at).encode(),
        ).hexdigest()[:16]
        return f"{self.prefix}:{target.config_uuid}:{version}"

    def allow(self, target, call_id):
        key = self.key(target)
        try:
            state = cache.get(key)
            if not state:
                return True
            if not self.policy["auto_recover"]:
                return False
            if state["until"] > time.time():
                return False
            # Atomic add allows just one real request to check recovery.
            # Let this lock expire: deleting it via get-then-delete could
            # delete a different request's newly acquired probe lock.
            return cache.add(
                key + ":check",
                call_id,
                timeout=int(target.timeout_seconds)
                * (self.policy["retries"] + 1)
                + 5,
            )
        except Exception as exc:
            logger.warning(
                "Model cooldown cache unavailable (%s)",
                type(exc).__name__,
            )
            return True

    def failure(self, target, reason):
        if reason not in self.policy["conditions"]:
            return
        try:
            cache.set(
                self.key(target),
                {
                    "until": time.time() + self.policy["cooldown_seconds"],
                    "reason": reason,
                },
                timeout=(86400 if self.policy["auto_recover"] else None),
            )
        except Exception as exc:
            logger.warning(
                "Cannot record model cooldown (%s)",
                type(exc).__name__,
            )

    def success(self, target):
        try:
            cache.delete(self.key(target))
        except Exception as exc:
            logger.warning(
                "Cannot clear model cooldown (%s)",
                type(exc).__name__,
            )

    def status(self, target):
        try:
            state = cache.get(self.key(target))
        except Exception:
            return {"state": "unknown", "remaining_seconds": None}
        if not state:
            return {"state": "ready", "remaining_seconds": 0}
        remaining = max(0, int(state["until"] - time.time()))
        return {
            "state": (
                "manual_recovery"
                if not self.policy["auto_recover"]
                else "cooling"
                if remaining
                else "awaiting_request"
            ),
            "remaining_seconds": remaining,
            "reason": state["reason"],
        }
