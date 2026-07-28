"""
Photon Push Notification Service for Project HENRI V2.

Sends real-time push notifications to mobile devices via Pushover, Webhooks, or iMessage/Google Drive inbox bridge
when long-running benchmarks, ARC-AGI-3 runs, or production jobs complete.
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict


class PhotonNotifier:
    """
    Sends push notifications via Pushover API, Google Drive Inbox Bridge, or custom webhooks.
    Configurable via environment variables:
      - PHOTON_PUSHOVER_USER_KEY
      - PHOTON_PUSHOVER_APP_TOKEN
      - PHOTON_WEBHOOK_URL
      - PHOTON_GDRIVE_INBOX_DIR (e.g. G:\My Drive\HENRI_Inbox)
    """

    def __init__(self, user_key: str = None, app_token: str = None, webhook_url: str = None):
        self.user_key = user_key or os.environ.get("PHOTON_PUSHOVER_USER_KEY")
        self.app_token = app_token or os.environ.get("PHOTON_PUSHOVER_APP_TOKEN")
        self.webhook_url = webhook_url or os.environ.get("PHOTON_WEBHOOK_URL")
        self.gdrive_inbox = os.environ.get(
            "PHOTON_GDRIVE_INBOX_DIR", r"G:\My Drive\HENRI_Inbox"
        )

    def send_notification(self, title: str, message: str, priority: int = 0) -> bool:
        """
        Sends a push notification to user's mobile device.
        Tries Pushover API, Webhook, and Google Drive Inbox Bridge.
        """
        success = False

        # 1. Pushover API
        if self.user_key and self.app_token:
            try:
                data = urllib.parse.urlencode(
                    {
                        "token": self.app_token,
                        "user": self.user_key,
                        "title": title,
                        "message": message,
                        "priority": priority,
                    }
                ).encode("utf-8")

                req = urllib.request.Request("https://api.pushover.net/1/messages.json", data=data)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        print(f"[Photon Push] Pushover notification sent successfully: {title}")
                        success = True
            except Exception as e:
                print(f"[Photon Push] Pushover delivery failed: {e}", file=sys.stderr)

        # 2. Webhook
        if self.webhook_url:
            try:
                payload = json.dumps({"title": title, "message": message, "priority": priority}).encode("utf-8")
                req = urllib.request.Request(
                    self.webhook_url, data=payload, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status in (200, 201, 204):
                        print(f"[Photon Push] Webhook notification delivered: {title}")
                        success = True
            except Exception as e:
                print(f"[Photon Push] Webhook delivery failed: {e}", file=sys.stderr)

        # 3. Google Drive / Photon Inbox Bridge Notification (Fallback / Audit Trail)
        if os.path.exists(self.gdrive_inbox):
            try:
                report_path = os.path.join(
                    self.gdrive_inbox, f"PHOTON_ALERT_{int(urllib.parse.time.time())}.txt"
                )
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(f"=== PHOTON MOBILE ALERT ===\nTitle: {title}\nPriority: {priority}\n\n{message}\n")
                print(f"[Photon Push] Google Drive Inbox Alert dropped to {report_path}")
                success = True
            except Exception as e:
                print(f"[Photon Push] GDrive Inbox drop failed: {e}", file=sys.stderr)

        if not success:
            print(f"[Photon Push Alert] {title}: {message}")

        return success


if __name__ == "__main__":
    notifier = PhotonNotifier()
    notifier.send_notification("HENRI V2 Photon Push Test", "Photon mobile alert system active and verified.")
