import http.server
import json
import threading
import torch

class ThreadSafeTelemetryState:
    """
    Synchronized thread-safe register holding the active telemetry values of Project HENRI.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            "active_tiles": [True] * 16,
            "coupling": 1.0,
            "veto_intensity": 0.0,
            "langevin_heat": 0.0,
            "status": "CONVERGED",
            "error_energy": 0.0,
            "lora_scale": 1.0000,
            "phase_data": [[0.0] * 64 for _ in range(64)],
            "intensity_data": [[1.0] * 64 for _ in range(64)]
        }

    def update(self, **kwargs):
        """
        Updates the register state. Detaches and converts PyTorch tensors safely.
        """
        with self.lock:
            for key, val in kwargs.items():
                if key in self.data:
                    if isinstance(val, torch.Tensor):
                        # Safely extract PyTorch tensor values to Python lists to avoid thread race conditions
                        detached_val = val.detach().clone().cpu()
                        if detached_val.ndim == 2:
                            self.data[key] = detached_val.tolist()
                        else:
                            self.data[key] = detached_val.item()
                    else:
                        self.data[key] = val

    def get_snapshot(self):
        """
        Returns a copy of the telemetry state.
        """
        with self.lock:
            return json.dumps(self.data)


# Global singleton instance for access across thread boundaries
telemetry_register = ThreadSafeTelemetryState()


class TelemetryHTTPHandler(http.server.BaseHTTPRequestHandler):
    """
    Custom HTTP request handler serving JSON telemetry payloads with explicit CORS protection.
    """
    def do_OPTIONS(self):
        """
        Handles pre-flight checks from browsers to bypass CORS security policies.
        """
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/telemetry':
            # Retrieve thread-safe serialized snapshot
            json_response = telemetry_register.get_snapshot()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*') # Essential CORS header
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
            self.end_headers()
            self.wfile.write(json_response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence default standard logging to prevent console pollution during 100ms polling loops
        pass


class NonBlockingTelemetryServer:
    """
    Spawns and manages a non-blocking background daemon thread for the telemetry HTTP server.
    """
    def __init__(self, host='0.0.0.0', port=8000):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None

    def start(self):
        self.server = http.server.HTTPServer((self.host, self.port), TelemetryHTTPHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"[Telemetry] Non-blocking server listening on http://{self.host}:{self.port}/telemetry")

    def stop(self):
        if self.server:
            # Instructs the background loop to exit gracefully
            self.server.shutdown()
            self.server.server_close()
            print("[Telemetry] Server successfully terminated.")
