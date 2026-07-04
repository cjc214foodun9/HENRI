import threading
import json
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

class TelemetryState:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            "active_tiles": [True] * 16,
            "coupling": 1.0,
            "veto_intensity": 0.0,
            "status": "HEALTHY",
            "error_energy": 0.0,
            "lora_scale": 1.0,
            "phase_data": None,
            "intensity_data": None
        }

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                # Convert PyTorch tensors or numpy arrays to lists for JSON serialization
                if hasattr(v, "tolist"):
                    v = v.tolist()
                elif isinstance(v, np.ndarray):
                    v = v.tolist()
                self.data[k] = v

    def get_json(self):
        with self.lock:
            out_data = dict(self.data)
            if out_data["phase_data"] is None:
                # Generate random 64x64 phase matrix
                out_data["phase_data"] = (np.random.rand(64, 64) * 2 * np.pi - np.pi).tolist()
            if out_data["intensity_data"] is None:
                out_data["intensity_data"] = np.random.rand(64, 64).tolist()
            return json.dumps(out_data)

telemetry_state = TelemetryState()

class TelemetryHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging request info to terminal to prevent spamming stdout during benchmark
        pass

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()

    def do_GET(self):
        if self.path == "/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(telemetry_state.get_json().encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

class NonBlockingTelemetryServer:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        def run_server():
            try:
                self.server = HTTPServer((self.host, self.port), TelemetryHTTPRequestHandler)
                print(f"[TELEMETRY SERVER] Listening on http://{self.host}:{self.port}/telemetry")
                self.server.serve_forever()
            except Exception as e:
                print(f"[TELEMETRY SERVER ERROR] Server crashed: {e}")

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("[TELEMETRY SERVER] Stopped.")

# Register object
class TelemetryRegister:
    def update(self, *args, **kwargs):
        telemetry_state.update(**kwargs)

telemetry_register = TelemetryRegister()
