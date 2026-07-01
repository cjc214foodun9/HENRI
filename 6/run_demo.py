import json
import torch
from axtree_transducer import AXTreeTransducer
from accessibility_bridge import HenriAccessibilityBridge
from sdui_generator import HenriSDUIGenerator

def run_accessibility_flow_demo():
    print("=====================================================================")
    print("            PROJECT HENRI ACCESSIBILITY BRIDGE SYSTEM DEMO            ")
    print("=====================================================================\n")

    # 1. Initialize system components
    transducer = AXTreeTransducer()
    bridge = HenriAccessibilityBridge()
    sdui = HenriSDUIGenerator()

    # 2. Define a raw screen layout containing typical WCAG violations
    # - Node 1: An image with no alternative text (SC 1.1.1 violation)
    # - Node 2: An input with no visible label name or labeled_by association (SC 3.3.2 violation)
    # - Node 3: An error alert containing raw system traceback info (SC 4.1.3 target)
    raw_layout = {
        "title": "Industrial Monitoring Console",
        "nodes": [
            {
                "id": "status_indicator_icon",
                "role": "image",
                "name": "Warning status lights",
                "value": "",
                "focus_state": False,
                "wcag_metadata": {
                    "labeled_by": "",
                    "described_by": "",
                    "required": False,
                    "invalid": False,
                    "alt_text": "" # Violation!
                }
            },
            {
                "id": "pressure_cutoff_val",
                "role": "input",
                "name": "", # Violation!
                "value": "120",
                "focus_state": True,
                "wcag_metadata": {
                    "labeled_by": "", # Violation!
                    "described_by": "",
                    "required": True,
                    "invalid": True,
                    "alt_text": ""
                }
            },
            {
                "id": "system_alert_log",
                "role": "alert",
                "name": "",
                "value": "Traceback (most recent call last):\n  File \"zone_b.py\", line 14\nValueError: Pressure out of bounds",
                "focus_state": False,
                "wcag_metadata": {
                    "labeled_by": "",
                    "described_by": "",
                    "required": False,
                    "invalid": False,
                    "alt_text": ""
                }
            }
        ]
    }
    
    axtree_json = json.dumps(raw_layout, indent=2)
    print("--- [STEP 1] Ingesting Raw UI Screen AXTree Layout ---")
    print(axtree_json)
    print("\n---------------------------------------------------------------------")

    # 3. Transduce the layout into the 4096-dimensional complex phasor space (S^4095)
    print("--- [STEP 2] Transducing UI Layout into phasor space (S^4095) ---")
    v_screen = transducer.transduce_tree(json.dumps(raw_layout))
    print(f" -> Resulting complex phasor shape: {v_screen.shape}")
    print(f" -> L2 Norm: {torch.norm(v_screen).item():.4f} (Verified on unit hypersphere)")
    print("\n---------------------------------------------------------------------")

    # 4. Check for WCAG compliance
    print("--- [STEP 3] Running WCAG Compliance Scanner ---")
    report = bridge.check_wcag_compliance(json.dumps(raw_layout))
    print(f" -> Compliant: {report['is_compliant']}")
    print(f" -> SC 1.1.1 violations detected: {len(report['sc_1_1_1_violations'])}")
    for v in report['sc_1_1_1_violations']:
        print(f"    * Node '{v['id']}': {v['msg']}")
    print(f" -> SC 3.3.2 violations detected: {len(report['sc_3_3_2_violations'])}")
    for v in report['sc_3_3_2_violations']:
        print(f"    * Node '{v['id']}': {v['msg']}")
    print("\n---------------------------------------------------------------------")

    # 5. Automatically repair the UI layout representation
    print("--- [STEP 4] Dynamic Accessibility Repair & Auto-Labeling ---")
    repaired_json = bridge.auto_repair_axtree(json.dumps(raw_layout))
    repaired_data = json.loads(repaired_json)
    print(f" -> Injected Alt Text: '{repaired_data['nodes'][0]['wcag_metadata']['alt_text']}'")
    print(f" -> Injected Input Label: '{repaired_data['nodes'][1]['name']}'")
    
    # 6. Simplify alert for Speech synthesizers (SC 4.1.3 status message)
    announcement = bridge.generate_speech_announcement(repaired_json)
    print(f" -> Screen Reader Audio Announcement: '{announcement}'")
    print("\n---------------------------------------------------------------------")

    # 7. Generate Server-Driven UI (SDUI) compliant HTML code
    print("--- [STEP 5] Compiling Compliant Server-Driven UI elements ---")
    form_fields = [
        {
            "id": "pressure_cutoff_val",
            "type": "number",
            "label": "Emergency Pressure Cut-off Limit (kPa)",
            "required": True,
            "helper_text": "Default threshold is 100 kPa.",
            "invalid": True
        }
    ]
    compiled_html = sdui.compile_form("emergency_cutoff_form", form_fields)
    print("Compiled HTML Output:")
    print(compiled_html)
    print("=====================================================================")

if __name__ == "__main__":
    run_accessibility_flow_demo()
