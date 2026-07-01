import json
import torch
from axtree_transducer import AXTreeTransducer
from accessibility_bridge import HenriAccessibilityBridge
from sdui_generator import HenriSDUIGenerator

def test_axtree_schema_and_transducer():
    print("[TEST 1] Testing AXTree-to-Phasor Transducer...")
    transducer = AXTreeTransducer()
    
    # 1. Create a sample AXTree JSON representing a Login Form
    login_tree = {
        "title": "Login Screen",
        "nodes": [
            {
                "id": "username_input",
                "role": "input",
                "name": "",
                "value": "user123",
                "focus_state": True,
                "wcag_metadata": {
                    "labeled_by": "",
                    "described_by": "",
                    "required": True,
                    "invalid": False,
                    "alt_text": ""
                }
            },
            {
                "id": "submit_btn",
                "role": "button",
                "name": "Submit",
                "value": "",
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
    
    # 2. Create another tree representing a Settings Menu
    settings_tree = {
        "title": "Settings Menu",
        "nodes": [
            {
                "id": "toggle_notifications",
                "role": "button",
                "name": "Enable Alerts",
                "value": "on",
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
    
    # Transduce both layouts
    v_login = transducer.transduce_tree(json.dumps(login_tree))
    v_settings = transducer.transduce_tree(json.dumps(settings_tree))
    
    # Verify shape
    assert v_login.shape == (4096,), "Login vector shape is not 4096!"
    assert v_settings.shape == (4096,), "Settings vector shape is not 4096!"
    
    # Verify that the two distinct layouts map to different phasors (low similarity)
    cos_sim = torch.abs(torch.dot(torch.conj(v_login), v_settings)).item()
    print(f" -> Cross-layout similarity (target near 0): {cos_sim:.6f}")
    assert cos_sim < 0.2, "Distinct screen layouts produced overlapping vector states!"
    print(" -> [PASS] AXTree transducer vector separation verified.")

def test_accessibility_bridge():
    print("\n[TEST 2] Testing Accessibility Bridge Daemon...")
    bridge = HenriAccessibilityBridge()
    
    # Create layout with critical WCAG violations
    uncompliant_tree = {
        "title": "Profile Dashboard",
        "nodes": [
            {
                "id": "avatar_img",
                "role": "image",
                "name": "User profile photo",
                "value": "",
                "focus_state": False,
                "wcag_metadata": {
                    "labeled_by": "",
                    "described_by": "",
                    "required": False,
                    "invalid": False,
                    "alt_text": "" # Missing alt text! (SC 1.1.1)
                }
            },
            {
                "id": "email_input",
                "role": "input",
                "name": "", # Missing visible label name!
                "value": "",
                "focus_state": False,
                "wcag_metadata": {
                    "labeled_by": "", # Missing labeled_by! (SC 3.3.2)
                    "described_by": "",
                    "required": True,
                    "invalid": False,
                    "alt_text": ""
                }
            },
            {
                "id": "error_alert",
                "role": "alert",
                "name": "",
                "value": "Traceback (most recent call last):\n  File \"main.py\", line 12\nTypeError: Cannot read property 'map' of null",
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
    
    axtree_str = json.dumps(uncompliant_tree)
    
    # 1. Run compliance check
    report = bridge.check_wcag_compliance(axtree_str)
    assert not report["is_compliant"], "Failed to detect accessibility violations!"
    assert len(report["sc_1_1_1_violations"]) == 1, "Failed to catch SC 1.1.1 (missing alt text)!"
    assert len(report["sc_3_3_2_violations"]) == 1, "Failed to catch SC 3.3.2 (unlabelled input)!"
    print(" -> [PASS] WCAG violation scanner successfully detected all targets.")
    
    # 2. Run auto repair
    repaired_json = bridge.auto_repair_axtree(axtree_str)
    repaired_report = bridge.check_wcag_compliance(repaired_json)
    assert repaired_report["is_compliant"], "Failed to correct accessibility violations!"
    
    # Verify alt text was generated
    repaired_data = json.loads(repaired_json)
    assert "AI-Generated" in repaired_data["nodes"][0]["wcag_metadata"]["alt_text"]
    print(" -> [PASS] AXTree auto-repair daemon resolved violations (SC 1.1.1 & SC 3.3.2).")
    
    # 3. Test dynamic status audio-dispatcher (SC 4.1.3)
    announcement = bridge.generate_speech_announcement(repaired_json)
    print(f" -> Speech Output: '{announcement}'")
    assert "System warning" in announcement, "Failed to simplify traceback error into clean status message!"
    print(" -> [PASS] SC 4.1.3 status message speech adapter verified.")

def test_sdui_generator():
    print("\n[TEST 3] Testing Server-Driven UI (SDUI) Generator...")
    generator = HenriSDUIGenerator()
    
    # 1. Test Form compilation
    fields = [
        {
            "id": "user_phone",
            "type": "tel",
            "label": "Phone Number",
            "required": True,
            "helper_text": "Include area code.",
            "invalid": True
        }
    ]
    form_html = generator.compile_form("contact_form", fields)
    assert 'label for="user_phone"' in form_html, "Form label relation missing!"
    assert 'input id="user_phone"' in form_html, "Form input ID missing!"
    assert 'aria-required="true"' in form_html, "aria-required attribute missing!"
    assert 'aria-describedby="user_phone_helper"' in form_html, "aria-describedby helper link missing!"
    assert 'aria-invalid="true"' in form_html, "aria-invalid error state missing!"
    print(" -> [PASS] SDUI Form compiler compliant with WCAG 3.3.2/1.3.1.")
    
    # 2. Test Table compilation
    headers = ["ID", "Score"]
    rows = [
        [1, 98],
        [2, ""] # Empty cell check
    ]
    table_html = generator.compile_table("grades_table", "Student Scores", headers, rows)
    assert 'role="region"' in table_html, "Table region wrapper missing!"
    assert 'tabindex="0"' in table_html, "Table tabindex scroll-focus missing!"
    assert '<caption id="grades_table_caption">Student Scores</caption>' in table_html, "Table caption tag missing!"
    assert '<th scope="col">ID</th>' in table_html, "Table col header scope missing!"
    assert '<th scope="row">1</th>' in table_html, "Table row header scope missing!"
    assert 'sr-only' in table_html, "Empty cell screen-reader context fallback missing!"
    print(" -> [PASS] SDUI Table compiler compliant with WCAG 1.3.1.")
    
    # 3. Test Card compilation
    card_html = generator.compile_card("card_item", "View Details", "Go", "/details", "/img.png", "Detailed schematic")
    assert 'tabindex="-1"' in card_html, "Card duplicate image link focus override missing!"
    assert '<img src="/img.png" alt="Detailed schematic">' in card_html, "Card image alt descriptor missing!"
    assert '<a href="/details">View Details</a>' in card_html, "Card focal action area title link missing!"
    print(" -> [PASS] SDUI Card compiler focus order and contrast verified.")

if __name__ == "__main__":
    print("=====================================================================")
    print("         BOOTING AI-DRIVEN ACCESSIBILITY BRIDGE TEST SUITE          ")
    print("=====================================================================\n")
    test_axtree_schema_and_transducer()
    test_accessibility_bridge()
    test_sdui_generator()
    print("\n[SUCCESS] Accessibility Bridge tests passed successfully.")
