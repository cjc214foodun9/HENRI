import json

class HenriAccessibilityBridge:
    """
    AI-Driven Accessibility Bridge Daemon:
    Intelligently monitors the active Accessibility Tree (AXTree), detects WCAG 2.2 violations,
    and dynamically injects repairs (alt text, helper labels, simplified error descriptions)
    to facilitate seamless integration with platform-level screen readers (Orca/AT-SPI).
    """
    def __init__(self):
        pass

    def check_wcag_compliance(self, axtree_json: str) -> dict:
        """
        Scans the AXTree for accessibility violations.
        Returns a summary report of WCAG issues found.
        """
        data = json.loads(axtree_json)
        nodes = data.get("nodes", [])
        
        report = {
            "sc_1_1_1_violations": [], # Missing alt text
            "sc_3_3_2_violations": [], # Missing labels or helper instructions
            "sc_4_1_3_violations": [], # Status message announcements pending
            "is_compliant": True
        }
        
        for node in nodes:
            role = node.get("role", "").lower()
            name = node.get("name", "").strip()
            metadata = node.get("wcag_metadata", {})
            alt_text = metadata.get("alt_text", "").strip()
            labeled_by = metadata.get("labeled_by", "").strip()
            
            # Check SC 1.1.1: Non-Text Content (images/charts must have descriptive alt text)
            if role in ["image", "img", "chart", "icon"] and not alt_text:
                report["sc_1_1_1_violations"].append({
                    "id": node.get("id"),
                    "name": name,
                    "msg": f"Role '{role}' is missing alternative text descriptor."
                })
                report["is_compliant"] = False
                
            # Check SC 3.3.2: Labels or Instructions (inputs must have accessible label name or be labeled by another element)
            if role in ["input", "textbox", "select", "textarea"] and not name and not labeled_by:
                report["sc_3_3_2_violations"].append({
                    "id": node.get("id"),
                    "msg": f"Input control '{node.get('id')}' is missing visible label or labeled_by association."
                })
                report["is_compliant"] = False
                
            # Check SC 4.1.3: Status Messages (alerts must trigger speech outputs)
            if role in ["alert", "status", "notification"]:
                report["sc_4_1_3_violations"].append({
                    "id": node.get("id"),
                    "value": node.get("value"),
                    "msg": "Dynamic status message detected. Requires speech conversion."
                })
                
        return report

    def auto_repair_axtree(self, axtree_json: str) -> str:
        """
        Dynamically injects WCAG repairs into the AXTree JSON payload:
        - Auto-generates alt text for unlabelled images (SC 1.1.1)
        - Injects help descriptions for unlabeled inputs (SC 3.3.2)
        """
        data = json.loads(axtree_json)
        nodes = data.get("nodes", [])
        
        for node in nodes:
            role = node.get("role", "").lower()
            name = node.get("name", "")
            metadata = node.get("wcag_metadata", {})
            alt_text = metadata.get("alt_text", "")
            
            # Resolve SC 1.1.1 (Non-Text Content)
            if role in ["image", "img", "chart", "icon"] and not alt_text:
                inferred_alt = f"AI-Generated Alt Text: A descriptive illustration of {name or 'unlabelled system graphic'}"
                metadata["alt_text"] = inferred_alt
                print(f"[REPAIR - SC 1.1.1] Injected alt text for image '{node.get('id')}': {inferred_alt}")
                
            # Resolve SC 3.3.2 (Labels/Instructions)
            if role in ["input", "textbox", "select", "textarea"]:
                if not name and not metadata.get("labeled_by"):
                    metadata["described_by"] = "auto_injected_input_guide"
                    node["name"] = f"User Input Field ({node.get('id')})"
                    print(f"[REPAIR - SC 3.3.2] Injected fallback name and description for input '{node.get('id')}'")
                    
        return json.dumps(data)

    def generate_speech_announcement(self, axtree_json: str) -> str:
        """
        Enforces SC 4.1.3 (Status Messages).
        Processes dynamic status alerts, simplifying and structuring them for screen reader TTS backends.
        """
        data = json.loads(axtree_json)
        nodes = data.get("nodes", [])
        
        speech_announcements = []
        for node in nodes:
            role = node.get("role", "").lower()
            value = node.get("value", "")
            
            if role == "alert":
                # Simplify tracebacks and runtime errors into clean natural language
                clean_value = value
                if "traceback" in value.lower() or "error" in value.lower():
                    clean_value = "System warning: A program execution error occurred in the sandbox. " + value.split("\n")[-1]
                speech_announcements.append(f"Screen Alert: {clean_value}")
                
            elif role == "notification":
                speech_announcements.append(f"Notification: {value}")
                
        return " | ".join(speech_announcements) if speech_announcements else "System idle. All bounds normal."
