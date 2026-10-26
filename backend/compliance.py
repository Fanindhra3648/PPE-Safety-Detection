"""Convert PPE detections into a simple, explainable compliance summary."""
from collections.abc import Iterable
from backend.config import CLASS_NAMES

HELMET_CLASSES = {"Helmet", "NoHelmet"}
VEST_CLASSES = {"Vest", "NoVest"}

def summarize_compliance(detections: Iterable[dict]) -> dict:
    names = {str(item.get("class_name", CLASS_NAMES.get(item.get("class_id"), "Unknown")))
             for item in detections}
    helmet = "Helmet" in names
    no_helmet = "NoHelmet" in names
    vest = "Vest" in names
    no_vest = "NoVest" in names

    violations = []
    if no_helmet:
        violations.append("No helmet detected")
    if no_vest:
        violations.append("No safety vest detected")

    if violations:
        status = "VIOLATION"
        message = "Potential PPE violation detected. Please review the image."
    elif helmet or vest:
        status = "REVIEW"
        message = "Some PPE items were detected. Confirm the full scene before marking compliant."
    else:
        status = "UNKNOWN"
        message = "No recognized PPE classes were detected."

    return {
        "status": status,
        "helmet_detected": helmet,
        "no_helmet_detected": no_helmet,
        "vest_detected": vest,
        "no_vest_detected": no_vest,
        "violations": violations,
        "message": message,
    }
