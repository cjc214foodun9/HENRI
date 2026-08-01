"""Deterministic promotion checks for graph envelopes."""
from .evidence_receipts import EvidenceReceipt, ReceiptError

def validate_receipts(receipts):
    checked=[]
    for item in receipts:
        receipt=item if isinstance(item,EvidenceReceipt) else EvidenceReceipt(**item)
        checked.append(receipt.validate())
    return checked

def promotion_status(claim_status, receipts):
    try: checked=validate_receipts(receipts)
    except (TypeError,ValueError,ReceiptError): return "blocked"
    if claim_status == "verified" and checked and all(r.status == "pass" for r in checked): return "verified"
    return "unverified"
