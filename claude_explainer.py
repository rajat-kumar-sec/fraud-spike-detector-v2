import os
import json

# ─────────────────────────────────────────────────────────────
# CLAUDE API EXPLAINER
# Supports two modes:
#   - "mock": generates rule-based explanations (no API needed)
#   - "live": calls real Claude API (needs ANTHROPIC_API_KEY env var)
# ─────────────────────────────────────────────────────────────

MODE = "mock" if not os.environ.get("ANTHROPIC_API_KEY") else "live"


def explain_transaction(txn):
    """
    Generate a human-friendly explanation for why a transaction
    was flagged as suspicious.

    Args:
        txn: dict with keys like transaction_id, amount, timestamp,
             card_id, device_id, geo_location, ml_probability,
             rule_burst, rule_geo_mismatch, rule_odd_hour, etc.

    Returns:
        str: explanation in plain English
    """
    if MODE == "live":
        return _explain_with_claude(txn)
    else:
        return _explain_with_rules(txn)


def _explain_with_claude(txn):
    """Call real Claude API for explanation."""
    try:
        import anthropic

        client = anthropic.Anthropic()

        prompt = f"""You are a fraud analyst assistant for a payment company.
A transaction has been flagged as potentially fraudulent. Provide a clear,
concise explanation (2-3 sentences max) of why this transaction is suspicious
and what the merchant/risk team should do.

Transaction details:
- ID: {txn.get('transaction_id')}
- Amount: INR {txn.get('amount', 0):.2f}
- Time: {txn.get('timestamp')}
- Card: {txn.get('card_id')}
- Device: {txn.get('device_id')}
- Location: {txn.get('geo_location')}
- Merchant: {txn.get('merchant_id')}
- ML confidence: {txn.get('ml_probability', 0)*100:.1f}%
- Rules triggered: {', '.join(_get_triggered_rules(txn)) or 'none'}

Respond in plain English, no bullet points, no markdown."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    except Exception as e:
        return f"[Claude API error: {e}] Falling back to rule-based explanation."


def _explain_with_rules(txn):
    """Generate explanation based on which rules fired — no API needed."""
    reasons = []
    amount = txn.get("amount", 0)
    probability = txn.get("ml_probability", 0)

    # Rule-based reasons
    if txn.get("rule_burst"):
        reasons.append(
            f"Rapid burst detected: this card/device made multiple transactions "
            f"in a very short time window, which is a classic sign of card testing "
            f"or automated fraud."
        )

    if txn.get("rule_geo_mismatch"):
        reasons.append(
            f"Impossible travel: the same card was used in geographically distant "
            f"locations within minutes, suggesting the card details are cloned "
            f"or stolen."
        )

    if txn.get("rule_odd_hour"):
        reasons.append(
            f"Unusual timing: a high-value transaction (INR {amount:,.0f}) occurred "
            f"during odd hours (1-5 AM) when legitimate cardholder activity is "
            f"typically minimal."
        )

    # ML-based context
    if probability > 0.9:
        reasons.append(
            f"The ML model is {probability*100:.0f}% confident this is fraud — "
            f"this matches known fraud patterns in our historical data."
        )
    elif probability > 0.7:
        reasons.append(
            f"The ML model flags this with {probability*100:.0f}% confidence. "
            f"While not certain, it shares characteristics with confirmed fraud cases."
        )

    if not reasons:
        reasons.append(
            f"This transaction was flagged with {probability*100:.0f}% ML confidence "
            f"based on spending patterns that deviate from the cardholder's norm."
        )

    return " ".join(reasons)


def _get_triggered_rules(txn):
    """Helper to list which rules fired."""
    rules = []
    if txn.get("rule_burst"):
        rules.append("Rapid Burst")
    if txn.get("rule_geo_mismatch"):
        rules.append("Geo-Mismatch")
    if txn.get("rule_odd_hour"):
        rules.append("Odd-Hour High-Value")
    return rules


def explain_batch(transactions):
    """Explain multiple transactions. Returns list of (id, explanation)."""
    return [(t.get("transaction_id"), explain_transaction(t)) for t in transactions]
