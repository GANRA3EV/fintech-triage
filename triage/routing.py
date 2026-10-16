def review_reasons(confidence, urgency_confidence, urgency, threshold):
    reasons = []
    if confidence < threshold:
        reasons.append("low_category_confidence")
    if urgency_confidence < threshold:
        reasons.append("low_urgency_confidence")
    if urgency == "critical":
        reasons.append("critical_urgency")
    return reasons
