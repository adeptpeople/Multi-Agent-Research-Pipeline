from __future__ import annotations

import re
from typing import Optional

from ..config import NUMERIC_CONFLICT_THRESHOLD
from ..schemas.finding import ContestedFinding, FindingSchema


def _extract_percentage(text: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        return float(match.group(1))
    return None


def _extract_number(text: str) -> Optional[float]:
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match:
        return float(match.group(1))
    return None


def _simple_topic_key(finding: FindingSchema) -> str:
    """Produce a rough topic key from the claim for grouping."""
    claim = finding.claim.lower()
    # Strip common filler words and keep content words
    stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "at", "to", "and", "or"}
    words = [w for w in re.findall(r"\w+", claim) if w not in stop]
    # Use first 4 content words as topic key
    return " ".join(words[:4])


def _numeric_conflict(a: FindingSchema, b: FindingSchema) -> bool:
    """Return True if both claims contain numeric values that differ beyond the threshold."""
    pct_a = _extract_percentage(a.claim)
    pct_b = _extract_percentage(b.claim)
    if pct_a is not None and pct_b is not None:
        base = max(pct_a, pct_b)
        if base > 0 and abs(pct_a - pct_b) / base > NUMERIC_CONFLICT_THRESHOLD:
            return True

    num_a = _extract_number(a.claim)
    num_b = _extract_number(b.claim)
    if num_a is not None and num_b is not None and num_a != num_b:
        base = max(num_a, num_b)
        if base > 0 and abs(num_a - num_b) / base > NUMERIC_CONFLICT_THRESHOLD:
            return True
    return False


def _qualitative_conflict(a: FindingSchema, b: FindingSchema) -> bool:
    """Detect simple qualitative contradictions (increase vs decrease, etc.)."""
    opposites = [
        ({"increase", "grew", "rose", "higher", "up"}, {"decrease", "fell", "dropped", "lower", "down"}),
        ({"approved", "allowed", "permitted"}, {"banned", "prohibited", "restricted", "denied"}),
        ({"positive", "growth", "expansion"}, {"negative", "contraction", "decline"}),
    ]
    claim_a = set(a.claim.lower().split())
    claim_b = set(b.claim.lower().split())
    for set1, set2 in opposites:
        if (claim_a & set1 and claim_b & set2) or (claim_a & set2 and claim_b & set1):
            return True
    return False


class ConflictDetector:
    def detect(
        self, findings: list[FindingSchema]
    ) -> tuple[list[FindingSchema], list[ContestedFinding]]:
        """
        Partition findings into (well_established, contested).
        Contested findings are preserved in full — values are NEVER silently averaged.
        """
        groups: dict[str, list[FindingSchema]] = {}
        for f in findings:
            key = _simple_topic_key(f)
            groups.setdefault(key, []).append(f)

        well_established: list[FindingSchema] = []
        contested: list[ContestedFinding] = []
        contested_ids: set[str] = set()

        for topic, group in groups.items():
            if len(group) == 1:
                well_established.append(group[0])
                continue

            conflict_pairs: list[tuple[FindingSchema, FindingSchema]] = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if _numeric_conflict(group[i], group[j]) or _qualitative_conflict(
                        group[i], group[j]
                    ):
                        conflict_pairs.append((group[i], group[j]))

            if conflict_pairs:
                involved = {f.finding_id for pair in conflict_pairs for f in pair}
                contested_ids.update(involved)
                contested.append(
                    ContestedFinding(
                        topic=topic,
                        sources=[f for f in group if f.finding_id in involved],
                        explanation=(
                            "Sources report conflicting values. "
                            "Possible causes: different sampling periods, "
                            "methodologies, or geographic scope."
                        ),
                    )
                )
                for f in group:
                    if f.finding_id not in involved:
                        well_established.append(f)
            else:
                well_established.extend(group)

        return well_established, contested
