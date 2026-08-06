"""Audit whether PVSG can support an ownership recall experiment.

Ownership is defined per object entity as the person entity it contacts most often
across a video's observation window. It is a property of the object's identity, not of
any single frame, so it is a candidate target for index feedback: the query frame is
one where the owner is absent, which means the answer is provably not in the percept
and the only route to it runs through the object's identity embedding.

Four quantities decide whether the task is real, and this script measures all four:

1. Non-degeneracy: videos need at least two person entities, or the owner is the only
   candidate and a frequency baseline solves the task.
2. Ownable objects: object entities with enough person-contact frames to define an
   owner at all.
3. Unambiguity: the fraction of an object's contact frames belonging to its top
   person. Low concentration means the object has no owner, only handlers.
4. Query supply: observations where the object is visible and its owner is not. These
   come from the object manifests, not the pair manifests, because the pair view holds
   only simultaneously visible positive pairs -- precisely the frames this task must
   exclude.

Reported per source, because the egocentric sources behave differently from VidOR.

Usage:
    python -m experiments.pvsg.ownership_audit --manifest-root <snapshot>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Person-like PVSG source categories.
PERSON_CATEGORIES = frozenset({"adult", "baby", "child"})
# Predicates that plausibly indicate custody rather than incidental co-location.
CONTACT_PREDICATES = frozenset(
    {"holding", "carrying", "touching", "playing with", "caressing", "brushing"}
)
MINIMUM_CONTACT_FRAMES = 5
MINIMUM_CONCENTRATION = 0.6


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _identity_categories(ontology: dict[str, Any]) -> dict[str, str]:
    return {entry["name"]: entry["category"] for entry in ontology["identities"]}


def audit_ownership(manifest_root: Path, *, role: str = "train") -> dict[str, Any]:
    ontology = json.loads(
        (manifest_root / "ontology.json").read_text(encoding="utf-8")
    )
    category_of = _identity_categories(ontology)
    pairs = [
        record
        for record in _read_jsonl(manifest_root / "blocked" / "train_pairs.jsonl")
        if record["experiment_split"] == role
    ]
    observations = [
        record
        for record in _read_jsonl(manifest_root / "blocked" / "train_objects.jsonl")
        if record["experiment_split"] == role
    ]

    persons_by_video: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in observations:
        video = (record["source"], record["video_id"])
        if category_of.get(record["identity"]) in PERSON_CATEGORIES:
            persons_by_video[video].add(record["identity"])

    # Contact frames per (object identity, person identity), either direction.
    contacts: dict[str, Counter] = defaultdict(Counter)
    video_of: dict[str, tuple[str, str]] = {}
    for record in pairs:
        if not CONTACT_PREDICATES.intersection(record["predicates"]):
            continue
        subject, object_ = record["subject_identity"], record["object_identity"]
        subject_person = category_of.get(subject) in PERSON_CATEGORIES
        object_person = category_of.get(object_) in PERSON_CATEGORIES
        if subject_person == object_person:
            continue  # person-person or thing-thing: not a custody relation
        person, thing = (subject, object_) if subject_person else (object_, subject)
        contacts[thing][person] += 1
        video_of[thing] = (record["source"], record["video_id"])

    owners: dict[str, str] = {}
    concentrations: list[float] = []
    for thing, counts in contacts.items():
        total = sum(counts.values())
        person, top = counts.most_common(1)[0]
        concentration = top / total
        concentrations.append(concentration)
        video = video_of[thing]
        if (
            total >= MINIMUM_CONTACT_FRAMES
            and concentration >= MINIMUM_CONCENTRATION
            and len(persons_by_video[video]) >= 2
        ):
            owners[thing] = person

    # Query supply: object visible, owner not visible in the same frame.
    visible: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for record in observations:
        key = (record["source"], record["video_id"], record["frame_index"])
        visible[key].add(record["identity"])
    queries = Counter()
    owner_present = Counter()
    for record in observations:
        thing = record["identity"]
        owner = owners.get(thing)
        if owner is None:
            continue
        key = (record["source"], record["video_id"], record["frame_index"])
        if owner in visible[key]:
            owner_present[record["source"]] += 1
        else:
            queries[record["source"]] += 1

    by_source: dict[str, dict[str, Any]] = {}
    for source in sorted({video[0] for video in persons_by_video}):
        videos = [v for v in persons_by_video if v[0] == source]
        multi = [v for v in videos if len(persons_by_video[v]) >= 2]
        source_owned = [t for t in owners if video_of[t][0] == source]
        by_source[source] = {
            "videos": len(videos),
            "videos_with_two_or_more_persons": len(multi),
            "person_entities": sum(len(persons_by_video[v]) for v in videos),
            "owned_objects": len(source_owned),
            "query_observations_owner_absent": queries[source],
            "observations_owner_present": owner_present[source],
            "mean_candidate_owners": (
                sum(len(persons_by_video[video_of[t]]) for t in source_owned)
                / max(len(source_owned), 1)
            ),
        }

    concentrations.sort()
    return {
        "contact_objects": len(contacts),
        "owned_objects": len(owners),
        "query_observations_owner_absent": sum(queries.values()),
        "observations_owner_present": sum(owner_present.values()),
        "concentration_median": (
            concentrations[len(concentrations) // 2] if concentrations else 0.0
        ),
        "concentration_at_least_threshold": sum(
            1 for value in concentrations if value >= MINIMUM_CONCENTRATION
        ),
        "thresholds": {
            "minimum_contact_frames": MINIMUM_CONTACT_FRAMES,
            "minimum_concentration": MINIMUM_CONCENTRATION,
            "contact_predicates": sorted(CONTACT_PREDICATES),
        },
        "by_source": by_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--role", default="train")
    arguments = parser.parse_args()
    print(json.dumps(audit_ownership(arguments.manifest_root, role=arguments.role), indent=2))


if __name__ == "__main__":
    main()
