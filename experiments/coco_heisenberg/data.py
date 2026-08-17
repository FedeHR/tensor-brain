"""Build the COCO corpus for the learned-index-layer experiment.

Each image contributes one record:

    x        which of the 12 COCO supercategories are present, from the
             *instance* annotations (exhaustive by protocol) -- the latent
    symbols  content words drawn from the five human *captions* -- the evidence

The two channels are independent annotation passes over the same image, which is
what makes ``x`` a ground truth rather than a relabelling of the evidence. A
caption word is not a deterministic function of the supercategory vector, so the
index layer has something to learn; that is the degeneracy trap this design is
built to avoid.

Nothing here needs the images, a GPU, or any NLP dependency.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Words that carry no evidence about which categories of thing are present.
STOPWORDS = frozenset("""
a an the this that these those there here and or but if then than so as of in on at to for
with without from into onto over under above below near next by is are was were be been being
am it its it's he she they them his her their we you i my your our some any all both each few
more most other another such no nor not only own same very can will just don should now
up down out off again further once who whom which what when where why how while during before
after between against about through above across behind beside beyond within along around
has have had having do does did doing would could may might must shall
one two three four five six seven eight nine ten many several
""".split())

# Suffixes stripped only when the shortened form is itself a frequent token.
_TOKEN = re.compile(r"[a-z]+")

SUPERCATEGORY_ORDER = [
    "person", "vehicle", "outdoor", "animal", "accessory", "sports",
    "kitchen", "food", "furniture", "electronic", "appliance", "indoor",
]


@dataclass(frozen=True)
class Corpus:
    """A materialized corpus: presence bits, symbol sets, and the vocabulary."""

    presence: np.ndarray          # [images, categories] uint8
    symbols: list[list[int]]      # per image, distinct symbol ids
    vocabulary: list[str]         # symbol id -> word
    categories: list[str]         # bit index -> supercategory name

    @property
    def num_images(self) -> int:
        return int(self.presence.shape[0])

    @property
    def num_categories(self) -> int:
        return int(self.presence.shape[1])

    @property
    def num_symbols(self) -> int:
        return len(self.vocabulary)

    def subset(self, index: np.ndarray) -> Corpus:
        return Corpus(
            presence=self.presence[index],
            symbols=[self.symbols[int(i)] for i in index],
            vocabulary=self.vocabulary,
            categories=self.categories,
        )


def _tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS]


def _singularize(counts: Counter[str]) -> dict[str, str]:
    """Fold a plural onto its singular when the singular is at least as common.

    A deliberately crude stand-in for a lemmatizer: it removes the one source of
    duplicate symbols that matters here (``dog``/``dogs``) without adding a
    dependency or a model.
    """

    mapping: dict[str, str] = {}
    for word in counts:
        if word.endswith("ss") or len(word) <= 3:
            continue
        stem = word[:-2] if word.endswith("es") and len(word) > 4 else None
        if word.endswith("s") and not word.endswith("es"):
            stem = word[:-1]
        if stem and counts.get(stem, 0) >= counts[word]:
            mapping[word] = stem
    return mapping


def _presence_from_instances(instances_path: Path) -> tuple[dict[int, set[str]], list[str]]:
    """Map every image id to the set of supercategories its instances cover."""

    with instances_path.open() as handle:
        payload = json.load(handle)

    supercategory = {c["id"]: c["supercategory"] for c in payload["categories"]}
    present: dict[int, set[str]] = {}
    for annotation in payload["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        bucket = present.setdefault(annotation["image_id"], set())
        bucket.add(supercategory[annotation["category_id"]])

    names = sorted(set(supercategory.values()))
    ordered = [c for c in SUPERCATEGORY_ORDER if c in names]
    ordered += [c for c in names if c not in ordered]
    return present, ordered


def build_corpus(
    annotations_root: Path,
    *,
    split: str = "train2017",
    vocabulary_size: int = 1000,
    min_symbols: int = 2,
) -> Corpus:
    """Materialize the corpus for one COCO split."""

    root = Path(annotations_root)
    present, categories = _presence_from_instances(root / f"instances_{split}.json")

    with (root / f"captions_{split}.json").open() as handle:
        caption_payload = json.load(handle)

    per_image: dict[int, list[str]] = {}
    for record in caption_payload["annotations"]:
        per_image.setdefault(record["image_id"], []).extend(_tokenize(record["caption"]))

    raw_counts = Counter(word for words in per_image.values() for word in words)
    folding = _singularize(raw_counts)
    counts = Counter()
    for word, total in raw_counts.items():
        counts[folding.get(word, word)] += total

    vocabulary = [word for word, _ in counts.most_common(vocabulary_size)]
    index_of = {word: i for i, word in enumerate(vocabulary)}
    category_of = {name: i for i, name in enumerate(categories)}

    presence_rows: list[np.ndarray] = []
    symbol_rows: list[list[int]] = []
    for image_id, words in per_image.items():
        found = present.get(image_id)
        if not found:
            continue  # no annotated instances: the latent is undefined, not empty
        ids = sorted({index_of[folding.get(w, w)] for w in words if folding.get(w, w) in index_of})
        if len(ids) < min_symbols:
            continue
        row = np.zeros(len(categories), dtype=np.uint8)
        for name in found:
            row[category_of[name]] = 1
        presence_rows.append(row)
        symbol_rows.append(ids)

    return Corpus(
        presence=np.stack(presence_rows),
        symbols=symbol_rows,
        vocabulary=vocabulary,
        categories=categories,
    )


def save(corpus: Corpus, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.concatenate([np.asarray(s, dtype=np.int32) for s in corpus.symbols])
    lengths = np.asarray([len(s) for s in corpus.symbols], dtype=np.int32)
    np.savez_compressed(
        path,
        presence=corpus.presence,
        symbols_flat=flat,
        symbols_lengths=lengths,
        vocabulary=np.asarray(corpus.vocabulary),
        categories=np.asarray(corpus.categories),
    )


def load(path: Path) -> Corpus:
    payload = np.load(Path(path), allow_pickle=False)
    bounds = np.concatenate([[0], np.cumsum(payload["symbols_lengths"])])
    flat = payload["symbols_flat"]
    symbols = [flat[bounds[i]:bounds[i + 1]].tolist() for i in range(len(bounds) - 1)]
    return Corpus(
        presence=payload["presence"],
        symbols=symbols,
        vocabulary=[str(w) for w in payload["vocabulary"]],
        categories=[str(c) for c in payload["categories"]],
    )


def split_indices(
    num_images: int, *, holdout: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic image-disjoint train/test split."""

    order = np.random.default_rng(seed).permutation(num_images)
    cut = int(round(num_images * (1.0 - holdout)))
    return np.sort(order[:cut]), np.sort(order[cut:])
