import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher


RESULTS_PATH = Path(__file__).with_name("my_results.json")


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def capitalized_terms(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "in", "on", "at", "of", "and", "it", "this",
        "that", "he", "she", "they", "we", "i", "you", "one", "all",
        "during", "prior", "only", "just",
    }
    return {
        word for word in re.findall(r"[A-Z][A-Za-z'-]*", text)
        if word.lower() not in stop
    }


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def corr(xs: list[float], ys: list[float]) -> float:
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if not den_x or not den_y:
        return 0.0
    return num / math.sqrt(den_x * den_y)


def classify(sample: dict, similarity: float, overlap: float, length_ratio: float) -> list[str]:
    ref = sample["en_ref"]
    hyp = sample["en_hyp"]
    labels: list[str] = []

    ref_numbers = numbers(ref)
    hyp_numbers = numbers(hyp)
    if ref_numbers != hyp_numbers:
        labels.append("number mismatch")

    ref_caps = capitalized_terms(ref)
    hyp_caps = capitalized_terms(hyp)
    if ref_caps and len(ref_caps & hyp_caps) / len(ref_caps) < 0.5:
        labels.append("named entity drift")

    if length_ratio < 0.8:
        labels.append("compression / omission")
    elif length_ratio > 1.2:
        labels.append("expansion / paraphrase")

    if similarity < 0.55 and overlap < 0.45:
        labels.append("semantic drift")

    if not labels:
        labels.append("mostly faithful paraphrase")
    return labels


def main() -> None:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    samples = data["samples"]

    rows = []
    labels = Counter()
    id_counts = Counter(sample["id"] for sample in samples)
    for sample in samples:
        ref = sample["en_ref"]
        hyp = sample["en_hyp"]
        ref_tokens = tokens(ref)
        hyp_tokens = tokens(hyp)
        ref_set = set(ref_tokens)
        hyp_set = set(hyp_tokens)
        similarity = SequenceMatcher(None, ref.lower(), hyp.lower()).ratio()
        overlap = len(ref_set & hyp_set) / len(ref_set) if ref_set else 0.0
        length_ratio = len(hyp_tokens) / len(ref_tokens) if ref_tokens else 0.0
        sample_labels = classify(sample, similarity, overlap, length_ratio)
        labels.update(sample_labels)
        rows.append({
            "id": sample["id"],
            "ref": ref,
            "hyp": hyp,
            "similarity": similarity,
            "overlap": overlap,
            "length_ratio": length_ratio,
            "labels": sample_labels,
            "total_s": sample["total_s"],
            "audio_dur": sample["audio_dur"],
            "rtf": sample["rtf"],
            "utmos": sample["utmos"],
            "tts_share": sample["tts_s"] / sample["total_s"] if sample["total_s"] else 0.0,
        })

    rows.sort(key=lambda row: (row["similarity"], row["overlap"]))
    rows_by_duration = sorted(rows, key=lambda row: row["audio_dur"])
    quartile = len(rows_by_duration) // 4

    print("Aggregate")
    print("---------")
    print(f"samples: {len(rows)}")
    print(f"BLEU: {data['bleu']:.2f}")
    print(f"chrF: {data['chrf']:.2f}")
    print(f"avg UTMOS: {data['avg_utmos']:.3f}")
    print(f"latency mean/p95: {data['latency']['mean']:.3f}s / {data['latency']['p95']:.3f}s")
    print(f"RTF mean/p95: {data['rtf']['mean']:.3f}x / {data['rtf']['p95']:.3f}x")
    print(f"mean hypothesis/reference token ratio: {statistics.mean(r['length_ratio'] for r in rows):.3f}")
    print(f"mean TTS share of total latency: {statistics.mean(r['tts_share'] for r in rows):.3f}")
    print(f"duplicate sample ids: {sum(count > 1 for count in id_counts.values())}")
    print()
    print("Error buckets")
    print("-------------")
    for label, count in labels.most_common():
        print(f"{label}: {count}")
    print()
    print("Correlations")
    print("------------")
    print(f"similarity vs total latency: {corr([r['similarity'] for r in rows], [r['total_s'] for r in rows]):.3f}")
    print(f"similarity vs audio duration: {corr([r['similarity'] for r in rows], [r['audio_dur'] for r in rows]):.3f}")
    print(f"similarity vs UTMOS: {corr([r['similarity'] for r in rows], [r['utmos'] for r in rows]):.3f}")
    print()
    print("Duration quartiles")
    print("------------------")
    for index in range(4):
        start = index * quartile
        end = (index + 1) * quartile if index < 3 else len(rows_by_duration)
        bucket = rows_by_duration[start:end]
        print(
            f"Q{index + 1}: dur={bucket[0]['audio_dur']:.2f}-{bucket[-1]['audio_dur']:.2f}s "
            f"sim={statistics.mean(r['similarity'] for r in bucket):.3f} "
            f"total={statistics.mean(r['total_s'] for r in bucket):.3f}s"
        )
    print()
    print("Representative examples by bucket")
    print("---------------------------------")
    for label in [
        "number mismatch",
        "named entity drift",
        "compression / omission",
        "expansion / paraphrase",
        "semantic drift",
    ]:
        print(label)
        subset = [row for row in rows if label in row["labels"]]
        subset.sort(key=lambda row: (row["overlap"], row["similarity"]))
        for row in subset[:3]:
            print(f"  ID {row['id']} | sim={row['similarity']:.3f} overlap={row['overlap']:.3f} len={row['length_ratio']:.2f}")
            print(f"  REF: {row['ref']}")
            print(f"  HYP: {row['hyp']}")
        print()

    print("Worst examples")
    print("--------------")
    for row in rows[:15]:
        print(f"ID {row['id']} | sim={row['similarity']:.3f} overlap={row['overlap']:.3f} len={row['length_ratio']:.2f} | {', '.join(row['labels'])}")
        print(f"REF: {row['ref']}")
        print(f"HYP: {row['hyp']}")
        print()


if __name__ == "__main__":
    main()