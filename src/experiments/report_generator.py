import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(
    "data/evaluation/ab_tests/prompt_ab_results.json"
)

REPORT = Path(
    "data/evaluation/ab_tests/prompt_ab_report.md"
)

CHART = Path(
    "data/evaluation/ab_tests/prompt_ab_chart.png"
)


def load_results():

    with open(RESULTS, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_chart(results):

    v1 = [
        item["prompt_v1"]["similarity"]
        for item in results
    ]

    v2 = [
        item["prompt_v2"]["similarity"]
        for item in results
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(v1) + 1),
        v1,
        marker="o",
        label="Prompt V1",
    )

    plt.plot(
        range(1, len(v2) + 1),
        v2,
        marker="o",
        label="Prompt V2",
    )

    plt.xlabel("Evaluation Question")
    plt.ylabel("Semantic Similarity")

    plt.title("Prompt A/B Test")

    plt.legend()

    plt.tight_layout()

    plt.savefig(CHART)

    plt.close()


def generate_report(results):

    avg_v1_similarity = sum(
        r["prompt_v1"]["similarity"]
        for r in results
    ) / len(results)

    avg_v2_similarity = sum(
        r["prompt_v2"]["similarity"]
        for r in results
    ) / len(results)

    avg_v1_latency = sum(
        r["prompt_v1"]["latency"]
        for r in results
    ) / len(results)

    avg_v2_latency = sum(
        r["prompt_v2"]["latency"]
        for r in results
    ) / len(results)

    winner = (
        "Prompt V2"
        if avg_v2_similarity >= avg_v1_similarity
        else "Prompt V1"
    )

    report = f"""# Prompt A/B Test Report

## Summary

| Metric | Prompt V1 | Prompt V2 |
|--------|-----------:|----------:|
| Average Semantic Similarity | {avg_v1_similarity:.3f} | {avg_v2_similarity:.3f} |
| Average Latency (sec) | {avg_v1_latency:.2f} | {avg_v2_latency:.2f} |

## Winner

**{winner}**

Prompt selection based on average semantic similarity.

See:

- prompt_ab_chart.png
- prompt_ab_results.json
"""

    REPORT.write_text(
        report,
        encoding="utf-8",
    )


def main():

    results = load_results()

    generate_chart(results)

    generate_report(results)

    print()

    print("Generated")

    print(REPORT)

    print(CHART)


if __name__ == "__main__":
    main()