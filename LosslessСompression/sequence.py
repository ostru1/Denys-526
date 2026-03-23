from __future__ import annotations

import argparse
import collections
import math
import random
import string
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SURNAME = "Чернецький"
DEFAULT_GROUP_NUMBER = "526-СТ"
DEFAULT_JOURNAL_NUMBER = 9
DEFAULT_SEQUENCE_LENGTH = 100
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate laboratory work sequences and calculate their characteristics."
        )
    )
    parser.add_argument("--surname", default=DEFAULT_SURNAME, help="Student surname.")
    parser.add_argument(
        "--group-number",
        default=DEFAULT_GROUP_NUMBER,
        help="Group number, digits only.",
    )
    parser.add_argument(
        "--journal-number",
        type=int,
        default=DEFAULT_JOURNAL_NUMBER,
        help="Student journal number.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Length of each generated sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible results.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory for generated files.",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    if args.length <= 0:
        raise ValueError("Sequence length must be positive.")
    if args.journal_number < 0 or args.journal_number > args.length:
        raise ValueError("Journal number must be in the range [0, sequence length].")
    if not args.surname:
        raise ValueError("Surname must not be empty.")
    if len(args.surname) < 2:
        raise ValueError("Surname must contain at least two characters.")
    if not extract_group_digits(args.group_number):
        raise ValueError("Group number must contain at least one digit.")
    if len(args.surname) > args.length:
        raise ValueError("Surname length must not exceed the sequence length.")


def extract_group_digits(group_number: str) -> str:
    return "".join(symbol for symbol in group_number if symbol.isdigit())


def build_equal_distribution(
    elements: list[str], total_length: int, rng: random.Random
) -> str:
    repeats, remainder = divmod(total_length, len(elements))
    result = elements * repeats + elements[:remainder]
    rng.shuffle(result)
    return "".join(result)


def build_sequence_1(
    journal_number: int, total_length: int, rng: random.Random
) -> str:
    sequence = ["1"] * journal_number + ["0"] * (total_length - journal_number)
    rng.shuffle(sequence)
    return "".join(sequence)


def build_sequence_2(surname: str, total_length: int) -> str:
    return surname + ("0" * (total_length - len(surname)))


def build_sequence_3(surname: str, total_length: int, rng: random.Random) -> str:
    sequence = list(surname) + (["0"] * (total_length - len(surname)))
    rng.shuffle(sequence)
    return "".join(sequence)


def build_sequence_4(surname: str, group_number: str, total_length: int) -> str:
    pattern = list(surname) + list(group_number)
    repeats, remainder = divmod(total_length, len(pattern))
    sequence = pattern * repeats + pattern[:remainder]
    return "".join(sequence)


def build_sequence_5(surname: str, group_number: str, total_length: int, rng: random.Random) -> str:
    elements = list(surname[:2]) + list(group_number)
    return build_equal_distribution(elements, total_length, rng)


def build_sequence_6(surname: str, group_number: str, total_length: int, rng: random.Random) -> str:
    letters = list(surname[:2])
    digits = list(group_number)
    letters_count = int(0.7 * total_length)
    digits_count = total_length - letters_count

    sequence: list[str] = []
    for _ in range(letters_count):
        sequence.append(rng.choice(letters))
    for _ in range(digits_count):
        sequence.append(rng.choice(digits))

    rng.shuffle(sequence)
    return "".join(sequence)


def build_sequence_7(total_length: int, rng: random.Random) -> str:
    elements = string.ascii_lowercase + string.digits
    return "".join(rng.choice(elements) for _ in range(total_length))


def build_sequence_8(total_length: int) -> str:
    return "1" * total_length


def analyze_sequence(sequence: str) -> dict[str, object]:
    counts = collections.Counter(sequence)
    sequence_alphabet_size = len(counts)
    sequence_size_bytes = len(sequence)
    probability = {
        symbol: count / len(sequence) for symbol, count in sorted(counts.items())
    }
    mean_probability = sum(probability.values()) / len(probability)
    equal = all(
        abs(prob - mean_probability) < 0.05 * mean_probability
        for prob in probability.values()
    )
    uniformity = "рівна" if equal else "нерівна"
    entropy = -sum(prob * math.log2(prob) for prob in probability.values())
    if sequence_alphabet_size > 1:
        source_excess = 1 - entropy / math.log2(sequence_alphabet_size)
    else:
        source_excess = 1

    return {
        "alphabet_size": sequence_alphabet_size,
        "size_bytes": sequence_size_bytes,
        "probability": probability,
        "mean_probability": mean_probability,
        "uniformity": uniformity,
        "entropy": entropy,
        "source_excess": source_excess,
    }


def write_sequences_file(
    sequences: list[tuple[str, str]], output_path: Path, metadata: dict[str, object]
) -> None:
    lines = [
        "Laboratory work #5: generated sequences",
        f"Surname: {metadata['surname']}",
        f"Group number: {metadata['group_number']}",
        f"Group digits used in sequences: {metadata['group_digits']}",
        f"Journal number: {metadata['journal_number']}",
        f"Sequence length: {metadata['length']}",
        "",
    ]
    for name, sequence in sequences:
        lines.append(f"{name}:")
        lines.append(sequence)
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_results_file(
    sequences: list[tuple[str, str]],
    analyses: list[dict[str, object]],
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    lines = [
        "Laboratory work #5: sequence characteristics",
        f"Surname: {metadata['surname']}",
        f"Group number: {metadata['group_number']}",
        f"Group digits used in sequences: {metadata['group_digits']}",
        f"Journal number: {metadata['journal_number']}",
        f"Sequence length: {metadata['length']}",
        f"Random seed: {metadata['seed']}",
        "Sequence size in bytes is calculated according to the laboratory condition: 1 symbol = 1 byte.",
        "",
    ]

    for index, ((name, sequence), analysis) in enumerate(zip(sequences, analyses), start=1):
        probability_str = ", ".join(
            f"{symbol}={prob:.4f}"
            for symbol, prob in analysis["probability"].items()
        )
        lines.extend(
            [
                f"Тестова послідовність №{index} ({name})",
                f"Послідовність: {sequence}",
                f"Розмір алфавіту: {analysis['alphabet_size']}",
                f"Розмір послідовності: {analysis['size_bytes']} байт",
                f"Ймовірності: {probability_str}",
                f"Середня ймовірність: {analysis['mean_probability']:.4f}",
                f"Тип ймовірності: {analysis['uniformity']}",
                f"Ентропія: {analysis['entropy']:.4f}",
                f"Надмірність джерела: {analysis['source_excess']:.4f}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def save_table(
    analyses: list[dict[str, object]],
    output_path: Path,
) -> None:
    headers = ["Розмір алфавіту", "Ентропія", "Надмірність", "Ймовірність"]
    rows = [f"Послідовність {index}" for index in range(1, len(analyses) + 1)]
    table_data = [
        [
            analysis["alphabet_size"],
            f"{analysis['entropy']:.2f}",
            f"{analysis['source_excess']:.2f}",
            analysis["uniformity"],
        ]
        for analysis in analyses
    ]

    fig, ax = plt.subplots(figsize=(14 / 1.54, len(analyses) / 1.54 + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        rowLabels=rows,
        loc="center",
        cellLoc="center",
    )
    table.set_fontsize(12)
    table.scale(0.9, 1.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    group_digits = extract_group_digits(args.group_number)

    sequences = [
        ("original sequence 1", build_sequence_1(args.journal_number, args.length, rng)),
        ("original sequence 2", build_sequence_2(args.surname, args.length)),
        ("original sequence 3", build_sequence_3(args.surname, args.length, rng)),
        ("original sequence 4", build_sequence_4(args.surname, group_digits, args.length)),
        ("original sequence 5", build_sequence_5(args.surname, group_digits, args.length, rng)),
        ("original sequence 6", build_sequence_6(args.surname, group_digits, args.length, rng)),
        ("original sequence 7", build_sequence_7(args.length, rng)),
        ("original sequence 8", build_sequence_8(args.length)),
    ]
    analyses = [analyze_sequence(sequence) for _, sequence in sequences]

    metadata = {
        "surname": args.surname,
        "group_number": args.group_number,
        "group_digits": group_digits,
        "journal_number": args.journal_number,
        "length": args.length,
        "seed": args.seed,
    }

    write_sequences_file(sequences, args.output_dir / "sequence.txt", metadata)
    write_results_file(
        sequences,
        analyses,
        args.output_dir / "results_sequence.txt",
        metadata,
    )
    save_table(
        analyses,
        args.output_dir / "Характеристики сформованих послідовностей.png",
    )

    print(f"Generated files in: {args.output_dir}")


if __name__ == "__main__":
    main()
