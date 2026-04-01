from __future__ import annotations

import ast
import collections
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BITS_PER_SYMBOL = 16
UNICODE_DICTIONARY_SIZE = 65536
BASE_DIR = Path(__file__).resolve().parent
SEQUENCE_PATH = BASE_DIR / "sequence.txt"
RESULTS_PATH = BASE_DIR / "results_rle_lzw.txt"
TABLE_PATH = BASE_DIR / "Результати стиснення методами RLE та LZW.png"


def parse_sequences(raw_text: str) -> list[tuple[str, str]]:
    stripped = raw_text.strip()
    if not stripped:
        raise ValueError("sequence.txt is empty.")

    if stripped.startswith("["):
        values = ast.literal_eval(stripped)
        if not isinstance(values, list):
            raise ValueError("Expected a list of sequences in sequence.txt.")
        return [
            (f"original sequence {index}", str(sequence).strip())
            for index, sequence in enumerate(values, start=1)
        ]

    lines = raw_text.splitlines()
    sequences: list[tuple[str, str]] = []
    current_name: str | None = None
    current_sequence_parts: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"original sequence \d+:", line, flags=re.IGNORECASE):
            if current_name is not None:
                sequences.append((current_name, "".join(current_sequence_parts).strip()))
            current_name = line[:-1]
            current_sequence_parts = []
            continue
        if current_name is not None:
            current_sequence_parts.append(line)

    if current_name is not None:
        sequences.append((current_name, "".join(current_sequence_parts).strip()))

    if not sequences:
        raise ValueError("Could not parse sequences from sequence.txt.")

    return sequences


def load_sequences(sequence_path: Path) -> list[tuple[str, str]]:
    raw_text = sequence_path.read_text(encoding="utf-8")
    sequences = parse_sequences(raw_text)
    return [(name, sequence.strip()) for name, sequence in sequences]


def analyze_sequence(sequence: str) -> tuple[dict[str, float], float]:
    counts = collections.Counter(sequence)
    probability = {
        symbol: count / len(sequence) for symbol, count in sorted(counts.items())
    }
    entropy = -sum(p * math.log2(p) for p in probability.values())
    if abs(entropy) < 1e-12:
        entropy = 0.0
    return probability, entropy


def encode_rle(sequence: str) -> tuple[str, list[tuple[str, int]]]:
    if not sequence:
        return "", []

    count = 1
    runs: list[tuple[str, int]] = []
    for index, item in enumerate(sequence):
        if index == 0:
            continue
        if item == sequence[index - 1]:
            count += 1
        else:
            runs.append((sequence[index - 1], count))
            count = 1

    runs.append((sequence[-1], count))
    encoded = "".join(f"{count}{symbol}" for symbol, count in runs)
    return encoded, runs


def decode_rle(runs: list[tuple[str, int]]) -> str:
    return "".join(symbol * count for symbol, count in runs)


def bits_for_lzw_code(code: int, dictionary_size: int) -> int:
    if code < UNICODE_DICTIONARY_SIZE:
        return BITS_PER_SYMBOL
    return math.ceil(math.log2(dictionary_size))


def encode_lzw(sequence: str) -> tuple[list[int], list[str], int]:
    if not sequence:
        return [], [], 0

    dictionary = {chr(index): index for index in range(UNICODE_DICTIONARY_SIZE)}
    current = ""
    result: list[int] = []
    log_lines: list[str] = []
    total_bits = 0

    for symbol in sequence:
        new_str = current + symbol
        if new_str in dictionary:
            current = new_str
            continue

        code = dictionary[current]
        result.append(code)
        dictionary[new_str] = len(dictionary)
        element_bits = bits_for_lzw_code(code, len(dictionary))
        log_lines.append(f"Code: {code}, Element: {current}, Bits: {element_bits}")
        total_bits += element_bits
        current = symbol

    if current:
        code = dictionary[current]
        last_bits = bits_for_lzw_code(code, len(dictionary))
        result.append(code)
        log_lines.append(f"Code: {code}, Element: {current}, Bits: {last_bits}")
        total_bits += last_bits

    return result, log_lines, total_bits


def decode_lzw(sequence: list[int]) -> str:
    if not sequence:
        return ""

    dictionary = {index: chr(index) for index in range(UNICODE_DICTIONARY_SIZE)}
    first_code = sequence[0]
    previous = dictionary[first_code]
    result = [previous]

    for code in sequence[1:]:
        if code in dictionary:
            current = dictionary[code]
        else:
            current = previous + previous[0]

        result.append(current)
        dictionary[len(dictionary)] = previous + current[0]
        previous = current

    return "".join(result)


def format_probability(probability: dict[str, float]) -> str:
    return ", ".join(f"{symbol}={value:.4f}" for symbol, value in probability.items())


def build_results(sequences: list[tuple[str, str]]) -> tuple[str, list[list[object]]]:
    sections: list[str] = []
    summary_rows: list[list[object]] = []

    sections.append("Лабораторна робота №6")
    sections.append("Стиснення без втрат методами RLE та LZW")
    sections.append("")

    for index, (name, sequence) in enumerate(sequences, start=1):
        probability, entropy = analyze_sequence(sequence)
        original_size_bits = len(sequence) * BITS_PER_SYMBOL

        encoded_rle, runs = encode_rle(sequence)
        decoded_rle = decode_rle(runs)
        if decoded_rle != sequence:
            raise ValueError(f"RLE decode mismatch for {name}.")
        encoded_rle_size_bits = len(encoded_rle) * BITS_PER_SYMBOL
        raw_rle_ratio = round(len(sequence) / len(encoded_rle), 2) if encoded_rle else 0
        compression_ratio_rle: str | float = raw_rle_ratio if raw_rle_ratio >= 1 else "-"

        encoded_lzw, lzw_log_lines, lzw_size_bits = encode_lzw(sequence)
        decoded_lzw = decode_lzw(encoded_lzw)
        if decoded_lzw != sequence:
            raise ValueError(f"LZW decode mismatch for {name}.")
        compression_ratio_lzw = (
            round((len(sequence) * BITS_PER_SYMBOL) / lzw_size_bits, 2)
            if lzw_size_bits
            else 0
        )

        summary_rows.append([round(entropy, 2), compression_ratio_rle, compression_ratio_lzw])

        sections.extend(
            [
                f"Послідовність {index} ({name})",
                f"Початкова послідовність: {sequence}",
                f"Ймовірності: {format_probability(probability)}",
                f"Ентропія: {entropy:.4f}",
                f"Розмір початкової послідовності: {original_size_bits} bits",
                "",
                "RLE кодування",
                f"Закодована RLE послідовність: {encoded_rle}",
                f"Серії RLE: {runs}",
                f"Розмір закодованої RLE послідовності: {encoded_rle_size_bits} bits",
                f"Коефіцієнт стиснення RLE: {compression_ratio_rle}",
                f"Декодована RLE послідовність: {decoded_rle}",
                f"Розмір декодованої RLE послідовності: {len(decoded_rle) * BITS_PER_SYMBOL} bits",
                "",
                "LZW кодування",
                *lzw_log_lines,
                f"Закодована LZW послідовність: {' '.join(map(str, encoded_lzw))}",
                f"Розмір закодованої LZW послідовності: {lzw_size_bits} bits",
                f"Коефіцієнт стиснення LZW: {compression_ratio_lzw}",
                f"Декодована LZW послідовність: {decoded_lzw}",
                f"Розмір декодованої LZW послідовності: {len(decoded_lzw) * BITS_PER_SYMBOL} bits",
                "",
                "=" * 80,
                "",
            ]
        )

    return "\n".join(sections).rstrip() + "\n", summary_rows


def save_table(results: list[list[object]], output_path: Path) -> None:
    row_count = len(results)
    headers = ["Ентропія", "КС RLE", "КС LZW"]
    rows = [f"Послідовність {index}" for index in range(1, row_count + 1)]

    fig, ax = plt.subplots(figsize=(14 / 1.54, row_count / 1.54))
    ax.axis("off")
    table = ax.table(
        cellText=results,
        colLabels=headers,
        rowLabels=rows,
        loc="center",
        cellLoc="center",
    )
    table.set_fontsize(14)
    table.scale(0.8, 2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sequences = load_sequences(SEQUENCE_PATH)
    results_text, table_rows = build_results(sequences)
    RESULTS_PATH.write_text(results_text, encoding="utf-8")
    save_table(table_rows, TABLE_PATH)
    print(f"Processed {len(sequences)} sequences.")
    print(f"Saved: {RESULTS_PATH}")
    print(f"Saved: {TABLE_PATH}")


if __name__ == "__main__":
    main()
