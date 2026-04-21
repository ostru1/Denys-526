from __future__ import annotations

import ast
import collections
import heapq
import math
import re
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
SEQUENCE_PATH = BASE_DIR / "sequence.txt"
RESULTS_PATH = BASE_DIR / "results_AC_CH.txt"
TABLE_PATH = BASE_DIR / "Результати стиснення методами AC та CH.png"


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
    current_parts: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"original sequence \d+:", line, flags=re.IGNORECASE):
            if current_name is not None:
                sequences.append((current_name, "".join(current_parts).strip()))
            current_name = line[:-1]
            current_parts = []
            continue
        if current_name is not None:
            current_parts.append(line)

    if current_name is not None:
        sequences.append((current_name, "".join(current_parts).strip()))

    if not sequences:
        raise ValueError("Could not parse sequences from sequence.txt.")

    return sequences


def load_sequences(sequence_path: Path) -> list[tuple[str, str]]:
    return parse_sequences(sequence_path.read_text(encoding="utf-8"))


def analyze_sequence(
    sequence: str,
) -> tuple[collections.Counter[str], dict[str, Fraction], float]:
    counts = collections.Counter(sequence)
    probabilities = {
        symbol: Fraction(count, len(sequence))
        for symbol, count in sorted(counts.items())
    }
    entropy = -sum(float(prob) * math.log2(float(prob)) for prob in probabilities.values())
    if abs(entropy) < 1e-12:
        entropy = 0.0
    return counts, probabilities, entropy


def format_probability(probabilities: dict[str, Fraction]) -> str:
    return ", ".join(f"{symbol}={float(prob):.4f}" for symbol, prob in probabilities.items())


def format_fraction(value: Fraction) -> str:
    with localcontext() as context:
        context.prec = 80
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
    formatted = format(decimal_value, "f").rstrip("0").rstrip(".")
    return formatted or "0"


def build_ranges(probabilities: dict[str, Fraction]) -> list[tuple[str, Fraction, Fraction]]:
    ranges: list[tuple[str, Fraction, Fraction]] = []
    border = Fraction(0, 1)
    for symbol, probability in probabilities.items():
        lower = border
        border += probability
        ranges.append((symbol, lower, border))
    return ranges


def normalize_probabilities(
    uniq_chars: list[str],
    probabilitys: dict[str, Fraction] | list[Fraction],
) -> dict[str, Fraction]:
    if isinstance(probabilitys, dict):
        return {symbol: probabilitys[symbol] for symbol in uniq_chars}
    return {symbol: probabilitys[index] for index, symbol in enumerate(uniq_chars)}


def binary_fraction_in_interval(low: Fraction, high: Fraction) -> tuple[str, Fraction]:
    if not (Fraction(0, 1) <= low < high <= Fraction(1, 1)):
        raise ValueError("Arithmetic interval must be inside [0, 1].")

    digits = 1
    while True:
        scale = 1 << digits
        min_k = (low.numerator * scale) // low.denominator + 1
        max_k = ((high.numerator * scale) - 1) // high.denominator
        if min_k <= max_k:
            point = Fraction(min_k, scale)
            return format(min_k, f"0{digits}b"), point
        digits += 1


def encode_ac(
    uniq_chars: list[str],
    probabilitys: dict[str, Fraction] | list[Fraction],
    alphabet_size: int,
    sequence: str,
) -> tuple[dict[str, object], str]:
    probabilities = normalize_probabilities(uniq_chars, probabilitys)
    if len(probabilities) != alphabet_size:
        raise ValueError("Alphabet size does not match the number of unique symbols.")

    ranges = build_ranges(probabilities)
    low = Fraction(0, 1)
    high = Fraction(1, 1)

    range_by_symbol = {symbol: (lower, upper) for symbol, lower, upper in ranges}
    for symbol in sequence:
        width = high - low
        symbol_low, symbol_high = range_by_symbol[symbol]
        new_low = low + width * symbol_low
        new_high = low + width * symbol_high
        low, high = new_low, new_high

    binary_code, point = binary_fraction_in_interval(low, high)
    encoded_data = {
        "payload": [
            point,
            alphabet_size,
            list(probabilities.keys()),
            [probabilities[symbol] for symbol in probabilities],
        ],
        "point": point,
        "binary_code": binary_code,
        "alphabet": list(probabilities.keys()),
        "probabilities": [probabilities[symbol] for symbol in probabilities],
        "final_interval": (low, high),
    }
    return encoded_data, binary_code


def decode_ac(encoded_data: dict[str, object], sequence_length: int) -> str:
    point = encoded_data["point"]
    alphabet = encoded_data["alphabet"]
    probability_values = encoded_data["probabilities"]
    if not isinstance(point, Fraction):
        raise TypeError("Encoded AC point must be Fraction.")
    if not isinstance(alphabet, list):
        raise TypeError("Encoded AC alphabet must be a list.")
    if not isinstance(probability_values, list):
        raise TypeError("Encoded AC probabilities must be a list.")

    probabilities = normalize_probabilities(alphabet, probability_values)

    ranges = build_ranges(probabilities)
    low = Fraction(0, 1)
    high = Fraction(1, 1)
    decoded: list[str] = []

    for _ in range(sequence_length):
        width = high - low
        normalized = (point - low) / width
        for symbol, symbol_low, symbol_high in ranges:
            if symbol_low <= normalized < symbol_high:
                decoded.append(symbol)
                new_low = low + width * symbol_low
                new_high = low + width * symbol_high
                low, high = new_low, new_high
                break
        else:
            raise ValueError("Could not decode arithmetic code.")

    return "".join(decoded)


class HuffmanNode:
    def __init__(
        self,
        weight: int,
        symbol: str | None = None,
        left: HuffmanNode | None = None,
        right: HuffmanNode | None = None,
    ) -> None:
        self.weight = weight
        self.symbol = symbol
        self.left = left
        self.right = right


def build_huffman_codes(counts: collections.Counter[str]) -> dict[str, str]:
    ordered_symbols = sorted(counts.items())
    if len(ordered_symbols) == 1:
        symbol = ordered_symbols[0][0]
        return {symbol: "0"}

    heap: list[tuple[int, int, HuffmanNode]] = []
    order = 0
    for symbol, weight in ordered_symbols:
        heapq.heappush(heap, (weight, order, HuffmanNode(weight, symbol=symbol)))
        order += 1

    while len(heap) > 1:
        left_weight, _, left_node = heapq.heappop(heap)
        right_weight, _, right_node = heapq.heappop(heap)
        parent = HuffmanNode(
            left_weight + right_weight,
            left=left_node,
            right=right_node,
        )
        heapq.heappush(heap, (parent.weight, order, parent))
        order += 1

    _, _, root = heap[0]
    codes: dict[str, str] = {}

    def walk(node: HuffmanNode, prefix: str) -> None:
        if node.symbol is not None:
            codes[node.symbol] = prefix or "0"
            return
        if node.left is not None:
            walk(node.left, prefix + "0")
        if node.right is not None:
            walk(node.right, prefix + "1")

    walk(root, "")
    return dict(sorted(codes.items()))


def encode_ch(
    uniq_chars: list[str],
    probabilitys: dict[str, Fraction] | list[Fraction],
    sequence: str,
) -> tuple[dict[str, object], str]:
    probabilities = normalize_probabilities(uniq_chars, probabilitys)
    counts = collections.Counter(sequence)
    if set(counts) != set(probabilities):
        raise ValueError("Alphabet and sequence symbols do not match.")

    codes = build_huffman_codes(counts)
    encoded = "".join(codes[symbol] for symbol in sequence)
    symbol_code = [[symbol, code] for symbol, code in sorted(codes.items())]
    encoded_data = {
        "payload": [encoded, symbol_code],
        "encoded": encoded,
        "codes": symbol_code,
    }
    return encoded_data, encoded


def decode_ch(encoded_data: dict[str, object]) -> str:
    encoded = encoded_data["encoded"]
    codes = encoded_data["codes"]
    if not isinstance(encoded, str):
        raise TypeError("Encoded CH data must contain a bit string.")
    if not isinstance(codes, list):
        raise TypeError("Encoded CH data must contain a code table.")

    reverse_codes = {code: symbol for symbol, code in codes}
    buffer = ""
    decoded: list[str] = []

    for bit in encoded:
        buffer += bit
        if buffer in reverse_codes:
            decoded.append(reverse_codes[buffer])
            buffer = ""

    if buffer:
        raise ValueError("Huffman bit stream ended with an incomplete code.")

    return "".join(decoded)


def float_bin(point: Fraction | float, size_cod: int) -> str:
    if not isinstance(point, Fraction):
        point = Fraction(point).limit_denominator()
    if point < 0 or point >= 1:
        raise ValueError("Point for binary conversion must be in [0, 1).")

    binary_code = ""
    current = point
    for _ in range(size_cod):
        current *= 2
        if current >= 1:
            binary_code += "1"
            current -= 1
        else:
            binary_code += "0"
        if current == 0:
            break
    return binary_code


def encode_hc(
    uniq_chars: list[str],
    probabilitys: dict[str, Fraction] | list[Fraction],
    sequence: str,
) -> tuple[dict[str, object], str]:
    return encode_ch(uniq_chars, probabilitys, sequence)


def decode_hc(encoded_data: dict[str, object]) -> str:
    return decode_ch(encoded_data)


def build_results(sequences: list[tuple[str, str]]) -> tuple[str, list[list[object]]]:
    sections: list[str] = [
        "Лабораторна робота №7",
        "Стиснення без втрат: арифметичне кодування та кодування Хаффмана",
        "",
    ]
    summary_rows: list[list[object]] = []

    for index, (name, sequence) in enumerate(sequences, start=1):
        counts, probabilities, entropy = analyze_sequence(sequence)
        uniq_chars = list(probabilities.keys())
        alphabet = "".join(uniq_chars)
        sequence_length = len(sequence)

        encoded_data_ac, encoded_sequence_ac = encode_ac(
            uniq_chars,
            probabilities,
            len(uniq_chars),
            sequence,
        )
        decoded_sequence_ac = decode_ac(encoded_data_ac, sequence_length)
        if decoded_sequence_ac != sequence:
            raise ValueError(f"AC decode mismatch for {name}.")
        bps_ac = len(encoded_sequence_ac) / sequence_length

        encoded_data_ch, encoded_sequence_ch = encode_ch(
            uniq_chars,
            probabilities,
            sequence,
        )
        decoded_sequence_ch = decode_ch(encoded_data_ch)
        if decoded_sequence_ch != sequence:
            raise ValueError(f"CH decode mismatch for {name}.")
        bps_ch = len(encoded_sequence_ch) / sequence_length

        summary_rows.append([round(entropy, 2), round(bps_ac, 4), round(bps_ch, 4)])

        ac_low, ac_high = encoded_data_ac["final_interval"]
        codes_table = ", ".join(f"{symbol}={code}" for symbol, code in encoded_data_ch["codes"])
        ac_payload = encoded_data_ac["payload"]
        ch_payload = encoded_data_ch["payload"]
        float_bin_check = float_bin(encoded_data_ac["point"], len(encoded_sequence_ac))

        sections.extend(
            [
                f"Послідовність {index} ({name})",
                f"Оригінальна послідовність: {sequence}",
                f"Довжина послідовності: {sequence_length}",
                f"Алфавіт ({len(probabilities)}): {alphabet}",
                f"Ймовірності: {format_probability(probabilities)}",
                f"Ентропія: {entropy:.4f}",
                "",
                "Арифметичне кодування (AC)",
                f"encoded_data_ac: {ac_payload}",
                f"Фінальний інтервал: [{format_fraction(ac_low)}, {format_fraction(ac_high)})",
                f"Ширина фінального інтервалу: {float(ac_high - ac_low):.6e}",
                f"Точка: {format_fraction(encoded_data_ac['point'])}",
                f"Двійковий код: {encoded_sequence_ac}",
                f"float_bin(point, len(code)): {float_bin_check}",
                f"Кількість біт: {len(encoded_sequence_ac)}",
                f"bps AC: {bps_ac:.4f}",
                f"Декодована послідовність AC: {decoded_sequence_ac}",
                "",
                "Кодування Хаффмана (CH)",
                f"encoded_data_ch: {ch_payload}",
                f"Коди символів: {codes_table}",
                f"Закодована послідовність CH: {encoded_sequence_ch}",
                f"Кількість біт: {len(encoded_sequence_ch)}",
                f"bps CH: {bps_ch:.4f}",
                f"Декодована послідовність CH: {decoded_sequence_ch}",
                "",
                "=" * 80,
                "",
            ]
        )

    return "\n".join(sections).rstrip() + "\n", summary_rows


def save_table(results: list[list[object]], output_path: Path) -> None:
    row_count = len(results)
    headers = ["Ентропія", "bps AC", "bps CH"]
    rows = [f"Послідовність {index}" for index in range(1, row_count + 1)]

    fig, ax = plt.subplots(figsize=(14 / 1.54, row_count / 1.54 + 1))
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
