from __future__ import annotations

import ast
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SEQUENCE_FILE = BASE_DIR / "sequence.txt"
RESULTS_FILE = BASE_DIR / "results_hamming.txt"
ALT_RESULTS_FILE = BASE_DIR / "result_hamming.txt"

CHUNK_LENGTH = 8
assert not CHUNK_LENGTH % 8, "Довжина блоку має бути кратна 8"
CHECK_BITS = [i for i in range(1, CHUNK_LENGTH + 1) if not i & (i - 1)]
ENCODED_CHUNK_LENGTH = CHUNK_LENGTH + len(CHECK_BITS)


def getCharsToBin(chars):
    assert not len(chars) * 16 % CHUNK_LENGTH, (
        "Довжина кодових даних повинна бути кратною довжині блоку кодування"
    )
    return "".join(bin(ord(char))[2:].zfill(16) for char in chars)


def getChunkIterator(text_bin, chunk_size=CHUNK_LENGTH):
    for i in range(0, len(text_bin), chunk_size):
        yield text_bin[i : i + chunk_size]


def getCheckBitsData(value_bin):
    check_bits_count_map = {check_bit: 0 for check_bit in CHECK_BITS}

    for index, value in enumerate(value_bin, 1):
        if int(value):
            for check_bit in CHECK_BITS:
                if index & check_bit:
                    check_bits_count_map[check_bit] += 1

    return {
        check_bit: 0 if not count % 2 else 1
        for check_bit, count in check_bits_count_map.items()
    }


def getSetEmptyCheckBits(value_bin):
    for bit in CHECK_BITS:
        value_bin = value_bin[: bit - 1] + "0" + value_bin[bit - 1 :]
    return value_bin


def getSetCheckBits(value_bin):
    value_bin = getSetEmptyCheckBits(value_bin)
    check_bits_data = getCheckBitsData(value_bin)

    for check_bit, bit_value in check_bits_data.items():
        value_bin = f"{value_bin[:check_bit - 1]}{bit_value}{value_bin[check_bit:]}"

    return value_bin


def getCheckBits(value_bin):
    check_bits = {}
    for index, value in enumerate(value_bin, 1):
        if index in CHECK_BITS:
            check_bits[index] = int(value)
    return check_bits


def getExcludeCheckBits(value_bin):
    clean_value_bin = ""
    for index, char_bin in enumerate(value_bin, 1):
        if index not in CHECK_BITS:
            clean_value_bin += char_bin
    return clean_value_bin


def getSetErrors(encoded):
    result = ""

    for chunk in getChunkIterator(encoded, ENCODED_CHUNK_LENGTH):
        num_bit = random.randint(1, len(chunk))
        chunk = f"{chunk[:num_bit - 1]}{int(chunk[num_bit - 1]) ^ 1}{chunk[num_bit:]}"
        result += chunk

    return result


def getCheckAndFixError(encoded_chunk):
    check_bits_encoded = getCheckBits(encoded_chunk)
    check_item = getExcludeCheckBits(encoded_chunk)
    check_item = getSetCheckBits(check_item)
    check_bits = getCheckBits(check_item)

    if check_bits_encoded != check_bits:
        invalid_bits = []
        for check_bit_encoded, value in check_bits_encoded.items():
            if check_bits[check_bit_encoded] != value:
                invalid_bits.append(check_bit_encoded)

        num_bit = sum(invalid_bits)
        encoded_chunk = (
            f"{encoded_chunk[:num_bit - 1]}"
            f"{int(encoded_chunk[num_bit - 1]) ^ 1}"
            f"{encoded_chunk[num_bit:]}"
        )

    return encoded_chunk


def getDiffIndexList(value_bin1, value_bin2):
    diff_index_list = []
    for index, char_bin_items in enumerate(zip(list(value_bin1), list(value_bin2)), 1):
        if char_bin_items[0] != char_bin_items[1]:
            diff_index_list.append(index)
    return diff_index_list


def encode(source):
    text_bin = getCharsToBin(source)
    result = ""

    for chunk_bin in getChunkIterator(text_bin):
        result += getSetCheckBits(chunk_bin)

    return text_bin, result


def decode(encoded, fix_errors=True):
    decoded_value = ""
    fixed_encoded_list = []

    for encoded_chunk in getChunkIterator(encoded, ENCODED_CHUNK_LENGTH):
        if fix_errors:
            encoded_chunk = getCheckAndFixError(encoded_chunk)
        fixed_encoded_list.append(encoded_chunk)

    clean_chunk_list = []
    for encoded_chunk in fixed_encoded_list:
        clean_chunk_list.append(getExcludeCheckBits(encoded_chunk))

    clean_chunk_list = "".join(clean_chunk_list)

    for i in range(0, len(clean_chunk_list), 16):
        clean_char = clean_chunk_list[i : i + 16]
        if len(clean_char) == 16:
            decoded_value += chr(int(clean_char, 2))

    return decoded_value


def format_sequence_report(
    sequence_index,
    source,
    source_bin,
    encoded,
    decoded,
    encoded_with_error,
    diff_index_list,
    decoded_with_error,
    decoded_without_error,
):
    lines = [
        f"Sequence #{sequence_index}",
        f"Original sequence (bytes): {source}",
        f"Original sequence (bits): {source_bin}",
        f"Original sequence size in bits: {len(source_bin)}",
        f"Chunk length: {CHUNK_LENGTH}",
        f"Check bits positions: {CHECK_BITS}",
        f"Relative redundancy: {len(CHECK_BITS) / CHUNK_LENGTH:.4f}",
        f"Encoded data: {encoded}",
        f"Encoded data size in bits: {len(encoded)}",
        f"Decoded data: {decoded}",
        f"Decoded data size in bits: {len(decoded) * 16}",
        f"Encoded data with errors: {encoded_with_error}",
        f"Errors count: {len(diff_index_list)}",
        f"Error indexes: {diff_index_list}",
        f"Decoded data without error correction: {decoded_with_error}",
        f"Decoded data with error correction: {decoded_without_error}",
        "-" * 120,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    random.seed(42)

    with RESULTS_FILE.open("w", encoding="utf-8") as result_file:
        with SEQUENCE_FILE.open("r", encoding="utf-8") as file:
            original_sequences = ast.literal_eval(file.read())
            original_sequences = [str(seq).strip("[]").strip("'") for seq in original_sequences]

        for sequence_index, sequence in enumerate(original_sequences, 1):
            source = sequence[:10]
            source_bin, encoded = encode(source)
            decoded = decode(encoded)
            encoded_with_error = getSetErrors(encoded)
            diff_index_list = getDiffIndexList(encoded, encoded_with_error)
            decoded_with_error = decode(encoded_with_error, fix_errors=False)
            decoded_without_error = decode(encoded_with_error)

            report_block = format_sequence_report(
                sequence_index=sequence_index,
                source=source,
                source_bin=source_bin,
                encoded=encoded,
                decoded=decoded,
                encoded_with_error=encoded_with_error,
                diff_index_list=diff_index_list,
                decoded_with_error=decoded_with_error,
                decoded_without_error=decoded_without_error,
            )
            result_file.write(report_block + "\n")

    ALT_RESULTS_FILE.write_text(RESULTS_FILE.read_text(encoding="utf-8"), encoding="utf-8")
