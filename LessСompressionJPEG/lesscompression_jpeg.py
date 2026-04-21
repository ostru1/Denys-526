from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import fftpack

from huffman import HuffmanTree


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "Images"
RESULTS_DIR = BASE_DIR / "Results"
RESULTS_FILE = BASE_DIR / "results_jpeg.txt"

TABLE_SIZE_BITS = 16
CATEGORY_BITS = 4
DC_CODE_LENGTH_BITS = 4
RUN_LENGTH_BITS = 4
SIZE_BITS = 4
AC_CODE_LENGTH_BITS = 8
BLOCKS_COUNT_BITS = 32
BLOCK_SIDE = 8


def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm="ortho").T, norm="ortho")


def load_quantization_table(component, table_id):
    tables = {
        1: {
            "lum": np.array(
                [
                    [2, 2, 2, 2, 3, 4, 5, 6],
                    [2, 2, 2, 2, 3, 4, 5, 6],
                    [2, 2, 2, 2, 4, 5, 7, 9],
                    [2, 2, 2, 4, 5, 7, 9, 12],
                    [3, 3, 4, 5, 8, 10, 12, 12],
                    [4, 4, 5, 7, 10, 12, 12, 12],
                    [5, 5, 7, 9, 12, 12, 12, 12],
                    [6, 6, 9, 12, 12, 12, 12, 12],
                ]
            ),
            "chrom": np.array(
                [
                    [3, 3, 5, 9, 13, 15, 15, 15],
                    [3, 4, 6, 11, 14, 12, 12, 12],
                    [5, 6, 9, 14, 12, 12, 12, 12],
                    [9, 11, 14, 12, 12, 12, 12, 12],
                    [13, 14, 12, 12, 12, 12, 12, 12],
                    [15, 12, 12, 12, 12, 12, 12, 12],
                    [15, 12, 12, 12, 12, 12, 12, 12],
                    [15, 12, 12, 12, 12, 12, 12, 12],
                ]
            ),
        },
        2: {
            "lum": np.array(
                [
                    [16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 48, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99],
                ]
            ),
            "chrom": np.array(
                [
                    [17, 18, 24, 47, 99, 99, 99, 99],
                    [18, 21, 26, 66, 99, 99, 99, 99],
                    [24, 26, 56, 99, 99, 99, 99, 99],
                    [47, 66, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                ]
            ),
        },
    }
    if component not in {"lum", "chrom"}:
        raise ValueError(f"component must be 'lum' or 'chrom', got {component}")
    return tables[table_id][component]


def quantize(block, component, table_id):
    q = load_quantization_table(component, table_id)
    return (block / q).round().astype(np.int32)


def zigzag_points(rows, cols):
    up, down, right, left, up_right, down_left = range(6)

    def move(direction, point):
        return {
            up: lambda point: (point[0] - 1, point[1]),
            down: lambda point: (point[0] + 1, point[1]),
            left: lambda point: (point[0], point[1] - 1),
            right: lambda point: (point[0], point[1] + 1),
            up_right: lambda point: move(up, move(right, point)),
            down_left: lambda point: move(down, move(left, point)),
        }[direction](point)

    def inbounds(point):
        return 0 <= point[0] < rows and 0 <= point[1] < cols

    point = (0, 0)
    move_up = True

    for _ in range(rows * cols):
        yield point
        if move_up:
            if inbounds(move(up_right, point)):
                point = move(up_right, point)
            else:
                move_up = False
                if inbounds(move(right, point)):
                    point = move(right, point)
                else:
                    point = move(down, point)
        else:
            if inbounds(move(down_left, point)):
                point = move(down_left, point)
            else:
                move_up = True
                if inbounds(move(down, point)):
                    point = move(down, point)
                else:
                    point = move(right, point)


def block_to_zigzag(block):
    return np.array([block[point] for point in zigzag_points(*block.shape)])


def bits_required(n):
    n = abs(int(n))
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def run_length_encode(arr):
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    symbols = []
    values = []
    run_length = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(int_to_binstr(0))
            break
        if elem == 0 and run_length < 15:
            run_length += 1
            continue
        while elem == 0 and run_length >= 15:
            symbols.append((15, 0))
            values.append(int_to_binstr(0))
            run_length -= 15
        size = bits_required(elem)
        symbols.append((run_length, size))
        values.append(int_to_binstr(int(elem)))
        run_length = 0

    if not symbols:
        symbols.append((0, 0))
        values.append(int_to_binstr(0))

    return symbols, values


def binstr_flip(binstr):
    if not set(binstr).issubset("01"):
        raise ValueError("binstr must contain only '0' and '1'")
    return "".join("0" if char == "1" else "1" for char in binstr)


def uint_to_binstr(number, size):
    return bin(number)[2:][-size:].zfill(size)


def int_to_binstr(n):
    if n == 0:
        return ""
    binstr = bin(abs(int(n)))[2:]
    return binstr if n > 0 else binstr_flip(binstr)


def write_to_file(filepath, dc, ac, blocks_count, tables):
    with open(filepath, "w", encoding="utf-8") as file:
        for table_name in ["dc_y", "ac_y", "dc_c", "ac_c"]:
            file.write(uint_to_binstr(len(tables[table_name]), TABLE_SIZE_BITS))
            for key, value in tables[table_name].items():
                if table_name in {"dc_y", "dc_c"}:
                    file.write(uint_to_binstr(key, CATEGORY_BITS))
                    file.write(uint_to_binstr(len(value), DC_CODE_LENGTH_BITS))
                    file.write(value)
                else:
                    file.write(uint_to_binstr(key[0], RUN_LENGTH_BITS))
                    file.write(uint_to_binstr(key[1], SIZE_BITS))
                    file.write(uint_to_binstr(len(value), AC_CODE_LENGTH_BITS))
                    file.write(value)

        file.write(uint_to_binstr(blocks_count, BLOCKS_COUNT_BITS))

        for block_index in range(blocks_count):
            for component in range(3):
                category = bits_required(dc[block_index, component])
                symbols, values = run_length_encode(ac[block_index, :, component])

                dc_table = tables["dc_y"] if component == 0 else tables["dc_c"]
                ac_table = tables["ac_y"] if component == 0 else tables["ac_c"]

                file.write(dc_table[category])
                file.write(int_to_binstr(dc[block_index, component]))

                for i, symbol in enumerate(symbols):
                    file.write(ac_table[tuple(symbol)])
                    file.write(values[i])


def normalize_huffman_table(table):
    return {key: value or "0" for key, value in table.items()}


class JPEGFileReader:
    def __init__(self, filepath):
        self._data = Path(filepath).read_text(encoding="utf-8")
        self._position = 0

    def read(self, size):
        data = self._data[self._position : self._position + size]
        if len(data) != size:
            raise ValueError("Unexpected end of encoded file.")
        self._position += size
        return data

    def read_uint(self, size):
        return int(self.read(size), 2)

    def read_int(self, size):
        if size == 0:
            return 0
        binstr = self.read(size)
        if binstr[0] == "1":
            return int(binstr, 2)
        return -int(binstr_flip(binstr), 2)

    def read_dc_table(self):
        table = {}
        table_size = self.read_uint(TABLE_SIZE_BITS)
        for _ in range(table_size):
            category = self.read_uint(CATEGORY_BITS)
            code_length = self.read_uint(DC_CODE_LENGTH_BITS)
            code = self.read(code_length)
            table[code] = category
        return table

    def read_ac_table(self):
        table = {}
        table_size = self.read_uint(TABLE_SIZE_BITS)
        for _ in range(table_size):
            run_length = self.read_uint(RUN_LENGTH_BITS)
            size = self.read_uint(SIZE_BITS)
            code_length = self.read_uint(AC_CODE_LENGTH_BITS)
            code = self.read(code_length)
            table[code] = (run_length, size)
        return table

    def read_blocks_count(self):
        return self.read_uint(BLOCKS_COUNT_BITS)

    def read_huffman_code(self, table):
        prefix = ""
        while prefix not in table:
            prefix += self.read(1)
        return table[prefix]


def read_image_file(filepath):
    reader = JPEGFileReader(filepath)

    tables = {}
    for table_name in ["dc_y", "ac_y", "dc_c", "ac_c"]:
        if "dc" in table_name:
            tables[table_name] = reader.read_dc_table()
        else:
            tables[table_name] = reader.read_ac_table()

    blocks_count = reader.read_blocks_count()
    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

    for block_index in range(blocks_count):
        for component in range(3):
            dc_table = tables["dc_y"] if component == 0 else tables["dc_c"]
            ac_table = tables["ac_y"] if component == 0 else tables["ac_c"]

            category = reader.read_huffman_code(dc_table)
            dc[block_index, component] = reader.read_int(category)

            cells_count = 0
            while cells_count < 63:
                run_length, size = reader.read_huffman_code(ac_table)
                if (run_length, size) == (0, 0):
                    while cells_count < 63:
                        ac[block_index, cells_count, component] = 0
                        cells_count += 1
                    break
                else:
                    for _ in range(run_length):
                        if cells_count >= 63:
                            break
                        ac[block_index, cells_count, component] = 0
                        cells_count += 1
                    if cells_count >= 63:
                        break
                    if size == 0:
                        ac[block_index, cells_count, component] = 0
                    else:
                        ac[block_index, cells_count, component] = reader.read_int(size)
                    cells_count += 1

    return dc, ac, tables, blocks_count


def zigzag_to_block(zigzag):
    rows = cols = int(math.sqrt(len(zigzag)))
    if rows * cols != len(zigzag):
        raise ValueError("Zigzag length must be a perfect square.")

    block = np.empty((rows, cols), np.int32)
    for i, point in enumerate(zigzag_points(rows, cols)):
        block[point] = zigzag[i]
    return block


def dequantize(block, component, table_id):
    q = load_quantization_table(component, table_id)
    return block * q


def idct_2d(image):
    return fftpack.idct(fftpack.idct(image.T, norm="ortho").T, norm="ortho")


def prepare_square_image(image_path):
    image = Image.open(image_path).convert("RGB")
    side = min(image.size)
    side -= side % BLOCK_SIDE
    image = image.crop((0, 0, side, side))
    image.save(image_path)
    return image_path


def ensure_test_images():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    side = 256

    weak_path = IMAGES_DIR / "weak_texture.bmp"
    medium_path = IMAGES_DIR / "medium_texture.bmp"
    strong_path = IMAGES_DIR / "strong_texture.bmp"

    if not weak_path.exists():
        x = np.linspace(0, 1, side)
        y = np.linspace(0, 1, side)
        xx, yy = np.meshgrid(x, y)
        weak = np.zeros((side, side, 3), dtype=np.uint8)
        weak[..., 0] = np.clip(255 * (0.2 + 0.6 * xx), 0, 255).astype(np.uint8)
        weak[..., 1] = np.clip(255 * (0.3 + 0.5 * yy), 0, 255).astype(np.uint8)
        weak[..., 2] = np.clip(255 * (0.4 + 0.3 * (xx + yy) / 2), 0, 255).astype(np.uint8)
        Image.fromarray(weak, "RGB").save(weak_path)

    if not medium_path.exists():
        x = np.linspace(0, 4 * math.pi, side)
        y = np.linspace(0, 4 * math.pi, side)
        xx, yy = np.meshgrid(x, y)
        wave1 = (np.sin(xx) + 1) * 0.5
        wave2 = (np.cos(yy * 1.5) + 1) * 0.5
        medium = np.zeros((side, side, 3), dtype=np.uint8)
        medium[..., 0] = (255 * wave1).astype(np.uint8)
        medium[..., 1] = (255 * wave2).astype(np.uint8)
        medium[..., 2] = (255 * ((wave1 + wave2) / 2)).astype(np.uint8)
        medium[48:208, 48:208, 0] = 220
        medium[80:176, 80:176, 1] = 40
        medium[112:144, :, 2] = 255
        Image.fromarray(medium, "RGB").save(medium_path)

    if not strong_path.exists():
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
        stripes = ((np.indices((side, side)).sum(axis=0) % 2) * 255).astype(np.uint8)
        strong = noise.copy()
        strong[..., 0] = (0.6 * strong[..., 0] + 0.4 * stripes).astype(np.uint8)
        strong[..., 1] = (0.7 * strong[..., 1] + 0.3 * np.roll(stripes, 3, axis=1)).astype(np.uint8)
        strong[..., 2] = (0.5 * strong[..., 2] + 0.5 * np.roll(stripes, 5, axis=0)).astype(np.uint8)
        Image.fromarray(strong, "RGB").save(strong_path)

    for image_path in [weak_path, medium_path, strong_path]:
        prepare_square_image(image_path)

    return [
        ("weak_texture", weak_path),
        ("medium_texture", medium_path),
        ("strong_texture", strong_path),
    ]


def encode(input_file, output_file, table_id):
    image = Image.open(input_file)
    ycbcr = image.convert("YCbCr")
    npmat = np.array(ycbcr, dtype=np.uint8)

    rows, cols = npmat.shape[0], npmat.shape[1]
    if not (rows % BLOCK_SIDE == 0 and cols % BLOCK_SIDE == 0 and rows == cols):
        raise ValueError("Image width and height must be equal and divisible by 8.")

    blocks_count = rows // BLOCK_SIDE * cols // BLOCK_SIDE
    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)
    block_index = 0

    for i in range(0, rows, BLOCK_SIDE):
        for j in range(0, cols, BLOCK_SIDE):
            for k in range(3):
                block = npmat[i : i + BLOCK_SIDE, j : j + BLOCK_SIDE, k].astype(np.int32) - 128
                dct_matrix = dct_2d(block)
                quant_matrix = quantize(dct_matrix, "lum" if k == 0 else "chrom", table_id)
                zigzag = block_to_zigzag(quant_matrix)
                dc[block_index, k] = zigzag[0]
                ac[block_index, :, k] = zigzag[1:]
            block_index += 1

    h_dc_y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
    h_dc_c = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
    h_ac_y = HuffmanTree(flatten(run_length_encode(ac[i, :, 0])[0] for i in range(blocks_count)))
    h_ac_c = HuffmanTree(
        flatten(run_length_encode(ac[i, :, j])[0] for i in range(blocks_count) for j in [1, 2])
    )

    tables = {
        "dc_y": normalize_huffman_table(h_dc_y.value_to_bitstring_table()),
        "ac_y": normalize_huffman_table(h_ac_y.value_to_bitstring_table()),
        "dc_c": normalize_huffman_table(h_dc_c.value_to_bitstring_table()),
        "ac_c": normalize_huffman_table(h_ac_c.value_to_bitstring_table()),
    }

    write_to_file(output_file, dc, ac, blocks_count, tables)
    return blocks_count


def decoder(encoded_file, output_image_path, table_id):
    dc, ac, _, blocks_count = read_image_file(encoded_file)
    image_side = int(math.sqrt(blocks_count)) * BLOCK_SIDE
    blocks_per_line = image_side // BLOCK_SIDE
    npmat = np.empty((image_side, image_side, 3), dtype=np.uint8)

    for block_index in range(blocks_count):
        i = block_index // blocks_per_line * BLOCK_SIDE
        j = block_index % blocks_per_line * BLOCK_SIDE

        for c in range(3):
            zigzag = [dc[block_index, c]] + list(ac[block_index, :, c])
            quant_matrix = zigzag_to_block(zigzag)
            dct_matrix = dequantize(quant_matrix, "lum" if c == 0 else "chrom", table_id)
            block = idct_2d(dct_matrix)
            npmat[i : i + BLOCK_SIDE, j : j + BLOCK_SIDE, c] = np.clip(block + 128, 0, 255)

    image = Image.fromarray(npmat, "YCbCr").convert("RGB")
    image.save(output_image_path, format="JPEG")
    return image


def process_image(image_name, image_path, table_id):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    encoded_path = RESULTS_DIR / f"{image_name}_table_{table_id}.asf"
    decoded_path = RESULTS_DIR / f"JPEG_{image_name}_table_{table_id}.jpg"

    encode(image_path, encoded_path, table_id)
    image = decoder(encoded_path, decoded_path, table_id)

    original_size = os.path.getsize(image_path)
    encoded_size = os.path.getsize(encoded_path)
    jpeg_size = os.path.getsize(decoded_path)
    width, height = image.size

    encoded_ratio = original_size / encoded_size if encoded_size else 0
    jpeg_ratio = original_size / jpeg_size if jpeg_size else 0

    return {
        "image_name": image_name,
        "table_id": table_id,
        "original_size": original_size,
        "encoded_size": encoded_size,
        "jpeg_size": jpeg_size,
        "width": width,
        "height": height,
        "encoded_ratio": encoded_ratio,
        "jpeg_ratio": jpeg_ratio,
        "encoded_path": encoded_path.name,
        "decoded_path": decoded_path.name,
    }


def write_results(results):
    lines = [
        "Практична робота №8",
        "Стиснення зображень методом JPEG",
        "",
    ]

    for result in results:
        lines.extend(
            [
                f"Зображення: {result['image_name']}",
                f"Таблиця квантування: {result['table_id']}",
                f"Розмір вихідного файла: {result['original_size']} байт",
                f"Розмір файлу з результатами стиснення: {result['encoded_size']} байт",
                f"Розмір файла JPEG: {result['jpeg_size']} байт",
                f"Розмір зображення JPEG: {result['width']}x{result['height']}",
                f"Коефіцієнт стиснення відносно encoded-файлу: {result['encoded_ratio']:.2f}",
                f"Коефіцієнт стиснення відносно JPEG: {result['jpeg_ratio']:.2f}",
                f"Файл з результатами стиснення: {result['encoded_path']}",
                f"Декодоване зображення JPEG: {result['decoded_path']}",
                "",
            ]
        )

    RESULTS_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    images = ensure_test_images()
    results = []
    for image_name, image_path in images:
        for table_id in (1, 2):
            results.append(process_image(image_name, image_path, table_id))
    write_results(results)
    print(f"Processed {len(results)} JPEG cases.")
    print(f"Saved results file: {RESULTS_FILE}")
    print(f"Saved outputs in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
