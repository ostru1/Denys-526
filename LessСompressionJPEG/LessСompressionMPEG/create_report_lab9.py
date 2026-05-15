from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pythoncom
import win32com.client


REPO_ROOT = Path(r"D:\Denys-526")
REPORT_DIR = REPO_ROOT / "screen"
RESULTS_DIR = REPO_ROOT / "LessСompressionJPEG" / "LessСompressionMPEG" / "Results"
DOCX_PATH = REPORT_DIR / "lab9.docx"
PDF_PATH = REPORT_DIR / "lab9.pdf"

METRICS = OrderedDict(
    [
        ("Пара кадрів", "2585 та 2586"),
        ("Відновлення кадру", "Повністю збігається з цільовим кадром"),
        ("Bits/pixel RGB для початкового кадру", "18.6566"),
        ("Bits/pixel RGB для міжкадрової різниці", "0.6199"),
        ("Bits/pixel RGB для залишкового кадру", "0.2461"),
        ("Коефіцієнт стиснення", "75.8199"),
        ("Bits/pixel R (початковий / різниця / залишок)", "6.5216 / 0.2065 / 0.0811"),
        ("Bits/pixel G (початковий / різниця / залишок)", "6.4736 / 0.2059 / 0.0820"),
        ("Bits/pixel B (початковий / різниця / залишок)", "5.6614 / 0.2074 / 0.0830"),
    ]
)

FIGURES = [
    {
        "file": RESULTS_DIR / "First frame.png",
        "caption": "Рисунок 1 – Перший (опорний) кадр відеопослідовності.",
        "text": (
            "На рисунку 1 наведено опорний кадр, який використовується як I-кадр "
            "для подальшого прогнозування наступного кадру."
        ),
    },
    {
        "file": RESULTS_DIR / "Second frame.png",
        "caption": "Рисунок 2 – Другий (цільовий) кадр відеопослідовності.",
        "text": (
            "На рисунку 2 наведено поточний кадр, який підлягає кодуванню. "
            "Саме для нього виконується пошук подібних блоків у попередньому кадрі."
        ),
    },
    {
        "file": RESULTS_DIR / "Difference between frame.png",
        "caption": "Рисунок 3 – Абсолютна різниця між сусідніми кадрами.",
        "text": (
            "На рисунку 3 показано міжкадрову різницю без компенсації руху. "
            "Більша частина кадру має малу інтенсивність змін, а локальні світлі "
            "ділянки відповідають зонам руху."
        ),
    },
    {
        "file": RESULTS_DIR / "Prediction frame.png",
        "caption": "Рисунок 4 – Прогнозований кадр після блочного пошуку.",
        "text": (
            "На рисунку 4 наведено прогнозований кадр, який сформовано алгоритмом "
            "block matching із використанням блоку 16x16 та зони пошуку 7 пікселів."
        ),
    },
    {
        "file": RESULTS_DIR / "Residual frame.png",
        "caption": "Рисунок 5 – Залишковий кадр після компенсації руху.",
        "text": (
            "На рисунку 5 наведено залишковий кадр. Низька енергія цього зображення "
            "підтверджує, що після компенсації руху для передавання залишається "
            "значно менше інформації."
        ),
    },
    {
        "file": RESULTS_DIR / "Restore frame.png",
        "caption": "Рисунок 6 – Відновлений кадр після декодування.",
        "text": (
            "На рисунку 6 наведено результат декодування. Відновлений кадр збігається "
            "з початковим цільовим кадром, що підтверджує коректність реалізації "
            "процедур обчислення залишку та реконструкції."
        ),
    },
    {
        "file": RESULTS_DIR / "Гістограма кількості біт на піксель для різних варіантів кодування.png",
        "caption": "Рисунок 7 – Гістограма кількості біт на піксель для різних варіантів кодування.",
        "text": (
            "На рисунку 7 наведено порівняння кількості біт на піксель для "
            "початкового кадру, простої різниці між кадрами та залишкового сигналу "
            "після компенсації руху. Найменше значення отримано для залишкового "
            "кадру, що відповідає суті MPEG/H.264-подібного міжкадрового кодування."
        ),
    },
]


WD_ALIGN_PARAGRAPH_LEFT = 0
WD_ALIGN_PARAGRAPH_CENTER = 1
WD_ALIGN_PARAGRAPH_JUSTIFY = 3
WD_PAGE_BREAK = 7
WD_FORMAT_PDF = 17
WD_STATISTIC_PAGES = 2
WD_STORY = 6
POINTS_PER_CM = 28.35


def ensure_inputs() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    missing = [str(item["file"]) for item in FIGURES if not item["file"].exists()]
    if missing:
        raise FileNotFoundError("Missing figures:\n" + "\n".join(missing))


def build_report() -> tuple[Path, Path]:
    ensure_inputs()
    pythoncom.CoInitialize()
    word = None
    doc = None

    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Add()
        selection = word.Selection

        def set_font(size: int = 14, bold: bool = False) -> None:
            selection.Font.Name = "Times New Roman"
            selection.Font.Size = size
            selection.Font.Bold = int(bold)

        def add_paragraph(
            text: str,
            alignment: int = WD_ALIGN_PARAGRAPH_LEFT,
            size: int = 14,
            bold: bool = False,
            first_line_indent_cm: float = 0.0,
        ) -> None:
            selection.ParagraphFormat.Alignment = alignment
            selection.ParagraphFormat.FirstLineIndent = first_line_indent_cm * POINTS_PER_CM
            set_font(size=size, bold=bold)
            selection.TypeText(text)
            selection.TypeParagraph()

        def add_body(text: str) -> None:
            add_paragraph(
                text,
                alignment=WD_ALIGN_PARAGRAPH_JUSTIFY,
                size=14,
                bold=False,
                first_line_indent_cm=1.25,
            )

        def add_blank_line() -> None:
            selection.TypeParagraph()

        def add_figure(text: str, image_path: Path, caption: str) -> None:
            add_body(text)
            add_blank_line()
            selection.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER
            selection.ParagraphFormat.FirstLineIndent = 0
            shape = selection.InlineShapes.AddPicture(str(image_path))
            if shape.Width > 430:
                shape.Width = 430
            selection.TypeParagraph()
            add_paragraph(caption, alignment=WD_ALIGN_PARAGRAPH_CENTER, size=12)
            add_blank_line()

        add_paragraph("МІНІСТЕРСТВО ОСВІТИ І НАУКИ УКРАЇНИ", WD_ALIGN_PARAGRAPH_CENTER, 14, True)
        add_paragraph("Національний аерокосмічний університет", WD_ALIGN_PARAGRAPH_CENTER, 14)
        add_paragraph("«Харківський авіаційний інститут»", WD_ALIGN_PARAGRAPH_CENTER, 14)
        add_paragraph(
            "Факультет радіотехніки, комп’ютерних систем і інфокомунікацій",
            WD_ALIGN_PARAGRAPH_CENTER,
            14,
        )
        add_paragraph(
            "Кафедра інформаційно-комунікаційних технологій ім. О. О. Зеленського",
            WD_ALIGN_PARAGRAPH_CENTER,
            14,
        )
        add_blank_line()
        add_paragraph("Лабораторна робота №9", WD_ALIGN_PARAGRAPH_CENTER, 14, True)
        add_paragraph("з дисципліни «Основи теорії цифрового зв’язку»", WD_ALIGN_PARAGRAPH_CENTER, 14)
        add_blank_line()
        add_paragraph(
            "на тему: «Реалізація алгоритму H.264 (MPEG-4 Part 10) для стиснення "
            "відеозображень за допомогою мови програмування Python»",
            WD_ALIGN_PARAGRAPH_CENTER,
            14,
        )
        add_blank_line()
        add_blank_line()
        add_paragraph("Виконав: студент 3 курсу групи № 526-СТ", WD_ALIGN_PARAGRAPH_LEFT, 14)
        add_paragraph("напряму підготовки (спеціальності)", WD_ALIGN_PARAGRAPH_LEFT, 14)
        add_paragraph("172 Телекомунікації та радіотехніка", WD_ALIGN_PARAGRAPH_LEFT, 14)
        add_paragraph("Чернецький Денис Олександрович", WD_ALIGN_PARAGRAPH_LEFT, 14)
        add_blank_line()
        add_paragraph("Прийняв: доц. В'юницький О.Г.", WD_ALIGN_PARAGRAPH_LEFT, 14)
        add_paragraph("Національна шкала: __________", WD_ALIGN_PARAGRAPH_LEFT, 14)
        add_paragraph("Кількість балів: _____", WD_ALIGN_PARAGRAPH_LEFT, 14)
        add_paragraph("Оцінка: ECTS _______", WD_ALIGN_PARAGRAPH_LEFT, 14)

        doc.Repaginate()
        while doc.ComputeStatistics(WD_STATISTIC_PAGES) < 2:
            add_blank_line()
            doc.Repaginate()

        selection.EndKey(WD_STORY)
        add_paragraph("Харків – 2026", WD_ALIGN_PARAGRAPH_CENTER, 14)
        selection.InsertBreak(WD_PAGE_BREAK)

        add_paragraph("МЕТА РОБОТИ", WD_ALIGN_PARAGRAPH_CENTER, 14, True)
        add_blank_line()
        add_body(
            "Ознайомитися з принципами міжкадрового кодування відеоданих, "
            "реалізувати базові етапи алгоритму H.264/MPEG-подібного стиснення "
            "засобами Python та оцінити ефективність компенсації руху за метрикою "
            "кількості біт на піксель."
        )
        add_blank_line()

        add_paragraph("ХІД РОБОТИ", WD_ALIGN_PARAGRAPH_CENTER, 14, True)
        add_blank_line()
        add_body(
            "У ході виконання лабораторної роботи було реалізовано зчитування двох "
            "сусідніх кадрів з відеофайлу, сегментацію кадру на блоки розміром 16x16, "
            "побудову зони пошуку в опорному кадрі та пошук найбільш схожих блоків "
            "за критерієм MAD. На основі знайдених блоків сформовано прогнозований "
            "кадр, обчислено залишковий кадр та виконано реконструкцію цільового кадру."
        )
        add_body(
            "Для оцінювання ефективності міжкадрового кодування було порівняно три "
            "варіанти представлення даних: початковий кадр, абсолютну різницю між "
            "сусідніми кадрами та залишковий кадр після компенсації руху. Для "
            "кожного з них розраховано середню кількість біт на піксель окремо для "
            "компонент R, G, B та для сумарного RGB-подання."
        )
        add_blank_line()

        add_paragraph("ОСНОВНІ РЕЗУЛЬТАТИ", WD_ALIGN_PARAGRAPH_CENTER, 14, True)
        add_blank_line()
        add_body(
            "Для демонстрації роботи алгоритму було використано сусідні кадри №2585 "
            "та №2586 з відеофайлу sample4.avi. Під час перевірки встановлено, що "
            "реконструйований кадр повністю збігається з цільовим кадром, а "
            "коефіцієнт стиску за оцінкою bits/pixel становить 75.8199."
        )
        add_blank_line()

        report_table = doc.Tables.Add(selection.Range, len(METRICS) + 1, 2)
        report_table.Borders.Enable = True
        report_table.Cell(1, 1).Range.Text = "Параметр"
        report_table.Cell(1, 2).Range.Text = "Значення"
        report_table.Rows(1).Range.Bold = True

        for row_index, (key, value) in enumerate(METRICS.items(), start=2):
            report_table.Cell(row_index, 1).Range.Text = key
            report_table.Cell(row_index, 2).Range.Text = value

        report_table.Range.Font.Name = "Times New Roman"
        report_table.Range.Font.Size = 12
        selection.MoveDown()
        add_blank_line()

        for figure in FIGURES:
            add_figure(figure["text"], figure["file"], figure["caption"])

        add_paragraph("ВИСНОВКИ", WD_ALIGN_PARAGRAPH_CENTER, 14, True)
        add_blank_line()
        add_body(
            "У лабораторній роботі реалізовано базову схему міжкадрового "
            "відеокодування з компенсацією руху, яка включає пошук подібних блоків, "
            "побудову прогнозованого кадру, формування залишкового сигналу та "
            "реконструкцію зображення. Отримані результати показали, що після "
            "компенсації руху енергетика залишкового кадру істотно зменшується "
            "порівняно з початковим кадром і навіть зі звичайною різницею між кадрами."
        )
        add_body(
            "Для досліджуваної пари кадрів сумарна оцінка bits/pixel зменшилась з "
            "18.6566 для початкового кадру до 0.2461 для залишкового кадру, що "
            "відповідає коефіцієнту стиску 75.8199. Відновлений кадр повністю "
            "збігається з цільовим, отже реалізація алгоритму кодування та "
            "декодування є коректною."
        )

        if DOCX_PATH.exists():
            DOCX_PATH.unlink()
        if PDF_PATH.exists():
            PDF_PATH.unlink()

        doc.SaveAs2(str(DOCX_PATH))
        doc.ExportAsFixedFormat(str(PDF_PATH), WD_FORMAT_PDF)
        doc.Close(False)
        word.Quit()
        doc = None
        word = None
        return DOCX_PATH, PDF_PATH
    finally:
        if doc is not None:
            doc.Close(False)
        if word is not None:
            word.Quit()
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    docx_path, pdf_path = build_report()
    print(f"Created: {docx_path}")
    print(f"Created: {pdf_path}")
