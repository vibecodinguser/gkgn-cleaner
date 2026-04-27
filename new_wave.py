"""
PDF из папки source → pandas DataFrame через docling → сохранение в result/ с префиксом converted_.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pandas as pd
from docling.document_converter import DocumentConverter

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "source"
RESULT_DIR = PROJECT_ROOT / "result"

TEMP_DUP_SUFFIX = "__TMP_DUPCOL__"

# Колонки 5 и 6 в нумерации «как в Excel» (1-based) → индексы 4 и 5
DMM_COL5_POS = 4
DMM_COL6_POS = 5
# Колонка 7 (1-based) → индекс 6: код листа X-XX-XXX
SHEET_CODE_COL_POS = 6

FINAL_COLUMN_NAMES: list[str] = [
    "Регистрационный номер",
    "Название географического объекта",
    "Тип объекта",
    "Административно-территориальная (муниципальная привязка)",
    "Широта",
    "Долгота",
    "Номенклатура листа карты масштаба 1:100 000",
]

# Градусы и минуты (DMM): допускаются °/º, пробел или ° между частями, необязательная '
DMM_REGEX = re.compile(
    r"([+-]?\d+(?:[.,]\d+)?)(?:\s*[°º]\s*|\s+)(\d+(?:[.,]\d+)?)\s*(?:[\'′’])?",
    re.UNICODE,
)

# Один символ, дефис, два символа, дефис, три символа (буквы/цифры латиница и кириллица)
SHEET_CODE_X_XX_XXX = re.compile(
    r"([0-9A-Za-zА-Яа-яЁё])-([0-9A-Za-zА-Яа-яЁё]{2})-([0-9A-Za-zА-Яа-яЁё]{3})",
    re.UNICODE,
)


def _cell_to_str(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


def _normalize_dmm_text(s: str) -> str:
    return (
        s.replace("º", "°")
        .replace("′", "'")
        .replace("’", "'")
        .replace(",", ".")
    )


def _canonical_dmm(m: re.Match) -> str:
    deg = m.group(1).replace(",", ".")
    mins = m.group(2).replace(",", ".")
    return f"{deg}° {mins}'"


def _find_all_dmm(text: str) -> list[re.Match]:
    if not text:
        return []
    normalized = _normalize_dmm_text(text)
    return list(DMM_REGEX.finditer(normalized))


def _clean_col5_col6_dmm(value5: object, value6: object) -> tuple[object, object]:
    """
    Кол. 5: оставить только DMM; если два вхождения — второе в кол. 6.
    Кол. 6: если второе не пришло из кол. 5 — очистить одно DMM в ячейке.
    """
    s5 = _cell_to_str(value5)
    matches5 = _find_all_dmm(s5)
    second_for_col6: object | None = None

    if len(matches5) >= 2:
        new5: object = _canonical_dmm(matches5[0])
        second_for_col6 = _canonical_dmm(matches5[1])
    elif len(matches5) == 1:
        new5 = _canonical_dmm(matches5[0])
    else:
        new5 = pd.NA

    if second_for_col6 is not None:
        new6: object = second_for_col6
    else:
        s6 = _cell_to_str(value6)
        m6 = _find_all_dmm(s6)
        new6 = _canonical_dmm(m6[0]) if m6 else pd.NA

    return new5, new6


def clean_dmm_columns_5_and_6(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.shape[1] <= DMM_COL5_POS:
        return df
    out = df.copy()
    while out.shape[1] <= DMM_COL6_POS:
        out[out.shape[1]] = pd.NA

    new5_list: list[object] = []
    new6_list: list[object] = []
    for i in range(len(out)):
        v5 = out.iat[i, DMM_COL5_POS]
        v6 = out.iat[i, DMM_COL6_POS]
        n5, n6 = _clean_col5_col6_dmm(v5, v6)
        new5_list.append(n5)
        new6_list.append(n6)

    out.iloc[:, DMM_COL5_POS] = new5_list
    out.iloc[:, DMM_COL6_POS] = new6_list
    return out


def _extract_sheet_code_x_xx_xxx(value: object) -> object:
    s = _cell_to_str(value)
    if not s:
        return pd.NA
    m = SHEET_CODE_X_XX_XXX.search(s)
    if not m:
        return pd.NA
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def clean_column_7_sheet_code(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.shape[1] <= SHEET_CODE_COL_POS:
        return df
    out = df.copy()
    col = out.iloc[:, SHEET_CODE_COL_POS]
    out.iloc[:, SHEET_CODE_COL_POS] = col.map(_extract_sheet_code_x_xx_xxx)
    return out


def drop_first_two_rows_and_set_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет первые 2 строки данных и задаёт имена семи колонок."""
    n = len(FINAL_COLUMN_NAMES)
    if df.empty:
        return pd.DataFrame(columns=FINAL_COLUMN_NAMES)
    out = df.iloc[2:].copy().reset_index(drop=True)
    while out.shape[1] < n:
        out[out.shape[1]] = pd.NA
    if out.shape[1] > n:
        out = out.iloc[:, :n].copy()
    out.columns = pd.Index(FINAL_COLUMN_NAMES)
    return out


def _normalize_coord_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    cleaned = value
    if "'" in cleaned:
        left = cleaned.split("'", 1)[0]
        cleaned = f"{left}'"
    cleaned = re.sub(r"[A-Za-zА-Яа-яЁё]", "", cleaned)
    cleaned = cleaned.replace(",", ".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _convert_coord_to_decimal_degrees(value: object) -> object:
    if pd.isna(value):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return value
    coord = value.strip().replace("º", "°")
    if not coord:
        return pd.NA
    dm_match = re.match(
        r"^([+-]?\d+(?:\.\d+)?)(?:\s*°\s*|\s+)(\d+(?:\.\d+)?)\s*'?$",
        coord,
    )
    if dm_match:
        degrees = float(dm_match.group(1))
        minutes = float(dm_match.group(2))
        sign = -1 if degrees < 0 else 1
        decimal_degrees = sign * (abs(degrees) + minutes / 60)
        return round(decimal_degrees, 6)
    dd_match = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*°?\s*'?$", coord)
    if dd_match:
        return round(float(dd_match.group(1)), 6)
    return value


def _format_decimal_degree_string(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    if isinstance(value, str):
        return value.replace(",", ".")
    return value


def convert_lat_lon_columns_to_decimal_degrees(df: pd.DataFrame) -> pd.DataFrame:
    """Колонки «Широта» и «Долгота»: DMM или градусы → десятичные градусы (строка с точкой)."""
    if df.empty:
        return df
    out = df.copy()
    for col in ("Широта", "Долгота"):
        if col not in out.columns:
            continue
        out[col] = (
            out[col]
            .map(_normalize_coord_value)
            .map(_convert_coord_to_decimal_degrees)
            .map(_format_decimal_degree_string)
        )
    return out


def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        counts: dict[str, int] = {}
        unique_cols = []
        for col in map(str, df.columns):
            current_count = counts.get(col, 0)
            unique_cols.append(
                col if current_count == 0 else f"{col}{TEMP_DUP_SUFFIX}{current_count}"
            )
            counts[col] = current_count + 1
        df.columns = pd.Index(unique_cols)
    return df


def concat_raw_tables(raw_tables: list[pd.DataFrame]) -> pd.DataFrame:
    prepared = [make_columns_unique(df.copy()) for df in raw_tables]
    return pd.concat(prepared, ignore_index=True)


def export_table_without_header(table, doc) -> pd.DataFrame:
    export_fn = table.export_to_dataframe
    signature = inspect.signature(export_fn)
    if "header" in signature.parameters:
        return export_fn(doc, header=None)
    df = export_fn(doc)
    df.columns = pd.RangeIndex(start=0, stop=df.shape[1], step=1)
    return df


def _cell_starts_with_digit(value: object) -> bool:
    """True, если первая значащая позиция в ячейке — цифра (шапки и текст без номера отбрасываются)."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        text = str(value).strip()
        return bool(text) and text[0].isdigit()
    text = str(value).strip()
    if not text:
        return False
    return text[0].isdigit()


def filter_rows_first_col_starts_with_digit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.shape[1] == 0:
        return df
    first = df.columns[0]
    mask = df[first].map(_cell_starts_with_digit)
    return df.loc[mask].reset_index(drop=True)


def extract_tables_dataframe(doc) -> pd.DataFrame | None:
    if not (hasattr(doc, "tables") and doc.tables):
        return None
    all_dfs: list[pd.DataFrame] = []
    for table in doc.tables:
        df = export_table_without_header(table, doc)
        df = filter_rows_first_col_starts_with_digit(df)
        if not df.empty:
            all_dfs.append(df)
    if not all_dfs:
        return None
    return concat_raw_tables(all_dfs)


def pdf_to_dataframe(
    pdf_path: Path, converter: DocumentConverter | None = None
) -> pd.DataFrame:
    if converter is None:
        converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    combined = extract_tables_dataframe(doc)
    if combined is None:
        return pd.DataFrame()
    cleaned = clean_dmm_columns_5_and_6(combined)
    cleaned = clean_column_7_sheet_code(cleaned)
    with_headers = drop_first_two_rows_and_set_column_headers(cleaned)
    return convert_lat_lon_columns_to_decimal_degrees(with_headers)


def save_converted_dataframe(df: pd.DataFrame, original_stem: str, result_dir: Path) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    out = result_dir / f"converted_{original_stem}.xlsx"
    df.to_excel(out, sheet_name="Data", index=False)
    return out


def main() -> None:
    pdf_files = sorted(SOURCE_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"В папке {SOURCE_DIR} нет PDF файлов.")
        return

    total = len(pdf_files)
    print(f"Найдено PDF: {total} (обработка по одному файлу, по очереди)")

    try:
        document_converter = DocumentConverter()
    except Exception as e:
        print(f"Ошибка инициализации DocumentConverter: {e}")
        sys.exit(1)

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{index}/{total}] Обработка: {pdf_path.name}…")
        try:
            df = pdf_to_dataframe(pdf_path, converter=document_converter)
            if df.empty:
                print(f"  Таблицы не извлечены или файл пуст: {pdf_path.name}")
                continue
            out_path = save_converted_dataframe(df, pdf_path.stem, RESULT_DIR)
            print(
                f"  Сохранено: {out_path.name} (строк: {len(df)}, столбцов: {len(df.columns)})"
            )
        except Exception as e:
            print(f"  Ошибка: {e}")

    print("Готово.")


if __name__ == "__main__":
    main()
