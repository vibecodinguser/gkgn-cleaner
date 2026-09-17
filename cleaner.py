import inspect
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from docling.document_converter import DocumentConverter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "source"
RESULT_DIR = PROJECT_ROOT / "result"
FINAL_COLUMN_NAMES: list[str] = [
    "Регистрационный номер",
    "Название географического объекта",
    "Тип объекта",
    "Административно-территориальная (муниципальная привязка)",
    "Широта",
    "Долгота",
    "Номенклатура листа карты",
]

DMM_REGEX = re.compile(
    r"([+-]?\d+(?:[.,]\d+)?)(?:\s*[°º]\s*|\s+)(\d+(?:[.,]\d+)?)\s*[\'′’]?",
    re.UNICODE,
)

SHEET_CODE_X_XX_XXX = re.compile(
    r"([0-9A-Za-zА-Яа-яЁё])-([0-9A-Za-zА-Яа-яЁё]{2})-([0-9A-Za-zА-Яа-яЁё]{3})",
    re.UNICODE,
)


def _cell_to_str(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, bool):
        return str(value)
    return str(value).strip()


def _normalize_dmm_text(s: str) -> str:
    return (
        s.replace("º", "°")
        .replace("′", "'")
        .replace("’", "'")
        .replace(",", ".")
    )


def _extract_coord_as_decimal(value: Any) -> Any:
    s = _cell_to_str(value)
    match = DMM_REGEX.search(_normalize_dmm_text(s))
    if not match:
        return pd.NA
    degrees = float(match.group(1).replace(",", "."))
    minutes = float(match.group(2).replace(",", "."))
    decimal = abs(degrees) + minutes / 60
    if degrees < 0:
        decimal = -decimal
    return f"{decimal:.6f}".rstrip("0").rstrip(".")


def _cell_contains_coordinate(value: Any) -> bool:
    return DMM_REGEX.search(_normalize_dmm_text(_cell_to_str(value))) is not None


def _cell_starts_with_sheet_code(value: Any) -> bool:
    s = _cell_to_str(value)
    return bool(s) and SHEET_CODE_X_XX_XXX.match(s) is not None


def _clean_cell_if_recognized(value: Any) -> Any:
    if value is None or value is pd.NA:
        return value
    if _cell_contains_coordinate(value):
        return _extract_coord_as_decimal(value)
    s = _cell_to_str(value)
    sheet_match = SHEET_CODE_X_XX_XXX.search(s)
    if sheet_match:
        return f"{sheet_match.group(1)}-{sheet_match.group(2)}-{sheet_match.group(3)}"
    return value


def clean_cells(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.map(_clean_cell_if_recognized)


def _column_mostly_matches(series: pd.Series, predicate) -> bool:
    sample = series.dropna().head(8)
    if sample.empty:
        return False
    hits = sum(bool(predicate(value)) for value in sample)
    return hits / len(sample) >= 0.6


def _missing_admin_column_layout(df: pd.DataFrame) -> bool:
    """Six-column tables: reg, name, type, lat, lon, sheet (no admin binding)."""
    if df.shape[1] != 6:
        return False
    lat_col = df.iloc[:, 3]
    lon_col = df.iloc[:, 4]
    sheet_col = df.iloc[:, 5]
    return (
        _column_mostly_matches(lat_col, _cell_contains_coordinate)
        and _column_mostly_matches(lon_col, _cell_contains_coordinate)
        and _column_mostly_matches(sheet_col, _cell_starts_with_sheet_code)
    )


def _insert_missing_admin_column(df: pd.DataFrame) -> pd.DataFrame:
    if not _missing_admin_column_layout(df):
        return df
    out = df.copy()
    out.insert(3, "_admin_binding", None)
    return out


def drop_first_two_rows_and_set_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FINAL_COLUMN_NAMES)
    out = df.iloc[2:].reset_index(drop=True)
    out = _insert_missing_admin_column(out)
    n = len(FINAL_COLUMN_NAMES)
    if out.shape[1] < n:
        for i in range(out.shape[1], n):
            out[i] = None
    elif out.shape[1] > n:
        out = out.iloc[:, :n]
    out.columns = FINAL_COLUMN_NAMES
    return out


def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        counts: dict[str, int] = {}
        unique_cols = []
        for col in map(str, df.columns):
            current_count = counts.get(col, 0)
            unique_cols.append(
                col if current_count == 0 else f"{col}__TMP_DUPCOL__{current_count}"
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


def _cell_starts_with_digit(value: Any) -> bool:
    if value is None or value is pd.NA or isinstance(value, bool):
        return False
    text = str(value).strip()
    return bool(text) and text[0].isdigit()


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
    assert converter is not None
    result = converter.convert(pdf_path)
    doc = result.document
    combined = extract_tables_dataframe(doc)
    if combined is None:
        return pd.DataFrame()
    with_headers = drop_first_two_rows_and_set_column_headers(combined)
    return clean_cells(with_headers)


def save_converted_dataframe(df: pd.DataFrame, original_stem: str, result_dir: Path) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    out = result_dir / f"converted_{original_stem}.xlsx"
    df.to_excel(out, sheet_name="Data", index=False)
    return out


def _silence_third_party_output() -> None:
    import docling_ibm_models.tableformer.data_management.matching_post_processor as _mpp
    from rapidocr.utils.log import logger as _rapidocr_logger
    from transformers.utils.logging import disable_progress_bar

    _mpp.LOG_LEVEL = logging.CRITICAL
    _rapidocr_logger.setLevel(logging.CRITICAL)
    _rapidocr_logger.handlers.clear()
    disable_progress_bar()

    loggers = [
        "MatchingPostProcessor",
        "docling",
        "docling_core",
        "docling_ibm_models",
        "transformers",
        "torch",
        "urllib3",
        "huggingface_hub",
    ]
    for name in loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.handlers.clear()
        logger.propagate = False


def main() -> None:
    _silence_third_party_output()

    pdf_files = sorted(SOURCE_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"В папке {SOURCE_DIR} нет PDF файлов.")
        return

    try:
        document_converter = DocumentConverter()
    except Exception as e:
        print(f"Ошибка инициализации DocumentConverter: {e}")
        sys.exit(1)

    print("\n")
    progress = tqdm(pdf_files, desc="Конвертация PDF", unit="file", leave=True)
    for pdf_path in progress:
        try:
            df = pdf_to_dataframe(pdf_path, converter=document_converter)
            if df.empty:
                progress.write(f"  Таблицы не извлечены или файл пуст: {pdf_path.name}")
                continue
            save_converted_dataframe(df, pdf_path.stem, RESULT_DIR)
        except Exception as e:
            progress.write(f"  Ошибка: {e}")

    print("Готово.")


if __name__ == "__main__":
    main()