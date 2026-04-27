"""
Конвертация PDF из папки source в pandas DataFrame через docling, сохранение в result/ с префиксом converted_.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pandas as pd
from docling.document_converter import DocumentConverter

# Корень проекта — рядом с этим скриптом
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "source"
RESULT_DIR = PROJECT_ROOT / "result"

# Настройки трансформации данных
RENAME_MAP = {
    "Географические координаты.Широта.Привязка к другим географическим объектам": "DDD Широта",
    "Географические координаты.Долгота.Привязка к другим географическим объектам": "DDD Долгота",
    "Административно территориальная ( муниципальная привязка.Административно территориальная ( муниципальная привязка.Административно территориальная ( муниципальная привязка": "АТД",
    "Тип объекта.Тип объекта.Тип объекта": "Тип объекта",
    "Название географического объекта.Название географического объекта.Название географического объекта": "Название географического объекта"
}

FINAL_COLUMNS = [
    "Рег.номер",
    "Название",
    "Тип объекта",
    "АТД",
    "Широта",
    "Долгота",
    "Лист",
]

TEMP_DUP_SUFFIX = "__TMP_DUPCOL__"


def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        counts: dict[str, int] = {}
        unique_cols = []
        for col in map(str, df.columns):
            current_count = counts.get(col, 0)
            unique_cols.append(col if current_count == 0 else f"{col}{TEMP_DUP_SUFFIX}{current_count}")
            counts[col] = current_count + 1
        df.columns = pd.Index(unique_cols)
    return df


def concat_raw_tables(raw_tables: list[pd.DataFrame]) -> pd.DataFrame:
    prepared = [make_columns_unique(df.copy()) for df in raw_tables]
    return pd.concat(prepared, ignore_index=True)


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = pd.Index([str(col).split(TEMP_DUP_SUFFIX)[0] for col in df.columns])
    df.columns = pd.Index([str(col).replace("\n", " ").strip() for col in df.columns])

    if df.columns.duplicated().any():
        unique_cols = []
        seen = set()
        for col in df.columns:
            if col not in seen:
                unique_cols.append(col)
                seen.add(col)

        new_data: dict[object, pd.Series] = {}
        for col in unique_cols:
            subset = df.loc[:, df.columns == col]
            if subset.shape[1] > 1:
                new_data[col] = subset.bfill(axis=1).iloc[:, 0]
            else:
                new_data[col] = subset.iloc[:, 0]
        df = pd.DataFrame(new_data, index=df.index)

    df = df.rename(columns=RENAME_MAP)

    if df.columns.duplicated().any():
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            n = int(sum(cols == dup))
            cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup for i in range(n)]
        df.columns = cols

    if hasattr(df, "map"):
        df = df.map(lambda x: x.replace("\n", " ") if isinstance(x, str) else x)
    else:
        df = df.applymap(lambda x: x.replace("\n", " ") if isinstance(x, str) else x)

    df = df.iloc[2:].reset_index(drop=True)

    if df.shape[1] > len(FINAL_COLUMNS):
        df = df.iloc[:, : len(FINAL_COLUMNS)]
    elif df.shape[1] < len(FINAL_COLUMNS):
        for i in range(df.shape[1], len(FINAL_COLUMNS)):
            df[i] = pd.NA

    df.columns = FINAL_COLUMNS

    def normalize_coord_value(value) -> object:
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

    def convert_coord_to_dd(value) -> object:
        if pd.isna(value):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return value
        coord = value.strip().replace("º", "°")
        if not coord:
            return pd.NA
        dm_match = re.match(
            r"^([+-]?\d+(?:\.\d+)?)(?:\s*°\s*|\s+)(\d+(?:\.\d+)?)\s*'?$", coord
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

    def format_coord_with_dot(value) -> object:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (int, float)):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        if isinstance(value, str):
            return value.replace(",", ".")
        return value

    for coord_col in ("Широта", "Долгота"):
        if coord_col in df.columns:
            df[coord_col] = (
                df[coord_col]
                .map(normalize_coord_value)
                .map(convert_coord_to_dd)
                .map(format_coord_with_dot)
            )

    def normalize_sheet_code(value) -> object:
        if pd.isna(value):
            return pd.NA
        raw = str(value).strip()
        if not raw:
            return pd.NA
        cleaned = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]", "", raw)
        if len(cleaned) < 6:
            return pd.NA
        token = cleaned[:6]
        return f"{token[0]}-{token[1:3]}-{token[3:6]}"

    if "Лист" in df.columns:
        df["Лист"] = df["Лист"].map(normalize_sheet_code)

    return df


def export_table_without_header(table, doc) -> pd.DataFrame:
    export_fn = table.export_to_dataframe
    signature = inspect.signature(export_fn)
    if "header" in signature.parameters:
        return export_fn(doc, header=None)
    df = export_fn(doc)
    df.columns = pd.RangeIndex(start=0, stop=df.shape[1], step=1)
    return df


def extract_tables_dataframe(doc) -> pd.DataFrame | None:
    if not (hasattr(doc, "tables") and doc.tables):
        return None
    all_dfs: list[pd.DataFrame] = []
    for table in doc.tables:
        all_dfs.append(export_table_without_header(table, doc))
    if not all_dfs:
        return None
    return concat_raw_tables(all_dfs)


def pdf_to_dataframe(
    pdf_path: Path, converter: DocumentConverter | None = None
) -> pd.DataFrame:
    """
    Открывает PDF, извлекает таблицы через docling, объединяет в один DataFrame и применяет transform_dataframe.
    """
    if converter is None:
        converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    combined = extract_tables_dataframe(doc)
    if combined is None:
        return pd.DataFrame()
    return transform_dataframe(combined)


def save_converted_dataframe(df: pd.DataFrame, original_stem: str, result_dir: Path) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    out = result_dir / f"converted_{original_stem}.xlsx"
    df.to_excel(out, sheet_name="MergedData", index=False)
    return out


def main() -> None:
    pdf_files = sorted(SOURCE_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"В папке {SOURCE_DIR} нет PDF файлов.")
        return

    print(f"Найдено PDF: {len(pdf_files)}")

    try:
        document_converter = DocumentConverter()
    except Exception as e:
        print(f"Ошибка инициализации DocumentConverter: {e}")
        sys.exit(1)

    for pdf_path in pdf_files:
        print(f"Обработка: {pdf_path.name}…")
        try:
            df = pdf_to_dataframe(pdf_path, converter=document_converter)
            if df.empty:
                print(f"  Таблицы не извлечены или файл пуст: {pdf_path.name}")
                continue
            out_path = save_converted_dataframe(df, pdf_path.stem, RESULT_DIR)
            print(f"  Сохранено: {out_path.name} (строк: {len(df)}, столбцов: {len(df.columns)})")
        except Exception as e:
            print(f"  Ошибка: {e}")

    print("Готово.")


if __name__ == "__main__":
    main()
