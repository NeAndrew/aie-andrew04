from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    n_zeros: int = 0 
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            n_zeros = int((s == 0).sum())
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                n_zeros=n_zeros,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значений.
    Возвращает словарь: колонка -> DataFrame со столбцами value/count/share.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


def compute_quality_flags(summary: DatasetSummary, missing_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Простейшие эвристики «качества» данных:
    - слишком много пропусков;
    - подозрительно мало строк;
    - наличие константных колонок;
    - категориальные признаки с огромным числом уникальных значений;
    - дубликаты в потенциальных ID-полях;
    - избыток нулей в числовых колонках.
    """
    flags: Dict[str, Any] = {}
    n_rows = summary.n_rows

    # --- Базовые проверки ---
    flags["too_few_rows"] = n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    # Эвристики

    # 1. Есть ли колонки, где все значения одинаковые
    # Такие колонки бесполезны для анализа или моделирования — просто шум.
    # Считаем колонку константной, если уникальных значений ≤ 1 и при этом есть хотя бы одно непустое значение.
    has_constant_columns = False
    for col in summary.columns:
        if col.non_null > 0 and col.unique <= 1:
            has_constant_columns = True
            break
    flags["has_constant_columns"] = has_constant_columns

    # 2. Есть ли категориальные колонки с подозрительно высокой кардинальностью
    # Например, если у признака почти столько же уникальных значений, сколько строк —
    # это может быть ошибка или бесполезный шум.
    # Порог - кардинальность > 90% от числа строк И уникальных значений > 100.
    has_high_cardinality_categoricals = False
    high_cardinality_threshold_ratio = 0.9
    high_cardinality_min_unique = 100
    for col in summary.columns:
        if not col.is_numeric and col.non_null > 0:
            cardinality_ratio = col.unique / n_rows if n_rows > 0 else 0
            if col.unique >= high_cardinality_min_unique and cardinality_ratio > high_cardinality_threshold_ratio:
                has_high_cardinality_categoricals = True
                break
    flags["has_high_cardinality_categoricals"] = has_high_cardinality_categoricals

    # 3. Есть ли дубликаты в поле, похожем на ID
    # Мы не можем точно знать, какое поле — ID, но если имя колонки содержит 'id', то это явный признак
    has_suspicious_id_duplicates = False
    for col in summary.columns:
        # Ищем колонки, похожие на ID по названию 
        if "id" in col.name.lower():
            # Если все значения уникальны — OK. Если нет, и при этом есть данные — тревога.
            if col.non_null > 0 and col.unique < col.non_null:
                has_suspicious_id_duplicates = True
                break
    flags["has_suspicious_id_duplicates"] = has_suspicious_id_duplicates

    # 4. Много нулей в числовых колонках ←←← ИСПРАВЛЕНО
    has_many_zero_values = False
    zero_ratio_threshold = 0.93
    for col in summary.columns:
        if col.is_numeric and n_rows > 0:
            zero_ratio = col.n_zeros / n_rows 
            if zero_ratio > zero_ratio_threshold:
                has_many_zero_values = True
                break
    flags["has_many_zero_values"] = has_many_zero_values

    # Проверка качества
    # Начинаем с идеального балла
    score = 1.0
    score -= max_missing_share  # штраф за пропуски
    if flags["too_few_rows"]:
        score -= 0.2
    if flags["too_many_columns"]:
        score -= 0.1
    if flags["has_constant_columns"]:
        score -= 0.1
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.05
    if flags["has_suspicious_id_duplicates"]:
        score -= 0.15  
    if flags["has_many_zero_values"]:
        score -= 0.05

    # Ограничиваем скор в [0, 1]
    flags["quality_score"] = max(0.0, min(1.0, score))

    return flags

def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)
