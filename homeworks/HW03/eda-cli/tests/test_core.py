from __future__ import annotations
import os
import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

def test_quality_flags_on_example_csv():
    filepath = "data/example.csv"
    assert os.path.exists(filepath), f"Файл {filepath} не найден"

    df = pd.read_csv(filepath)

    assert "user_id" in df.columns

    # Дубликат user_id = 1025 (2 раза)
    assert df["user_id"].duplicated().any(), "Ожидался дубликат в user_id"

    # Запускаем анализ
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # Проверяем флаги
    assert flags["has_suspicious_id_duplicates"] is True, \
        "Флаг has_suspicious_id_duplicates должен быть True из-за дубликата в user_id"
    # Остальные флаги должны быть False (в этом датасете нет таких проблем)
    assert flags["has_constant_columns"] is False, \
        "Нет константных колонок"
    assert flags["has_high_cardinality_categoricals"] is False, \
        "Нет категориальных признаков с кардинальностью >100 и >90%"
    assert flags["has_many_zero_values"] is False, \
        "Нет числовых колонок с >90% нулей (revenue и purchases имеют много нулей, но не >90%)"

    # Качество в [0,1]
    assert 0.0 <= flags["quality_score"] <= 1.0

def test_quality_flags_on_books_csv():
    filepath = "data/books.csv"
    assert os.path.exists(filepath), f"Файл {filepath} не найден"

    df = pd.read_csv(filepath)

    # Проверим, что данные загружены
    assert len(df) > 0
    assert "book_id" in df.columns
    assert "isbn" in df.columns
    assert "average_rating" in df.columns

    # Запускаем анализ
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_high_cardinality_categoricals"] is True, \
        "Флаг has_high_cardinality_categoricals должен быть True из-за уникальных isbn/authors"

    # Остальные флаги — должны быть False
    assert flags["has_constant_columns"] is False, "Нет константных колонок"
    assert flags["has_suspicious_id_duplicates"] is False, "Нет дубликатов в колонках с 'id' (или все уникальны)"
    assert flags["has_many_zero_values"] is False, "Нет числовых колонок, заполненных >90% нулями"

    assert 0.0 <= flags["quality_score"] <= 1.0

def test_level_impossible():
    n = 200

    df = pd.DataFrame({
        # 1) Константная колонка
        "all_equal": [42] * n,

        # 2) Категориальная колонка с высокой кардинальностью
        "category_big": [f"cat_{i}" for i in range(195)] + ["cat_x"] * 5,

        # 3) Колонка, где почти все значения нули
        "zeros95": [0] * 195 + list(range(5)),

        # 4) Идентификатор с умышленными дубликатами
        "session_id": list(range(199)) + [10],

        # 5) Нормальная колонка
        "value": list(range(n)),
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # ПРОВЕРКА ФЛАГОВ, ЧТО ОНИ ВСЕ FALSE (СПЕЦИАЛЬНО ДЛЯ ДАННОГО ДАТАСЕТА)
    assert flags["has_constant_columns"] is True
    assert flags["has_high_cardinality_categoricals"] is True
    assert flags["has_suspicious_id_duplicates"] is True
    assert flags["has_many_zero_values"] is True
    # Качество не может быть идеальным
    assert flags["quality_score"] < 1.0