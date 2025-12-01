from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(
        6, help="Максимум числовых колонок для гистограмм."
    ),
    top_k_categories: int = typer.Option(
        5, help="Сколько top-значений выводить для категориальных признаков."
    ),
    title: str = typer.Option(
        "EDA-отчёт", help="Заголовок markdown-отчёта."
    ),
    min_missing_share: float = typer.Option(
        0.3, help="Порог доли пропусков для отметки колонки как проблемной."
    ),
) -> None:
    """
    Полный EDA-отчёт с параметрами кастомизации.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)

    # === Новое: передаём top_k в топ-категории ===
    top_cats_all = top_categories(df, top_k=top_k_categories)

    # 2. Качество
    quality_flags = compute_quality_flags(summary, missing_df)

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)

    # === Новое: сохраняем top-k ===
    save_top_categories_tables(top_cats_all, out_root / "top_categories")

    # === Новое: определяем проблемные колонки ===
    problematic = []
    if not missing_df.empty:
        for col, row in missing_df.iterrows():
            if row["missing_share"] >= min_missing_share:
                problematic.append(col)

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Порог проблемных колонок по пропускам: **{min_missing_share:.2%}**\n")
        f.write(f"- Максимум гистограмм: **{max_hist_columns}**\n")
        f.write(f"- top-k категорий: **{top_k_categories}**\n\n")

        f.write("## Проблемные колонки по пропускам\n\n")
        if not problematic:
            f.write("Нет колонок, превышающих порог.\n\n")
        else:
            for col in problematic:
                f.write(f"- {col}\n")
            f.write("\n")

        f.write("## Колонки\nСм. `summary.csv`.\n\n")

        f.write("## Пропуски\n")
        if missing_df.empty:
            f.write("Датасет не содержит пропусков.\n\n")
        else:
            f.write("См. `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляции\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n")
        if not top_cats_all:
            f.write("Категориальных колонок нет.\n\n")
        else:
            f.write(
                f"Сохранены топ-{top_k_categories} значений в каталоге "
                "`top_categories/`.\n\n"
            )

        f.write("## Гистограммы\n")
        f.write(
            f"Сохранено до {max_hist_columns} числовых гистограмм. "
            "Файлы `hist_*.png`.\n"
        )

    # 5. Графики
    # === Новое: передаём max_hist_columns ===
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)

    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в: {out_root}")


if __name__ == "__main__":
    app()
