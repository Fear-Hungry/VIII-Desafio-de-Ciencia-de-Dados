import datetime
from typing import Any, Dict, Optional

import polars as pl
from typing import List, Optional, Union, Tuple
import logging

# Configuração do logger
logger = logging.getLogger(__name__)


class SymbolCSVParser:
    def __init__(
        self,
        csv_file_path: str,
        symbol_name: str,
        parse_dates_flag: bool,
        time_format_str: Optional[str],
        column_map_resolved: Dict[str, str],
        internal_time_col_name: str,
        original_time_col_name: str,
        internal_column_names_map: Dict[str, str],
        filter_start_date: Optional[datetime.datetime],
        filter_end_date: Optional[datetime.datetime],
        custom_column_dtypes: Optional[Dict[str, Any]],
    ):
        self.csv_file_path = csv_file_path
        self.symbol_name = symbol_name
        self.parse_dates_flag = parse_dates_flag
        self.time_format_str = time_format_str
        self.column_map_resolved = column_map_resolved
        self.internal_time_col_name = internal_time_col_name
        self.original_time_col_name = original_time_col_name
        self.internal_column_names_map = internal_column_names_map
        self.filter_start_date = filter_start_date
        self.filter_end_date = filter_end_date
        self.custom_column_dtypes = custom_column_dtypes or {}

    def _cast_time_col_from_string(self, lazy_df: pl.LazyFrame) -> pl.LazyFrame:
        current_schema = lazy_df.schema
        if (
            self.internal_time_col_name in current_schema
            and current_schema[self.internal_time_col_name] == pl.Utf8
        ):
            if self.time_format_str:
                return lazy_df.with_columns(
                    pl.col(self.internal_time_col_name).str.to_datetime(
                        format=self.time_format_str, strict=False, time_unit="us"
                    )
                )
            else:
                return lazy_df.with_columns(
                    pl.col(self.internal_time_col_name).str.to_datetime(
                        strict=False, time_unit="us"
                    )
                )
        return lazy_df

    def _ensure_datetime_type(self, lazy_df: pl.LazyFrame) -> pl.LazyFrame:
        current_schema = lazy_df.schema
        time_col_dtype = current_schema.get(self.internal_time_col_name)
        if time_col_dtype == pl.Date:
            return lazy_df.with_columns(
                pl.col(self.internal_time_col_name).cast(pl.Datetime(time_unit="us"))
            )
        elif time_col_dtype != pl.Datetime:
            if (
                time_col_dtype != pl.Utf8
            ):  # Utf8 is handled by _cast_time_col_from_string
                logger.warning(
                    f"Coluna de tempo {self.internal_time_col_name} em {self.symbol_name} tem tipo {time_col_dtype} após tentativa de conversão, esperado Datetime[μs]."
                )
        return lazy_df

    def parse(self) -> Optional[pl.LazyFrame]:
        try:
            ldf = pl.scan_csv(
                self.csv_file_path,
                try_parse_dates=self.parse_dates_flag,
                truncate_ragged_lines=True,
            )

            # 1. Renomear colunas para o padrão interno
            rename_mapping = {}
            for csv_col_original_case in ldf.columns:
                csv_col_lower = csv_col_original_case.lower()
                if csv_col_lower in self.column_map_resolved:
                    internal_name = self.column_map_resolved[csv_col_lower]
                    if internal_name != csv_col_original_case:
                        rename_mapping[csv_col_original_case] = internal_name
            if rename_mapping:
                ldf = ldf.rename(rename_mapping)

            # 2. Garantir que a coluna de tempo (com nome interno self.internal_time_col_name) exista
            if self.internal_time_col_name not in ldf.columns:
                original_time_col_input_lower = self.original_time_col_name.lower()
                found_original_time_col = None
                for col_in_ldf in ldf.columns:
                    if col_in_ldf.lower() == original_time_col_input_lower:
                        found_original_time_col = col_in_ldf
                        break
                if (
                    found_original_time_col
                    and found_original_time_col != self.internal_time_col_name
                ):
                    ldf = ldf.rename(
                        {found_original_time_col: self.internal_time_col_name}
                    )
                elif self.internal_time_col_name not in ldf.columns:
                    raise ValueError(
                        f"Coluna de tempo '{self.internal_time_col_name}' (mapeada de '{self.original_time_col_name}') não encontrada no CSV de {self.symbol_name}."
                    )

            # 3. Converter consistentemente a coluna de tempo para pl.Datetime
            ldf = ldf.pipe(self._cast_time_col_from_string)
            ldf = ldf.pipe(self._ensure_datetime_type)

            final_schema = ldf.schema
            if (
                self.internal_time_col_name not in final_schema
                or final_schema[self.internal_time_col_name] != pl.Datetime
            ):
                if (
                    self.internal_time_col_name in final_schema
                    and isinstance(
                        final_schema[self.internal_time_col_name], pl.Datetime
                    )
                    and final_schema[self.internal_time_col_name].time_unit != "us"
                ):
                    ldf = ldf.with_columns(
                        pl.col(self.internal_time_col_name).cast(
                            pl.Datetime(time_unit="us")
                        )
                    )
                elif self.internal_time_col_name not in final_schema:
                    raise ValueError(
                        f"Falha crítica: Coluna de tempo {self.internal_time_col_name} não presente no schema final para {self.symbol_name}. Schema: {final_schema}"
                    )
                elif final_schema[self.internal_time_col_name] != pl.Datetime:
                    raise TypeError(
                        f"Falha crítica: Coluna de tempo {self.internal_time_col_name} para {self.symbol_name} não é Datetime[μs] após conversões. Tipo final: {final_schema[self.internal_time_col_name]}. Schema: {final_schema}"
                    )

            # 3.5 Adicionar colunas internas faltantes (como 'volume' ou 'adj_close') com nulos
            current_ldf_columns_schema = ldf.schema

            cols_to_add_as_null = {}
            vol_col_name = self.internal_column_names_map.get("volume", "volume")
            if vol_col_name not in current_ldf_columns_schema:
                cols_to_add_as_null[vol_col_name] = pl.Float64

            adj_close_col_name = self.internal_column_names_map.get(
                "adj_close", "adj_close"
            )
            if adj_close_col_name not in current_ldf_columns_schema:
                cols_to_add_as_null[adj_close_col_name] = pl.Float64

            if cols_to_add_as_null:
                ldf = ldf.with_columns(
                    [
                        pl.lit(None, dtype=dtype).alias(name)
                        for name, dtype in cols_to_add_as_null.items()
                    ]
                )

            # 4. Aplicar filtro de data
            if self.filter_start_date:
                ldf = ldf.filter(
                    pl.col(self.internal_time_col_name) >= self.filter_start_date
                )
            if self.filter_end_date:
                ldf = ldf.filter(
                    pl.col(self.internal_time_col_name) <= self.filter_end_date
                )

            # 5. Aplicar dtypes customizados para outras colunas
            cols_to_cast = {}
            # Get schema again after adding null columns
            current_schema_after_adds = ldf.schema
            for col_name, target_type in self.custom_column_dtypes.items():
                # Check if column exists and type is different
                if (
                    col_name in current_schema_after_adds
                    and current_schema_after_adds.get(col_name) != target_type
                ):
                    cols_to_cast[col_name] = target_type

            if cols_to_cast:
                ldf = ldf.with_columns(
                    [pl.col(c).cast(t, strict=False) for c, t in cols_to_cast.items()]
                )

            # 6. Selecionar colunas finais
            # Garantir que as colunas mapeadas em internal_column_names_map estejam presentes,
            # e também quaisquer outras colunas que possam ter vindo do CSV e não foram explicitamente mapeadas/removidas.
            final_select_cols = []
            processed_internal_names = set()

            # Adiciona todas as colunas que são valores no internal_column_names_map e existem no LDF
            for internal_target_name in self.internal_column_names_map.values():
                if (
                    internal_target_name in ldf.columns
                ):  # Check if column exists in current ldf
                    final_select_cols.append(internal_target_name)
                    processed_internal_names.add(internal_target_name)

            # Adiciona quaisquer outras colunas do LDF que não foram processadas
            # Isso mantém colunas extras que o usuário pode ter no CSV
            for ldf_col in ldf.columns:
                if ldf_col not in processed_internal_names:
                    final_select_cols.append(ldf_col)

            if final_select_cols:
                # Verifica se todas as colunas em final_select_cols realmente existem no ldf.columns
                actual_cols_in_ldf = ldf.columns
                valid_final_select_cols = [
                    col for col in final_select_cols if col in actual_cols_in_ldf
                ]
                if valid_final_select_cols:
                    ldf = ldf.select(valid_final_select_cols)
                else:
                    # Isso não deveria acontecer se a lógica acima estiver correta,
                    # mas é uma salvaguarda. Se valid_final_select_cols for vazio, não seleciona nada.
                    logger.warning(
                        f"Nenhuma coluna válida para seleção final em {self.symbol_name}. O LDF pode ficar vazio ou com todas as colunas."
                    )

            return ldf

        except Exception as e:
            logger.error(
                f"Erro ao processar CSV para {self.symbol_name} ({self.csv_file_path}): {e}. Retornando None."
            )
            return None
