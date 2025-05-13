from typing import Any, Dict, Optional

import polars as pl
from typing import Dict, List, Optional, Set, Union
import logging

# Configuração do logger
logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(
        self,
        validate_ohlc_flag: bool = True,
        fill_missing_method: str = "ffill",  # Padrão consistente com CSVDataHandler
        outlier_detection_config: Optional[Dict[str, Any]] = None,
    ):
        self.validate_ohlc_flag = validate_ohlc_flag
        self.fill_missing_method = fill_missing_method
        self.outlier_detection_config = outlier_detection_config or {
            "method": None,
            "threshold": 3.0,
            "action": "clip",
            "columns": ["open", "high", "low", "close", "volume"],
        }
        # Validação da config de outlier, similar ao CSVDataHandler
        if self.outlier_detection_config.get("method") not in [
            None,
            "std_dev",
            "iqr",
            "absolute",
        ]:
            logger.warning(
                f"(DataValidator): Método de detecção de outliers '{self.outlier_detection_config.get('method')}' inválido. Desativando."
            )
            self.outlier_detection_config["method"] = None

    def validate_ohlc_integrity(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """
        Valida a integridade dos dados OHLC e corrige inconsistências quando possível.
        """
        if not self.validate_ohlc_flag or df.is_empty():
            return df

        issues_count = 0  # Manter para contagem, log no final

        # Verificar se high >= low
        high_low_invalid_count = df.filter(
            pl.col("high") < pl.col("low")
        ).height  # Contar antes de corrigir
        if high_low_invalid_count > 0:
            issues_count += high_low_invalid_count
            logger.debug(
                f"(DataValidator): {high_low_invalid_count} barras com High < Low para {symbol}. Corrigindo..."
            )

        df = (
            df.with_columns(
                [
                    pl.when(pl.col("high") < pl.col("low"))
                    .then(pl.struct(high=pl.col("low"), low=pl.col("high")))
                    .otherwise(pl.struct(high=pl.col("high"), low=pl.col("low")))
                    .alias("temp_ohlc")
                ]
            )
            .with_columns(
                [
                    pl.col("temp_ohlc").struct.field("high").alias("high"),
                    pl.col("temp_ohlc").struct.field("low").alias("low"),
                ]
            )
            .drop("temp_ohlc")
        )

        # Verificar se open está entre high e low
        open_invalid_count = df.filter(
            (pl.col("open") > pl.col("high")) | (pl.col("open") < pl.col("low"))
        ).height
        if open_invalid_count > 0:
            issues_count += open_invalid_count
            logger.debug(
                f"(DataValidator): {open_invalid_count} barras com Open fora do intervalo High-Low para {symbol}. Corrigindo..."
            )
        df = df.with_columns(
            [
                pl.when(pl.col("open") > pl.col("high"))
                .then(pl.col("high"))
                .when(pl.col("open") < pl.col("low"))
                .then(pl.col("low"))
                .otherwise(pl.col("open"))
                .alias("open")
            ]
        )

        # Verificar se close está entre high e low
        close_invalid_count = df.filter(
            (pl.col("close") > pl.col("high")) | (pl.col("close") < pl.col("low"))
        ).height
        if close_invalid_count > 0:
            issues_count += close_invalid_count
            logger.debug(
                f"(DataValidator): {close_invalid_count} barras com Close fora do intervalo High-Low para {symbol}. Corrigindo..."
            )
        df = df.with_columns(
            [
                pl.when(pl.col("close") > pl.col("high"))
                .then(pl.col("high"))
                .when(pl.col("close") < pl.col("low"))
                .then(pl.col("low"))
                .otherwise(pl.col("close"))
                .alias("close")
            ]
        )

        # Verificar volume negativo (se a coluna 'volume' existir)
        if "volume" in df.columns:
            volume_invalid_count = df.filter(pl.col("volume") < 0).height
            if volume_invalid_count > 0:
                issues_count += volume_invalid_count
                logger.debug(
                    f"(DataValidator): {volume_invalid_count} barras com Volume negativo para {symbol}. Corrigindo..."
                )
            df = df.with_columns(
                [
                    pl.when(pl.col("volume") < 0)
                    .then(0.0)
                    .otherwise(pl.col("volume"))
                    .alias("volume")
                ]
            )
        if issues_count > 0:
            logger.info(
                f"Total de {issues_count} problemas de integridade OHLCV corrigidos para {symbol} por DataValidator"
            )
        return df

    def fill_missing_values(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """
        Preenche valores ausentes nas colunas OHLCV.
        Usa self.fill_missing_method configurado no construtor.
        """
        if df.is_empty():
            return df

        # Colunas a verificar e preencher
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        cols_to_process = [
            col for col in ohlcv_cols if col in df.columns and df[col].is_null().any()
        ]

        if not cols_to_process:
            return df  # Sem valores nulos nas colunas alvo ou colunas não existem

        logger.debug(
            f"Valores ausentes detectados em {symbol} por DataValidator em colunas: {cols_to_process}"
        )

        if self.fill_missing_method == "drop":
            original_height = df.height
            df = df.drop_nulls(subset=cols_to_process)
            logger.info(
                f"{original_height - df.height} linhas com valores ausentes removidas para {symbol} por DataValidator (método: drop)"
            )
        elif self.fill_missing_method == "zeros":
            for col in cols_to_process:
                df = df.with_columns(pl.col(col).fill_null(0.0).alias(col))
            logger.info(
                f"Valores ausentes preenchidos com zeros para {symbol} por DataValidator em colunas: {cols_to_process}"
            )
        elif self.fill_missing_method == "interpolate":
            for col in cols_to_process:
                df = df.with_columns(pl.col(col).interpolate().alias(col))
            logger.info(
                f"Valores ausentes interpolados para {symbol} por DataValidator em colunas: {cols_to_process}"
            )
        # ffill_bfill é tratado como ffill aqui
        elif (
            self.fill_missing_method == "ffill"
            or self.fill_missing_method == "ffill_bfill"
        ):
            for col in cols_to_process:
                df = df.with_columns(
                    pl.col(col).fill_null(strategy="forward").alias(col)
                )
            # Adicionar bfill se a estratégia for ffill_bfill
            if self.fill_missing_method == "ffill_bfill":
                for col in cols_to_process:
                    # Aplicar bfill apenas se ainda houver nulos (especialmente no início)
                    if df[col].is_null().any():
                        df = df.with_columns(
                            pl.col(col).fill_null(strategy="backward").alias(col)
                        )
                logger.info(
                    f"Valores ausentes preenchidos com ffill_bfill para {symbol} por DataValidator em colunas: {cols_to_process}"
                )
            else:
                logger.info(
                    f"Valores ausentes preenchidos com ffill para {symbol} por DataValidator em colunas: {cols_to_process}"
                )
        else:  # Método desconhecido ou não implementado aqui, não faz nada
            logger.warning(
                f"(DataValidator): Método de preenchimento '{self.fill_missing_method}' desconhecido. Nenhum preenchimento realizado para {symbol}."
            )
            return df

        # Opcional: Verificar se ainda existem nulos e dropar como último recurso
        # Recalcular cols_to_process pois algumas colunas podem ter sido totalmente preenchidas
        final_cols_to_check_nulls = [
            col for col in ohlcv_cols if col in df.columns and df[col].is_null().any()
        ]
        if final_cols_to_check_nulls:
            original_height_before_final_drop = df.height
            logger.warning(
                f"(DataValidator): Ainda existem valores ausentes em {symbol} ({final_cols_to_check_nulls}) após preenchimento com '{self.fill_missing_method}'. Removendo linhas problemáticas."
            )
            df = df.drop_nulls(subset=final_cols_to_check_nulls)
            if df.height < original_height_before_final_drop:
                logger.info(
                    f"{original_height_before_final_drop - df.height} linhas adicionais removidas devido a NaNs persistentes para {symbol}."
                )
        return df

    def handle_outliers(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """
        Detecta e trata outliers nas colunas configuradas.
        Usa self.outlier_detection_config.
        """
        if df.is_empty() or self.outlier_detection_config.get("method") is None:
            return df

        method = self.outlier_detection_config["method"]
        threshold = self.outlier_detection_config.get("threshold", 3.0)
        action = self.outlier_detection_config.get("action", "clip")
        config_columns = self.outlier_detection_config.get(
            "columns", ["open", "high", "low", "close", "volume"]
        )

        columns_to_process = [col for col in config_columns if col in df.columns]
        if not columns_to_process:
            return df

        outliers_found_total = 0  # Para log
        result_df = (
            df.clone()
        )  # Clonar para evitar modificar o original se action='remove'

        for col in columns_to_process:
            # Para aplicar std_dev, iqr em dataframes de uma linha (comuns no update_bars),
            # precisamos de um tratamento especial ou garantir que df tenha mais pontos.
            # Se df é sempre uma linha, mean, std, quantiles não são significativos.
            # Assumindo que este método pode ser chamado com mais de uma linha.
            if df.height <= 1 and method in ["std_dev", "iqr"]:
                logger.debug(
                    f"(DataValidator): Não é possível aplicar método '{method}' de outlier em DataFrame com <=1 linha para {symbol}, coluna {col}. Pulando."
                )
                continue

            if method == "std_dev":
                mean = df[col].mean()
                std = df[col].std()
                if mean is None or std is None or std == 0:
                    continue  # Não é possível calcular limites
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            elif method == "iqr":
                # Polars usa 'nearest' por padrão, linear pode ser melhor
                q1 = df[col].quantile(0.25, interpolation="linear")
                q3 = df[col].quantile(0.75, interpolation="linear")
                if q1 is None or q3 is None:
                    continue
                iqr = q3 - q1
                if iqr == 0:
                    continue  # Evita divisão por zero ou limites idênticos se todos os valores forem iguais
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
            elif method == "absolute":
                # Este método é mais problemático para DataFrames de uma linha se threshold é percentual.
                # Se for um valor absoluto, seria mais simples. A lógica original parecia percentual da mediana.
                # Para uma linha, mediana é o próprio valor, então limites seriam (valor +/- X% de valor).
                # Esta lógica pode precisar de revisão para o caso de uma linha.
                # Por simplicidade, se for uma linha, vamos pular 'absolute' a menos que a lógica seja mais robusta.
                if df.height <= 1:
                    continue
                median = df[col].median()
                if median is None:
                    continue
                lower_bound = median * (1 - threshold / 100.0)
                upper_bound = median * (1 + threshold / 100.0)
                if col in ["volume", "open", "high", "low", "close"]:
                    # Preços e volumes não podem ser negativos
                    lower_bound = max(0, lower_bound)
            else:
                continue  # Método não reconhecido

            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            if outliers_count > 0:
                outliers_found_total += outliers_count

            if action == "clip":
                result_df = result_df.with_columns(
                    pl.when(pl.col(col) < lower_bound)
                    .then(lower_bound)
                    .when(pl.col(col) > upper_bound)
                    .then(upper_bound)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            elif action == "remove":
                # Cuidado ao remover linhas, pois altera o shape do DataFrame
                # Se for chamado para cada barra, isso significa que a barra pode ser completamente removida.
                result_df = result_df.filter(~outliers_mask)
                if (
                    result_df.is_empty()
                ):  # Se a remoção esvaziou o DF (ex: era uma linha e foi removida)
                    logger.warning(
                        f"(DataValidator): Remoção de outlier para {symbol} na coluna {col} resultou em DataFrame vazio."
                    )
                    return result_df  # Retorna o DF vazio

        if outliers_found_total > 0:
            action_msg = "apenas detectados"
            if action == "clip":
                action_msg = "valores limitados"
            elif action == "remove":
                action_msg = f"{df.height - result_df.height} linhas removidas"
            logger.info(
                f"DataValidator: {outliers_found_total} outliers em {symbol} ({method}). Ação: {action_msg}."
            )

        return result_df
