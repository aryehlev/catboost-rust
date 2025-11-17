use crate::Model;
use crate::error::{CatBoostError, CatBoostResult};
use polars::prelude::*;

/// Extension trait for CatBoost Model to support Polars DataFrames
pub trait ModelPolarsExt {
    /// Predict using a Polars DataFrame as input (numeric features only)
    ///
    /// This method efficiently converts the DataFrame to the format CatBoost expects.
    /// All numeric columns will be used as float features.
    ///
    /// # Arguments
    /// * `df` - Input DataFrame with numeric features
    ///
    /// # Returns
    /// A vector of prediction values
    ///
    /// # Example
    /// ```no_run
    /// # use catboost_rust::{Model, ModelPolarsExt};
    /// # use polars::prelude::*;
    /// let model = Model::load("model.cbm").unwrap();
    ///
    /// let df = df! {
    ///     "feature1" => [1.0f32, 2.0, 3.0],
    ///     "feature2" => [4.0f32, 5.0, 6.0],
    /// }.unwrap();
    ///
    /// let predictions = model.predict_dataframe(&df).unwrap();
    /// ```
    fn predict_dataframe(&self, df: &DataFrame) -> CatBoostResult<Vec<f64>>;

    /// Predict using specific columns from a Polars DataFrame
    ///
    /// # Arguments
    /// * `df` - Input DataFrame
    /// * `columns` - Column names to use as features (in order)
    fn predict_dataframe_with_columns(
        &self,
        df: &DataFrame,
        columns: &[&str],
    ) -> CatBoostResult<Vec<f64>>;

    /// Predict using a DataFrame with both float and categorical features
    ///
    /// # Arguments
    /// * `df` - Input DataFrame
    /// * `float_columns` - Names of columns to use as float features
    /// * `cat_columns` - Names of columns to use as categorical features (must be String type)
    fn predict_dataframe_with_types(
        &self,
        df: &DataFrame,
        float_columns: &[&str],
        cat_columns: &[&str],
    ) -> CatBoostResult<Vec<f64>>;
}

impl ModelPolarsExt for Model {
    fn predict_dataframe(&self, df: &DataFrame) -> CatBoostResult<Vec<f64>> {
        let float_features = dataframe_to_float_features(df)?;
        let cat_features: Vec<Vec<String>> = vec![vec![]; df.height()];

        self.calc_model_prediction(float_features, cat_features)
    }

    fn predict_dataframe_with_columns(
        &self,
        df: &DataFrame,
        columns: &[&str],
    ) -> CatBoostResult<Vec<f64>> {
        let column_names: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        let selected = df.select(column_names).map_err(|e| CatBoostError {
            description: format!("Failed to select columns: {}", e),
        })?;

        self.predict_dataframe(&selected)
    }

    fn predict_dataframe_with_types(
        &self,
        df: &DataFrame,
        float_columns: &[&str],
        cat_columns: &[&str],
    ) -> CatBoostResult<Vec<f64>> {
        // Extract float features
        let float_col_names: Vec<String> = float_columns.iter().map(|s| s.to_string()).collect();
        let float_df = df.select(float_col_names).map_err(|e| CatBoostError {
            description: format!("Failed to select float columns: {}", e),
        })?;
        let float_features = dataframe_to_float_features(&float_df)?;

        // Extract categorical features
        let cat_features = if cat_columns.is_empty() {
            vec![vec![]; df.height()]
        } else {
            let cat_col_names: Vec<String> = cat_columns.iter().map(|s| s.to_string()).collect();
            let cat_df = df.select(cat_col_names).map_err(|e| CatBoostError {
                description: format!("Failed to select categorical columns: {}", e),
            })?;
            dataframe_to_cat_features(&cat_df)?
        };

        self.calc_model_prediction(float_features, cat_features)
    }
}

/// Convert a Polars DataFrame to CatBoost float features format (Vec<Vec<f32>>)
///
/// Each inner Vec represents one row of features.
/// Optimized column-by-column conversion for better cache locality.
fn dataframe_to_float_features(df: &DataFrame) -> CatBoostResult<Vec<Vec<f32>>> {
    let num_rows = df.height();
    let num_features = df.width();

    if num_rows == 0 || num_features == 0 {
        return Err(CatBoostError {
            description: "DataFrame has zero rows or columns".to_string(),
        });
    }

    // Pre-allocate result rows
    let mut result: Vec<Vec<f32>> = (0..num_rows)
        .map(|_| Vec::with_capacity(num_features))
        .collect();

    // Process column by column - cast to Float32 for simplicity and speed
    for column in df.get_columns() {
        let series = column.as_materialized_series();

        // Cast to Float32 - Polars handles all type conversions efficiently
        let f32_series = series.cast(&DataType::Float32).map_err(|e| CatBoostError {
            description: format!("Failed to cast column to f32: {}", e),
        })?;

        let ca = f32_series.f32().map_err(|e| CatBoostError {
            description: format!("Failed to get f32 array: {}", e),
        })?;

        for (row_idx, opt_val) in ca.iter().enumerate() {
            let val = opt_val.ok_or_else(|| CatBoostError {
                description: format!("Null value at row {}", row_idx),
            })?;
            result[row_idx].push(val);
        }
    }

    Ok(result)
}

/// Convert a Polars DataFrame to CatBoost categorical features format (Vec<Vec<String>>)
/// Optimized column-by-column conversion for better cache locality.
fn dataframe_to_cat_features(df: &DataFrame) -> CatBoostResult<Vec<Vec<String>>> {
    let num_rows = df.height();
    let num_features = df.width();

    if num_rows == 0 || num_features == 0 {
        return Err(CatBoostError {
            description: "DataFrame has zero rows or columns".to_string(),
        });
    }

    // Fast path for single row
    if num_rows == 1 {
        let mut row_features = Vec::with_capacity(num_features);
        for col in df.get_columns() {
            let series = col.as_materialized_series();
            let value = extract_string_value(series, 0)?;
            row_features.push(value);
        }
        return Ok(vec![row_features]);
    }

    // Pre-allocate result rows
    let mut result: Vec<Vec<String>> = (0..num_rows)
        .map(|_| Vec::with_capacity(num_features))
        .collect();

    // Process column by column for better cache locality
    for col in df.get_columns() {
        let series = col.as_materialized_series();

        // For String columns, use direct iteration
        if matches!(series.dtype(), DataType::String) {
            let ca = series.str().map_err(|e| CatBoostError {
                description: format!("Failed to cast to String: {}", e),
            })?;

            for (row_idx, opt_val) in ca.iter().enumerate() {
                let val = opt_val.ok_or_else(|| CatBoostError {
                    description: format!("Null value at row {}", row_idx),
                })?;
                result[row_idx].push(val.to_string());
            }
        } else {
            // Fallback for other types
            for row_idx in 0..num_rows {
                let value = extract_string_value(series, row_idx)?;
                result[row_idx].push(value);
            }
        }
    }

    Ok(result)
}

/// Helper to extract a value from a ChunkedArray with explicit bounds and null checking
macro_rules! get_checked_value {
    ($ca:expr, $idx:expr) => {{
        if $idx >= $ca.len() {
            return Err(CatBoostError {
                description: format!("Index {} out of bounds (length: {})", $idx, $ca.len()),
            });
        }
        $ca.get($idx).ok_or_else(|| CatBoostError {
            description: format!("Null value at index {}", $idx),
        })?
    }};
}

/// Extract an f32 value from a Series at the given index
fn extract_f32_value(series: &Series, idx: usize) -> CatBoostResult<f32> {
    use DataType::*;

    match series.dtype() {
        Float32 => {
            let ca = series.f32().map_err(|e| CatBoostError {
                description: format!("Failed to cast to f32: {}", e),
            })?;
            Ok(get_checked_value!(ca, idx))
        }
        Float64 => {
            let ca = series.f64().map_err(|e| CatBoostError {
                description: format!("Failed to cast to f64: {}", e),
            })?;
            Ok(get_checked_value!(ca, idx) as f32)
        }
        Int8 => {
            let ca = series.i8().map_err(|e| CatBoostError {
                description: format!("Failed to cast to i8: {}", e),
            })?;
            Ok(get_checked_value!(ca, idx) as f32)
        }
        Int16 => {
            let ca = series.i16().map_err(|e| CatBoostError {
                description: format!("Failed to cast to i16: {}", e),
            })?;
            Ok(get_checked_value!(ca, idx) as f32)
        }
        Int32 => {
            let ca = series.i32().map_err(|e| CatBoostError {
                description: format!("Failed to cast to i32: {}", e),
            })?;
            Ok(get_checked_value!(ca, idx) as f32)
        }
        Int64 => {
            let ca = series.i64().map_err(|e| CatBoostError {
                description: format!("Failed to cast to i64: {}", e),
            })?;
            Ok(get_checked_value!(ca, idx) as f32)
        }
        UInt8 => {
            let ca = series.u8().map_err(|e| CatBoostError {
                description: format!("Failed to cast to u8: {}", e),
            })?;
            Ok(get_checked_value!(ca, idx) as f32)
        }
        UInt16 => {
            let ca = series.u16().map_err(|e| CatBoostError {
                description: format!("Failed to cast to u16: {}", e),
            })?;
            Ok(get_checked_value!(ca, idx) as f32)
        }
        UInt32 => {
            let ca = series.u32().map_err(|e| CatBoostError {
                description: format!("Failed to cast to u32: {}", e),
            })?;
            Ok(get_checked_value!(ca, idx) as f32)
        }
        UInt64 => {
            let ca = series.u64().map_err(|e| CatBoostError {
                description: format!("Failed to cast to u64: {}", e),
            })?;
            Ok(get_checked_value!(ca, idx) as f32)
        }
        Boolean => {
            let ca = series.bool().map_err(|e| CatBoostError {
                description: format!("Failed to cast to bool: {}", e),
            })?;
            let val = get_checked_value!(ca, idx);
            Ok(if val { 1.0 } else { 0.0 })
        }
        dt => Err(CatBoostError {
            description: format!("Unsupported data type for float conversion: {}", dt),
        }),
    }
}

/// Extract a String value from a Series at the given index
fn extract_string_value(series: &Series, idx: usize) -> CatBoostResult<String> {
    use DataType::*;

    match series.dtype() {
        String => {
            let ca = series.str().map_err(|e| CatBoostError {
                description: format!("Failed to cast to String: {}", e),
            })?;
            Ok(get_checked_value!(ca, idx).to_string())
        }
        // Convert numeric types to strings for categorical features
        Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64 => {
            let value = extract_f32_value(series, idx)?;
            Ok(format!("{}", value as i64))
        }
        Boolean => {
            let ca = series.bool().map_err(|e| CatBoostError {
                description: format!("Failed to cast to bool: {}", e),
            })?;
            let val = get_checked_value!(ca, idx);
            Ok(if val {
                "true".to_string()
            } else {
                "false".to_string()
            })
        }
        dt => Err(CatBoostError {
            description: format!("Unsupported data type for categorical conversion: {}", dt),
        }),
    }
}
