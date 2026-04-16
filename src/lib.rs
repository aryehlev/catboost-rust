// Include the CatBoost C API bindings
mod sys;

mod error;
pub use crate::error::{CatBoostError, CatBoostResult};

mod features;
pub use crate::features::{
    EmptyCatFeatures, EmptyEmbeddingFeatures, EmptyFloatFeatures, EmptyTextFeatures,
    ObjectsOrderFeatures,
};

mod model;
pub use crate::model::Model;
