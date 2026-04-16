use crate::error::{CatBoostError, CatBoostResult};
use crate::features::{EmptyEmbeddingFeatures, EmptyTextFeatures, ObjectsOrderFeatures};
use crate::sys;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

pub struct Model {
    handle: *mut sys::ModelCalcerHandle,
    /// Buffer owner for zero-copy loading - keeps the buffer alive for model's lifetime
    #[cfg(catboost_zero_copy)]
    _buffer_owner: Option<Vec<u8>>,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    fn new() -> Self {
        let model_handle = unsafe { sys::ModelCalcerCreate() };
        Model {
            handle: model_handle,
            #[cfg(catboost_zero_copy)]
            _buffer_owner: None,
        }
    }

    /// Load a model from a file
    pub fn load<P: AsRef<Path>>(path: P) -> CatBoostResult<Self> {
        let model = Model::new();
        let path_c_str = CString::new(path.as_ref().to_str().unwrap()).unwrap();
        CatBoostError::check_return_value(unsafe {
            sys::LoadFullModelFromFile(model.handle, path_c_str.as_ptr())
        })?;
        Ok(model)
    }

    /// Load a model from a buffer using zero-copy approach
    ///
    /// This method uses LoadFullModelZeroCopy which does NOT copy the model data.
    /// Instead, the model keeps a reference to the buffer and reads from it directly.
    ///
    /// Requires CatBoost v1.2.9+.
    ///
    /// # Example
    /// ```no_run
    /// use catboost_rust::Model;
    /// use std::fs;
    ///
    /// let buffer = fs::read("model.cbm").unwrap();
    /// let model = Model::load_buffer_zero_copy(buffer).unwrap();
    /// ```
    #[cfg(catboost_zero_copy)]
    pub fn load_buffer_zero_copy(buffer: Vec<u8>) -> CatBoostResult<Self> {
        let mut model = Model::new();

        CatBoostError::check_return_value(unsafe {
            sys::LoadFullModelZeroCopy(
                model.handle,
                buffer.as_ptr() as *const std::os::raw::c_void,
                buffer.len(),
            )
        })?;

        model._buffer_owner = Some(buffer);
        Ok(model)
    }

    /// Load a model from a buffer
    pub fn load_buffer<P: AsRef<Vec<u8>>>(buffer: P) -> CatBoostResult<Self> {
        let model = Model::new();
        CatBoostError::check_return_value(unsafe {
            sys::LoadFullModelFromBuffer(
                model.handle,
                buffer.as_ref().as_ptr() as *const std::os::raw::c_void,
                buffer.as_ref().len(),
            )
        })?;
        Ok(model)
    }

    fn set_or_check_object_count<
        TFeature,
        TObjectFeatures: AsRef<[TFeature]>,
        TFeatures: AsRef<[TObjectFeatures]>,
    >(
        object_count: &mut Option<usize>,
        features: &TFeatures,
    ) -> CatBoostResult<()> {
        let features_array_size = features.as_ref().len();
        if features_array_size > 0 {
            match object_count {
                Some(count) => {
                    if *count != features_array_size {
                        return Err(CatBoostError {
                            description: "features arguments have different nonzero sizes"
                                .to_owned(),
                        });
                    }
                }
                None => {
                    object_count.replace(features_array_size);
                }
            }
        }
        Ok(())
    }

    /// Calculate raw model predictions
    pub fn predict<
        TObjectFloatFeatures: AsRef<[f32]>,
        TFloatFeatures: AsRef<[TObjectFloatFeatures]>,
        TCatFeatureString: AsRef<str>,
        TObjectCatFeatures: AsRef<[TCatFeatureString]>,
        TCatFeatures: AsRef<[TObjectCatFeatures]>,
        TTextFeatureString: AsRef<CStr>,
        TObjectTextFeatures: AsRef<[TTextFeatureString]>,
        TTextFeatures: AsRef<[TObjectTextFeatures]>,
        TEmbedding: AsRef<[f32]>,
        TObjectEmbeddingFeatures: AsRef<[TEmbedding]>,
        TEmbeddingFeatures: AsRef<[TObjectEmbeddingFeatures]>,
    >(
        &self,
        features: ObjectsOrderFeatures<
            TFloatFeatures,
            TCatFeatures,
            TTextFeatures,
            TEmbeddingFeatures,
        >,
    ) -> CatBoostResult<Vec<f64>> {
        let mut object_count = None;
        Self::set_or_check_object_count(&mut object_count, &features.float_features)?;
        Self::set_or_check_object_count(&mut object_count, &features.cat_features)?;
        Self::set_or_check_object_count(&mut object_count, &features.text_features)?;
        Self::set_or_check_object_count(&mut object_count, &features.embedding_features)?;
        if object_count.is_none() {
            return Err(CatBoostError {
                description: "all features arguments are empty".to_owned(),
            });
        }

        let mut float_features_ptr = features
            .float_features
            .as_ref()
            .iter()
            .map(|x| x.as_ref().as_ptr())
            .collect::<Vec<_>>();

        let hashed_cat_features = features
            .cat_features
            .as_ref()
            .iter()
            .map(|doc_cat_features| {
                doc_cat_features
                    .as_ref()
                    .iter()
                    .map(|cat_feature| unsafe {
                        sys::GetStringCatFeatureHash(
                            cat_feature.as_ref().as_ptr() as *const std::os::raw::c_char,
                            cat_feature.as_ref().len(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut hashed_cat_features_ptr = hashed_cat_features
            .iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();

        let mut text_features_ptr_storage = features
            .text_features
            .as_ref()
            .iter()
            .map(|object_text_features| {
                object_text_features
                    .as_ref()
                    .iter()
                    .map(|text| text.as_ref().as_ptr())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut text_features_ptr = text_features_ptr_storage
            .iter_mut()
            .map(|object_texts_ptrs: &mut Vec<*const c_char>| object_texts_ptrs.as_mut_ptr())
            .collect::<Vec<_>>();

        let mut embedding_dimensions = if !features.embedding_features.as_ref().is_empty() {
            features.embedding_features.as_ref()[0]
                .as_ref()
                .iter()
                .map(|x| x.as_ref().len())
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        let mut embedding_features_ptr_storage = features
            .embedding_features
            .as_ref()
            .iter()
            .map(|object_embeddings| {
                object_embeddings
                    .as_ref()
                    .iter()
                    .map(|embedding| embedding.as_ref().as_ptr())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut embedding_features_ptr = embedding_features_ptr_storage
            .iter_mut()
            .map(|object_embeddings_ptrs: &mut Vec<*const f32>| object_embeddings_ptrs.as_mut_ptr())
            .collect::<Vec<_>>();

        let mut prediction = vec![0.0; object_count.unwrap() * self.get_dimensions_count()];

        #[cfg(catboost_embeddings)]
        {
            // v1.1.1+: Use function with embedding support
            CatBoostError::check_return_value(unsafe {
                sys::CalcModelPredictionWithHashedCatFeaturesAndTextAndEmbeddingFeatures(
                    self.handle,
                    object_count.unwrap(),
                    float_features_ptr.as_mut_ptr(),
                    if features.float_features.as_ref().is_empty() {
                        0
                    } else {
                        features.float_features.as_ref()[0].as_ref().len()
                    },
                    hashed_cat_features_ptr.as_mut_ptr(),
                    if features.cat_features.as_ref().is_empty() {
                        0
                    } else {
                        features.cat_features.as_ref()[0].as_ref().len()
                    },
                    text_features_ptr.as_mut_ptr(),
                    if features.text_features.as_ref().is_empty() {
                        0
                    } else {
                        features.text_features.as_ref()[0].as_ref().len()
                    },
                    embedding_features_ptr.as_mut_ptr(),
                    embedding_dimensions.as_mut_ptr(),
                    embedding_dimensions.len(),
                    prediction.as_mut_ptr(),
                    prediction.len(),
                )
            })?;
        }

        #[cfg(not(catboost_embeddings))]
        {
            // v1.0.x: Use function without embedding support (embeddings will be ignored)
            if !features.embedding_features.as_ref().is_empty() {
                return Err(CatBoostError {
                    description: "Embedding features are not supported in this CatBoost version. Please use v1.1.1 or later.".to_string()
                });
            }

            CatBoostError::check_return_value(unsafe {
                sys::CalcModelPredictionWithHashedCatFeaturesAndTextFeatures(
                    self.handle,
                    object_count.unwrap(),
                    float_features_ptr.as_mut_ptr(),
                    if features.float_features.as_ref().is_empty() {
                        0
                    } else {
                        features.float_features.as_ref()[0].as_ref().len()
                    },
                    hashed_cat_features_ptr.as_mut_ptr(),
                    if features.cat_features.as_ref().is_empty() {
                        0
                    } else {
                        features.cat_features.as_ref()[0].as_ref().len()
                    },
                    text_features_ptr.as_mut_ptr(),
                    if features.text_features.as_ref().is_empty() {
                        0
                    } else {
                        features.text_features.as_ref()[0].as_ref().len()
                    },
                    prediction.as_mut_ptr(),
                    prediction.len(),
                )
            })?;
        }

        Ok(prediction)
    }

    /// Calculate raw model predictions on float features and string categorical feature values
    pub fn calc_model_prediction<
        TFloatFeature: AsRef<[f32]>,
        TFloatFeatures: AsRef<[TFloatFeature]>,
        TString: AsRef<str>,
        TCatFeature: AsRef<[TString]>,
        TCatFeatures: AsRef<[TCatFeature]>,
    >(
        &self,
        float_features: TFloatFeatures,
        cat_features: TCatFeatures,
    ) -> CatBoostResult<Vec<f64>> {
        self.predict(ObjectsOrderFeatures {
            float_features,
            cat_features,
            text_features: EmptyTextFeatures {},
            embedding_features: EmptyEmbeddingFeatures {},
        })
    }

    /// Get expected float feature count for model
    pub fn get_float_features_count(&self) -> usize {
        unsafe { sys::GetFloatFeaturesCount(self.handle) }
    }

    /// Get expected categorical feature count for model
    pub fn get_cat_features_count(&self) -> usize {
        unsafe { sys::GetCatFeaturesCount(self.handle) }
    }

    /// Get expected text feature count for model
    /// Only available in CatBoost v1.2+
    #[cfg(catboost_text_count)]
    pub fn get_text_features_count(&self) -> usize {
        unsafe { sys::GetTextFeaturesCount(self.handle) }
    }

    /// Get expected text feature count for model (returns 0 for older versions)
    #[cfg(not(catboost_text_count))]
    pub fn get_text_features_count(&self) -> usize {
        0
    }

    /// Get expected embedding feature count for model
    /// Only available in CatBoost v1.1.1+
    #[cfg(catboost_embeddings)]
    pub fn get_embedding_features_count(&self) -> usize {
        unsafe { sys::GetEmbeddingFeaturesCount(self.handle) }
    }

    /// Get expected embedding feature count for model (returns 0 for older versions)
    #[cfg(not(catboost_embeddings))]
    pub fn get_embedding_features_count(&self) -> usize {
        0
    }

    /// Get number of trees in model
    pub fn get_tree_count(&self) -> usize {
        unsafe { sys::GetTreeCount(self.handle) }
    }

    /// Get number of dimensions in model
    pub fn get_dimensions_count(&self) -> usize {
        unsafe { sys::GetDimensionsCount(self.handle) }
    }

    pub fn enable_gpu_evaluation(&self) -> CatBoostResult<()> {
        CatBoostError::check_return_value(unsafe { sys::EnableGPUEvaluation(self.handle, 0) })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { sys::ModelCalcerDelete(self.handle) };
    }
}
