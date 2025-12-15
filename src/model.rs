use crate::error::{CatBoostError, CatBoostResult};
use crate::features::{EmptyEmbeddingFeatures, EmptyTextFeatures, ObjectsOrderFeatures};
use crate::sys;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

pub struct Model {
    handle: *mut sys::ModelCalcerHandle,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    fn new() -> Self {
        let model_handle = unsafe { sys::ModelCalcerCreate() };
        Model {
            handle: model_handle,
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

    /// # Safety
    ///
    /// This function is unsafe because it dereferences a raw pointer and assumes a memory
    /// allocation contract with an external C API.
    ///
    /// - `ptr` must be a valid pointer to a C-allocated buffer containing `count` elements of type `T`,
    ///   or it must be a null pointer if `count` is 0.
    /// - The buffer must have been allocated by a `malloc`-compatible allocator, as the CatBoost C API
    ///   documentation for functions like `GetFloatFeatureIndices` and `GetModelUsedFeaturesNames`
    ///   stipulates that the caller is responsible for freeing the returned buffer. The standard C
    ///   mechanism for this is `free()`.
    ///   (Source: https://github.com/catboost/catboost/blob/master/catboost/libs/model_interface/c_api.h)
    ///
    /// This function takes ownership of the buffer and frees it with `libc::free` after copying
    /// the data into a Rust `Vec`.
    unsafe fn from_c_allocated_buffer<T: Copy>(ptr: *mut T, count: usize) -> Vec<T> {
        if ptr.is_null() {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            result.push(unsafe { *ptr.add(i) });
        }
        unsafe { libc::free(ptr as *mut _) };
        result
    }

    /// Converts a C-style array of feature indices into a `Vec<usize>`, freeing the C buffer.
    fn get_feature_indices_from_c(
        indices_ptr: *mut usize,
        count: usize,
        err_msg: &str,
    ) -> CatBoostResult<Vec<usize>> {
        if indices_ptr.is_null() {
            if count == 0 {
                return Ok(Vec::new());
            }
            return Err(CatBoostError {
                description: err_msg.to_owned(),
            });
        }
        // SAFETY: The contract for CatBoost functions like `GetFloatFeatureIndices` is that they
        // return a `malloc`-allocated buffer that the caller must free. `from_c_allocated_buffer`
        // upholds this contract by copying the data and then calling `libc::free`.
        let indices = unsafe { Self::from_c_allocated_buffer(indices_ptr, count) };
        Ok(indices)
    }

    /// Converts a C-style array of C strings into a `Vec<String>`, freeing all associated C memory.
    fn get_feature_names_from_c(
        names_ptr: *mut *mut std::ffi::c_char,
        count: usize,
        err_msg: &str,
    ) -> CatBoostResult<Vec<String>> {
        if names_ptr.is_null() {
            if count == 0 {
                return Ok(Vec::new());
            }
            return Err(CatBoostError {
                description: err_msg.to_owned(),
            });
        }
        // SAFETY: The contract for `GetModelUsedFeaturesNames` is that it returns a `malloc`-allocated
        // array of `malloc`-allocated strings. The caller must free both the outer array and each
        // inner string pointer. This block upholds that contract.
        let mut names = Vec::with_capacity(count);
        for i in 0..count {
            let ptr = unsafe { *names_ptr.add(i) };
            let s = unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned();
            names.push(s);
            unsafe { libc::free(ptr as *mut _) };
        }
        unsafe { libc::free(names_ptr as *mut _) };
        Ok(names)
    }

    /// Get names of specific type of features used in model,
    /// returns error if index out of bounds
    fn get_specific_feature_names(
        &self,
        indices_fn: unsafe extern "C" fn(
            *mut sys::ModelCalcerHandle,
            *mut *mut usize,
            *mut usize,
        ) -> bool,
        err_msg: &str,
    ) -> CatBoostResult<Vec<String>> {
        let all_names = self.get_feature_names()?;
        let indices = self.get_feature_indices(indices_fn, err_msg)?;
        indices
            .into_iter()
            .map(|i| {
                all_names
                    .get(i)
                    .ok_or_else(|| CatBoostError {
                        description: format!("feature index {} out of bounds", i),
                    })
                    .map(|s| s.clone())
            })
            .collect()
    }

    /// Get names of features used in model
    #[cfg(catboost_feature_indices)]
    pub fn get_feature_names(&self) -> CatBoostResult<Vec<String>> {
        unsafe {
            let mut names_ptr: *mut *mut std::ffi::c_char = std::ptr::null_mut();
            let mut count: usize = 0;

            let ok = sys::GetModelUsedFeaturesNames(self.handle, &mut names_ptr, &mut count);
            CatBoostError::check_return_value(ok)?;

            Self::get_feature_names_from_c(
                names_ptr,
                count,
                "GetModelUsedFeaturesNames returned null pointer",
            )
        }
    }

    fn get_feature_indices(
        &self,
        indices_fn: unsafe extern "C" fn(
            *mut sys::ModelCalcerHandle,
            *mut *mut usize,
            *mut usize,
        ) -> bool,
        err_msg: &str,
    ) -> CatBoostResult<Vec<usize>> {
        let mut indices_ptr: *mut usize = std::ptr::null_mut();
        let mut count: usize = 0;
        CatBoostError::check_return_value(unsafe {
            indices_fn(self.handle, &mut indices_ptr, &mut count)
        })?;
        Self::get_feature_indices_from_c(indices_ptr, count, err_msg)
    }

    /// Get names of float features used in model
    #[cfg(catboost_feature_indices)]
    pub fn get_float_feature_names(&self) -> CatBoostResult<Vec<String>> {
        self.get_specific_feature_names(
            sys::GetFloatFeatureIndices,
            "GetFloatFeatureIndices returned null pointer",
        )
    }

    /// Get names of cat features used in model
    #[cfg(catboost_feature_indices)]
    pub fn get_cat_feature_names(&self) -> CatBoostResult<Vec<String>> {
        self.get_specific_feature_names(
            sys::GetCatFeatureIndices,
            "GetCatFeatureIndices returned null pointer",
        )
    }

    /// Get names of text features used in model
    #[cfg(catboost_feature_indices)]
    pub fn get_text_feature_names(&self) -> CatBoostResult<Vec<String>> {
        self.get_specific_feature_names(
            sys::GetTextFeatureIndices,
            "GetTextFeatureIndices returned null pointer",
        )
    }

    /// Get names of embedding features used in model
    #[cfg(catboost_feature_indices)]
    pub fn get_embedding_feature_names(&self) -> CatBoostResult<Vec<String>> {
        self.get_specific_feature_names(
            sys::GetEmbeddingFeatureIndices,
            "GetEmbeddingFeatureIndices returned null pointer",
        )
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
