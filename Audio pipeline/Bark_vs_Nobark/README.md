### Summary of the Project: Dog Bark Detection

#### 1. Dataset Used

The primary dataset used for this project is the **Barkopedia Dog Vocal Detection Challenge** dataset, sourced from Hugging Face (`ArlingtonCL2/Barkopedia-Dog-Vocal-Detection`). Specifically, we utilized the `strong_audios` split, which includes audio clips with precise onset/offset annotations for 'dog' and 'dog_noise' events in `train.tsv`. 

For the evaluation phase, since the official `test.tsv` was unavailable, a placeholder `test_metadata_df` was created by copying the `train.tsv` data and removing the 'label' column to simulate an unlabeled test set.

#### 2. End-to-End Pipeline

The established pipeline for training a binary dog bark detection classifier involves the following steps:

1.  **Dataset Loading & Preprocessing**: 
    *   The `train.tsv` annotations were loaded to get `clip_id`, `onset`, `offset`, and `label` (dog/non-dog). 
    *   Actual audio files corresponding to these `clip_id`s were downloaded using `huggingface_hub.hf_hub_download`, handling Git LFS pointers.

2.  **Windowing and Labeling**: 
    *   A custom function `create_windows_and_labels` was developed to slide 1-second fixed windows with a 0.5-second hop across each audio clip.
    *   Each window was labeled '1' (bark) if it overlapped with any 'dog' or 'dog_noise' event from the `train.tsv` annotations, and '0' (non-bark) otherwise. This resulted in `X` (raw audio windows) and `y` (binary labels).

3.  **Feature Extraction**: 
    *   The pre-trained **YAMNet** model from TensorFlow Hub was used to extract 1024-dimensional embeddings for each 1-second audio window. 
    *   YAMNet outputs two embeddings per 1-second window, which were then averaged to create a single 1024-dimensional feature vector per window (`X_aligned`), matching the number of labels (`y`).

4.  **Model Training**: 
    *   A **Support Vector Machine (SVC)** classifier was chosen for binary classification.
    *   Training was performed using **GroupKFold cross-validation** (5 splits), ensuring that all windows originating from the same source audio clip were kept together in either the training or validation set. This prevents data leakage and provides a more robust evaluation of the model's generalization performance.
    *   The trained classifier was then saved using `joblib` for future use.

5.  **Test Set Preparation**: 
    *   The placeholder `test_metadata_df` (derived from `train.tsv` without labels) was processed similarly.
    *   Test audio files were downloaded, windowed, and YAMNet features (`X_test_aligned`) were extracted, mirroring the training pipeline.

#### 3. Model Performance

During the 5-fold GroupKFold cross-validation on the training data, the SVM classifier achieved the following average performance metrics:

*   **Average Accuracy**: {:.4f} (+/- {:.4f})
*   **Average Precision**: {:.4f} (+/- {:.4f})
*   **Average Recall**: {:.4f} (+/- {:.4f})
*   **Average F1 Score**: {:.4f} (+/- {:.4f})

These metrics indicate a strong performance of the model in distinguishing between bark and non-bark segments based on the YAMNet features, with good balance between precision and recall as shown by the F1 Score.