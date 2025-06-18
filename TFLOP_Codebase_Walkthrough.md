# TFLOP Codebase Walkthrough

This document provides a detailed walkthrough of the TFLOP codebase, aligning the concepts described in the TFLOP paper with their concrete implementations in Python.

## Section 1: Data Preprocessing and Input Pipeline

This section details how raw data is processed and prepared for the TFLOP model.

### 1. Initial Inputs for Data Preprocessing (Subtask 1)

The TFLOP pipeline expects the following raw inputs for a single data sample, primarily processed by `dataset/preprocess_data.py`:

*   **Image Data:**
    *   An image file (e.g., PNG from PubTabNet). The path is constructed from a base path, split name, and filename (e.g., `PMC12345_table_0.png`) found in the PubTabNet JSONL metadata.
    *   This image is loaded as a PIL.Image object later in `tflop/datamodule/datasets/tflop.py`.
*   **Ground-Truth HTML Structure:**
    *   Originates from `loaded_data["html"]["structure"]["tokens"]` in `dataset/preprocess_data.py` (from PubTabNet JSONL).
    *   This is a list of HTML tokens representing the table structure (e.g., `['<table>', '<tr>', '<td>', ...]`).
    *   Stored as `"org_html"` in the intermediate JSONL files.
*   **OCR/Text Region Data (Detected Regions):**
    *   Initially loaded from pickle files (`pse_det_data`) in `dataset/preprocess_data.py`. `pse_det_result["bbox"]` holds raw OCR bounding boxes.
    *   The function `preprocess_det_bbox` in `dataset/preprocess_data.py` matches these OCR boxes to ground-truth HTML cells and groups them.
    *   This processed data, including grouped OCR box coordinates and their mapping to original HTML cell indices, is stored as `"dr_coord"` in the intermediate JSONL files.

### 2. HTML to OTSL Sequence Transformation (Subtask 2)

The ground-truth HTML structure is converted into a sequence of OTSL (Ordered Table Structure Language) tokens. This conversion is primarily handled by the `convert_html_to_otsl` function in `dataset/preprocess_data_utils.py`, using tag mappings from `dataset/data_preprocessing_config.json`.

*   **Role of `convert_html_to_otsl`:**
    *   Takes a list of HTML tokens and `otsl_tag_maps` (e.g., `{"C": "C-tag", "NL": "NL-tag", ...}`).
    *   Internally uses helper functions `get_OTSL_array` (to create a 2D grid representation with OTSL tags) and `convert_OTSL_array_to_OTSL_seq` (to flatten the grid into a sequence).
*   **OTSL Tag Definitions (from `data_preprocessing_config.json`):**
    *   `C-tag`: A standard data cell. Also the origin cell of a span.
    *   `NL-tag`: Marks a new line, indicating the end of a table row.
    *   `L-tag`: ("Left") Marks a cell covered by a `colspan` from a cell to its left.
    *   `U-tag`: ("Up") Marks a cell covered by a `rowspan` from a cell above it.
    *   `X-tag`: ("Cross") Marks a cell covered by both a `rowspan` and `colspan`.
*   **Mapping Standard Table Tags:**
    *   `<thead>`, `<tbody>`, `</thead>`, `</tbody>`: Preserved directly in the final OTSL sequence.
    *   `<tr>`, `</tr>`: Used to segment HTML into rows; results in an `NL-tag` at the end of each row's cell tags.
    *   `<td>`: Becomes a `C-tag`. If it starts a span, subsequent covered cells become `L-tag`, `U-tag`, or `X-tag`.
*   **Handling `rowspan` (in `get_OTSL_array` and `insert_data_into_OTSL`):**
    *   When a `<td>` with `rowspan="N"` is encountered, its position in the 2D OTSL array gets a `C-tag`.
    *   The `N-1` cells directly below it in the same column receive `U-tag`s.
    *   If the cell also has a `colspan`, cells covered by both rowspan and colspan (excluding the first row/column of the span) get `X-tag`s.
*   **Handling `colspan` (in `get_OTSL_array` and `insert_data_into_OTSL`):**
    *   When a `<td>` with `colspan="M"` is encountered, its position gets a `C-tag`.
    *   The `M-1` cells to its right in the same row receive `L-tag`s.
    *   If the cell also has a `rowspan`, cells covered by both (excluding the first row/column of the span) get `X-tag`s.
*   The `dataset/preprocess_data.py` script calls `convert_html_to_otsl` with `loaded_data["html"]["structure"]["tokens"]` and `data_config["OTSL_TAG"]`, storing the result as `otsl_seq`.

### 3. Bounding Box Normalization/Rescaling (Subtask 3)

Bounding box coordinates (from OCR/text regions or gold cells) are rescaled to match the model's fixed input image dimensions. This process involves `prepare_image_tensor` from `tflop/datamodule/preprocess/image_utils.py` and `rescale_bbox` from `tflop/datamodule/preprocess/common_utils.py`.

*   **1. Image Resizing and Padding (`prepare_image_tensor`):**
    *   An input image is resized to fit within the model's target dimensions (`target_img_size`) while maintaining aspect ratio (using `PIL.Image.thumbnail`).
    *   Padding is then added to reach the exact `target_img_size`. `padding_dims = (left_pad, top_pad, right_pad, bottom_pad)` are calculated.
    *   This function returns the padded image tensor, original image size (`org_img_size`), and `padding_dims`.
*   **2. Role of `rescale_bbox`:**
    *   This function takes `list_of_coords`, `org_img_size`, `new_img_size` (the model's target image size), and `padding_dims`.
*   **3. Mathematical Transformation in `rescale_bbox`:**
    *   **Effective Content Dimensions:**
        *   `img_only_width = new_img_size[0] - padding_dims[0] - padding_dims[2]`
        *   `img_only_height = new_img_size[1] - padding_dims[1] - padding_dims[3]`
        These are the dimensions of the actual resized image content within the padded target image.
    *   **Scaling Factors:**
        *   `width_scale = img_only_width / org_img_size[0]`
        *   `height_scale = img_only_height / org_img_size[1]`
    *   **Transformation:** For each original coordinate `(x1, y1, x2, y2)`:
        *   `x_new = (x_orig * width_scale) + padding_dims[0]` (left padding)
        *   `y_new = (y_orig * height_scale) + padding_dims[1]` (top padding)
*   **4. Clipping:**
    *   The transformed coordinates are clipped (e.g., `max(0, x1_new)`, `min(new_img_size[0], x2_new)`) to ensure they stay within the boundaries of `new_img_size`.
*   **Usage:** In `TFLOPDataset.__getitem__` (in `tflop/datamodule/datasets/tflop.py`), `load_and_prepare_img_tensor` provides `org_img_size` and `padding_dims`. `rescale_bbox` is then called (either directly in `get_dr_pointer_label` or via `rescale_gold_coordinates`) to transform DR or gold coordinates.

### 4. Creation of Pointer Targets (Subtask 5 & 6)

"Pointer targets" link the structural `<td>` tags (represented by `C-tag`s in the OTSL sequence) to specific text region bounding boxes. This is a two-stage process:

*   **Stage 1: Initial Association (`dataset/preprocess_data.py`):**
    *   The `preprocess_det_bbox` function is key. It takes raw OCR boxes and ground-truth HTML cell structures.
    *   It matches OCR boxes to the HTML cells they geometrically belong to (using IoU/IoP).
    *   It groups all OCR boxes that fall within the same HTML cell.
    *   The output, stored as `dr_coord` in the JSONL files, is a dictionary where each entry (representing an original HTML cell with detected content) contains:
        *   A list of one or more OCR box coordinates for that cell.
        *   The index of the original HTML cell these OCR boxes correspond to.
*   **Stage 2: Final Pointer Label Generation (`tflop/datamodule/preprocess/common_utils.py`):**
    *   Functions: `get_dr_pointer_label` (for OCR regions) or `get_cell_pointer_label` (for gold cell regions). These are called by `TFLOPDataset.__getitem__`.
    *   **Inputs:** Tokenized OTSL sequence (`input_ids`), `dr_coord` (or `gold_cell_coords`), `gold_coord_isFilled` (marks if an original HTML cell should have content), and `bbox_token_cnt` (max number of target boxes).
    *   **Logic (`get_dr_pointer_label` as example):**
        1.  **Prepare Target Boxes:** `dr_coord` is processed. OCR boxes are rescaled. A flat, ordered list of these target OCR boxes (`new_filtered_dr_coords`) is created (typically sorted top-to-bottom, left-to-right via `serialize_bbox_top_left_bottom_right`). A parallel list (`new_org_index_list`) stores the original HTML cell index for each box in `new_filtered_dr_coords`. This list is truncated if it exceeds `bbox_token_cnt - 1`.
        2.  **Iterate OTSL:** The code iterates through `input_ids`. A `data_cell_index` tracks the N-th logical cell from the original HTML.
        3.  **Map `C-tag`:** When a `C-tag` is found:
            *   If `gold_coord_isFilled[data_cell_index]` is `False`, the `C-tag`'s pointer target is set to index 0 (empty cell).
            *   If `True`, it searches `new_org_index_list` for all entries matching `data_cell_index`. For each match, the index `k` (position in `new_filtered_dr_coords`) is used to set the `(k+1)`-th bit in the pointer target vector for the current `C-tag`. This can be multi-hot if one `C-tag` maps to multiple OCR boxes.
        *   The final `pointer_label` tensor has shape `(num_structural_OTSL_tokens, bbox_token_cnt)`.

### 5. Handling Multiple Text Regions in One `<td>` Cell (Subtask 14)

*   **`preprocess_det_bbox` (`dataset/preprocess_data.py`):**
    *   As described above, this function groups all OCR boxes detected within the geometric boundaries of the *same* ground-truth HTML `<td>` cell. The `dr_coord` structure stores these as a list of OCR boxes associated with that single HTML cell's original index.
*   **`get_dr_pointer_label` (`tflop/datamodule/preprocess/common_utils.py`):**
    *   When processing `dr_coord`, if an HTML cell (identified by `data_cell_index`) was associated with multiple OCR boxes, the `filtered_org_index_list` (before global sorting and truncation) will contain multiple entries for that `data_cell_index`, each pointing to one of the constituent OCR boxes in `filtered_dr_coords`.
    *   During pointer label creation, a single `C-tag` (representing that HTML cell) will then look up `data_cell_index` in `new_org_index_list`. If multiple matches are found (meaning multiple OCR boxes for that cell are still present in `new_filtered_dr_coords`), the pointer target vector for that `C-tag` will have '1's at all indices corresponding to these constituent OCR boxes. This results in a **multi-hot pointer target**.

### 6. Final Batch Assembly (`TFLOPDataset.__getitem__`) (Subtask 16)

The `__getitem__` method of `TFLOPDataset` in `tflop/datamodule/datasets/tflop.py` returns a tuple (which is then collated into a batch by the DataLoader). For training, this includes:

*   `img_tensor`: Processed image tensor.
*   `input_ids`: Token IDs of the input OTSL sequence (sliced for text part).
*   `coords_int_padded`: Integer coordinates `(x1,y1,x2,y2)` of text regions (target boxes for the pointer), including a dummy "empty cell" coord at index 0, padded to `bbox_token_cnt`.
*   `valid_coord_length`: Number of valid (non-dummy, non-padding) coordinates in `coords_int_padded`.
*   `token_pred_labels`: Target token IDs for language modeling (OTSL generation).
*   `pointer_label`: Boolean tensor (`num_OTSL_tokens` x `bbox_token_cnt`) linking `C-tag`s to `coords_int_padded` indices.
*   `pointer_mask_label`: Boolean tensor marking which OTSL tokens are `C-tag`s needing a pointer.
*   `chosen_bbox_coeff_tensor`: Coefficients for span-aware contrastive loss.

For validation, `token_pred_labels` is replaced with items like `prompt_end_index`, `input_parse` (full tokenized string), `html_with_content` (for TEDS), `cell_text_collated`, and `sample["file_name"]`.

## Section 2: The TFLOP Model Architecture

This section outlines the main components of the TFLOP model itself. The main model class is `TFLOP` in `tflop/model/model/TFLOP.py`, inheriting from `transformers.PreTrainedModel`.

### 1. Image Encoder (Subtask 8)

*   **Component:** `SwinEncoder`, defined in `tflop/model/visual_encoder/swin.py`.
*   **Instantiation in `TFLOP.__init__`:**
    ```python
    swin_input_size = (self.config.input_size[1], self.config.input_size[0]) # (height, width)
    self.encoder = SwinEncoder(
        input_size=swin_input_size,
        align_long_axis=self.config.align_along_axis,
        window_size=self.config.window_size,
        name_or_path=self.config.name_or_path, # For pretrained Swin weights
        encoder_layer=self.config.encoder_layer,
    )
    ```
*   **Internal Model:** `SwinEncoder` internally uses a `timm.models.swin_transformer.SwinTransformer`.
*   **Function:** Processes the input image tensor and outputs feature maps that are passed to the decoder for cross-attention.

### 2. Layout Encoder (Subtask 9)

The "Layout Encoder" is not a single distinct module but a collection of components within `MBARTDecoder` (defined in `tflop/model/decoder/mbart_decoder.py`) that process spatial and optional visual information from text regions.

*   **Coordinate Embedding Layers (in `MBARTDecoder.__init__`):**
    *   `self.x_coord_embedding = nn.Embedding(image_width + 4, d_model // 4, ...)`
    *   `self.y_coord_embedding = nn.Embedding(image_height + 4, d_model // 4, ...)`
    *   These embed the discrete integer coordinates (x1, y1, x2, y2) of bounding boxes into dense vectors. Concatenated, they form a `d_model` vector per box. This is done in `MBARTDecoder.embed_coord_tensor`.
*   **RoIAlign Components (Conditional, if `use_imgRoiAlign` is True, in `MBARTDecoder.__init__`):**
    *   `self.roi_align = RoIAlign(...)`: Extracts visual features from the image encoder's output based on bounding box locations.
    *   `self.roi_proj = nn.Sequential(...)`: An MLP that processes these RoI-aligned visual features.
*   **Merging MLPs (Conditional, in `MBARTDecoder.__init__`):**
    *   `self.bbox_coord_merge = nn.Sequential(...)`: MLP to process coordinate embeddings.
    *   `self.roi_merge = nn.Sequential(...)`: MLP to process projected RoI features.
    *   In `MBARTDecoder.forward`, the outputs are combined: `self.bbox_coord_merge(coord_embed) + self.roi_merge(roi_features)`.
*   **Function:** These components transform bounding box coordinates and (optionally) visual features from text regions into layout embeddings. These layout embeddings (referred to as `batch_coord_embedding` in `MBARTDecoder.forward`) are then concatenated with text token embeddings and fed to the structure decoder.

### 3. Logical Structure Decoder (Subtask 10)

*   **Component:** `MBARTDecoder`, defined in `tflop/model/decoder/mbart_decoder.py`.
*   **Instantiation in `TFLOP.__init__`:**
    ```python
    self.decoder = MBARTDecoder(
        tokenizer=self.tokenizer,
        decoder_layer=self.config.decoder_layer,
        max_length=self.config.max_length,
        # ... other relevant config parameters like bbox_token_cnt, input_size ...
    )
    ```
*   **Internal Model:** `MBARTDecoder` internally uses a `transformers.MBartForCausalLM` model, configured with `is_decoder=True` and `add_cross_attention=True`.
*   **Function:** An autoregressive transformer decoder that generates the OTSL sequence representing the table's logical structure. It attends to the image encoder's output (cross-attention) and the combined sequence of previously generated OTSL tokens and the layout embeddings (self-attention).

### 4. Layout Pointer Module (Subtask 11 & 12)

This module links the generated logical structure tokens (specifically `C-tag`s) to the corresponding layout region embeddings. It's part of `MBARTDecoder`.

*   **Pointer Head (Projection Layers in `MBARTDecoder.__init__`):**
    ```python
    self.k_linear = nn.Linear(
        self.model.config.d_model, self.model.config.d_model, bias=False
    ) # For Keys (layout embeddings)
    self.q_linear = nn.Linear(
        self.model.config.d_model, self.model.config.d_model, bias=False
    ) # For Queries (decoder hidden states for OTSL tokens)
    ```
*   **Dot-Product Similarity (in `MBARTDecoder.get_tag2coord_ptr_loss` for training, `TFLOP.get_dr_point_pred` for inference) (Subtask 13):**
    1.  Keys are derived from layout embeddings (e.g., `output_seq[:, :total_bbox_token_length]` in `get_tag2coord_ptr_loss`) passed through `k_linear`.
    2.  Queries are derived from decoder hidden states of OTSL tokens (e.g., `output_seq[:, total_bbox_token_length + 1 :]`) passed through `q_linear`.
    3.  Both are L2 normalized.
    4.  Similarity scores (logits) are computed via batched matrix multiplication: `data_combined_feat = torch.bmm(normalized_query_feature, normalized_key_feature.transpose(1, 2))`.
    5.  **Temperature Scaling (Training):** In `get_tag2coord_ptr_loss`, these logits (for non-empty cells) are divided by a `temperature` (default 0.1) before being used in `CrossEntropyLoss`.
*   **Function:** Predicts which layout region (bounding box embedding) a generated `C-tag` token refers to.

### 5. Special Embedding for Empty Cells (Subtask 15 & 17)

*   **Pointer Target:** In `get_dr_pointer_label` (and `get_cell_pointer_label`), if a `C-tag` corresponds to an empty cell, its pointer target in `pointer_label` is set to index 0.
*   **Embedding Construction (in `MBARTDecoder.forward`):**
    *   A `dummy_coord_embedding` is created by passing hardcoded, out-of-range integer coordinates (e.g., `image_width + 1`) through `self.embed_coord_tensor`.
    *   If `use_imgRoiAlign` is true, a separate learnable `self.dummy_embed` provides the visual features (`dummy_ROIAlign`).
    *   This "empty cell embedding" (derived from `dummy_coord_embedding` and potentially `dummy_ROIAlign`) is prepended to the sequence of embeddings of actual text regions. Thus, index 0 of the layout embeddings (keys for the pointer) corresponds to this special empty cell representation.
*   **Loss for Empty Cell Prediction (in `MBARTDecoder.get_tag2coord_ptr_loss`):**
    *   The logit for the empty cell slot is `is_data_only_pred[:, 0]` (similarity score with the empty cell embedding at key index 0).
    *   `nn.BCEWithLogitsLoss()` is applied to these logits and the corresponding labels (`is_data_only_label[:, 0]`) to calculate `is_empty_loss`. This is part of `L_ptr`.

## Section 3: The Training and Evaluation Loop

This section describes how the TFLOP model is trained and evaluated.

### 1. Main Training Script (Subtask 17)

*   **Script:** `train.py` is the main script for launching and managing the training process.
*   **Framework:** It uses **PyTorch Lightning (`pytorch_lightning as pl`)**.
    *   It defines `TFLOPModelPLModule` (a `pl.LightningModule`) and `DataPLModule` (a `pl.LightningDataModule`).
    *   Training is orchestrated by `pl.Trainer`, with the main call being `trainer.fit(model_module, data_module)`.
*   **Execution:** The script `scripts/training/train_pubtabnet.sh` executes `python train.py` with necessary configurations.

### 2. Key Arguments for Training Script (Subtask 18)

*   **`--exp_config` (str, required):** Path to the experiment configuration YAML file (e.g., `config/exp_configs/general_exp.yaml`). Contains model hyperparameters, learning rate, etc.
*   **`--data_config` (str, required):** Path to the data configuration YAML file (e.g., `config/exp_configs/data_pubtabnet.yaml`). Contains dataset paths, batch sizes, etc.
*   **Other CLI arguments:** Additional parameters can be passed as `key=value` pairs, which are merged into the OmegaConf object (e.g., `max_length=1376`, `lr=0.00008`).

### 3. Optimizer and Learning Rate Scheduler (Subtask 19)

Defined in `TFLOPModelPLModule.configure_optimizers` (in `tflop/lightning_module/lightning_module.py`):

*   **Optimizer:** `torch.optim.Adam`
    ```python
    optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
    ```
*   **Learning Rate Scheduler:** `torch.optim.lr_scheduler.LambdaLR` implementing a cosine decay schedule with linear warmup.
    *   Instantiated via `self.cosine_scheduler(optimizer, training_steps, warmup_steps)`.
    *   The `cosine_scheduler` static method defines an `lr_lambda` function that:
        1.  Performs linear warmup from 0 to base LR for `warmup_steps` (typically 2% of `training_steps`).
        2.  Performs cosine decay from base LR towards 0 for the remaining steps.

### 4. Span-aware Contrastive Supervision (L_contr) (Subtask 14, 15, 20)

*   **Loss Module:** `TableCL` class in `tflop/loss.py`. It internally uses `SupConLoss` (also in `tflop/loss.py`).
*   **Instantiation:** `self.TableCL_loss = TableCL(temperature=0.1)` in `MBARTDecoder.__init__` if `use_bbox_HiMulConET` is true.
*   **Coefficient Matrix Generation (`bbox_coeff_tensor`):**
    *   Functions `get_rowwise_HiMulConET_Coeff` and `get_columnwise_HiMulConET_Coeff` in `tflop/datamodule/preprocess/hi_mul_con_table.py` are responsible.
    *   `breakdown_otsl_seq` (in `hi_mul_con_table.py`) parses the OTSL sequence into a 2D grid, identifying cell spans (`num_rows_spanned`, `num_cols_spanned` - these are `S_p` and `S_j`).
    *   These functions then iterate through this grid. For cells in the same row/column, they are positive pairs.
    *   **Span Coefficient `c_p(j)` (Subtask 21):** For spanning cells, `eval_rowspan_coeff` and `eval_colspan_coeff` (in `hi_mul_con_table.py`) calculate the coefficient. If `mode="proportional"`, this is `(overlap_count / S_p) * (overlap_count / S_j)`. This value weights the positive relationship.
    *   The resulting coefficient matrices are stacked into `bbox_coeff_tensor` by `get_dr_pointer_label` or `get_cell_pointer_label` in `common_utils.py`.
*   **Loss Calculation (`SupConLoss.forward` in `tflop/loss.py`):**
    *   The `bbox_coeff_tensor` slice (e.g., row-wise mask) is passed as `mask`.
    *   `dot_contrast = torch.div(torch.matmul(features, features.transpose(1, 2)), self.temperature)` calculates pairwise similarities of bounding box features.
    *   `log_prob` is calculated based on these similarities (log_softmax).
    *   The loss for an anchor `j` (`L_contr,j`) is `(mask[j,:] * log_prob[j,:]).sum() / (mask[j,:].sum() + 1e-6)`, effectively summing `c_p(k) * log_prob(j,k)` over positive pairs `k` and normalizing.
    *   The final loss is `-1 * L_contr,j.mean()`.
*   This loss is computed in `MBARTDecoder.get_bbox_TableCL_loss` and added to the total training objective.

### 5. Combination of All Loss Components (Subtask 22)

In `MBARTDecoder.forward`:

1.  **`token_cls_loss` (L_cls):** Calculated by `self.get_token_classification_loss()`.
2.  **`tag2coord_pointer_loss` (L_ptr):** Calculated by summing weighted `is_empty_loss` (BCE for empty slot) and `is_not_empty_loss` (CrossEntropy for non-empty slots) from `self.get_tag2coord_ptr_loss()`. Coefficients are `self.empty_cell_ptr_loss_coeff` and `self.non_empty_cell_ptr_loss_coeff`. (Pointer loss L_ptr includes L_empty and L_cell as per paper's Eq 2,3,4).
3.  **`bbox_TableCL_loss` (L_contr):** Calculated by `self.get_bbox_TableCL_loss()`. This function returns individual row-wise and column-wise contrastive losses. These are summed and then scaled by `1 / sum(self.contrastive_loss_config.values())`. If both row and column contrastive losses are active (config value 1.0 each), this means `bbox_TableCL_loss = 0.5 * L_contr_row + 0.5 * L_contr_col`.

**Final Aggregation:**
```python
loss = token_cls_loss + tag2coord_pointer_loss
if self.use_bbox_HiMulConET:
    loss += bbox_TableCL_loss
```
The implicit weight for L_cls and L_ptr is 1.0. The `lambda_contr_row` and `lambda_contr_col` are effectively 0.5 each if both are enabled, or 1.0 if only one is (due to the normalization by the sum of their config flags).

### 6. Inference Process and Pointer Usage (Subtask 20)

*   **`TFLOP.inference` Method:**
    1.  Image features are obtained from `self.encoder`.
    2.  The core structure generation is done by `self.decoder.model.generate(...)`. This autoregressively generates the OTSL token sequence.
        *   During generation, the trained pointer mechanism (weights of `k_linear`, `q_linear` and transformer layers) implicitly influences the choice of the next OTSL token by allowing attention to layout embeddings.
    3.  **Explicit Pointer Prediction:** After the OTSL sequence is generated, `TFLOP.get_dr_point_pred(last_hidden_states, generated_sequence)` is called.
        *   It calculates query features (from generated OTSL token hidden states via `self.decoder.q_linear`) and key features (from layout embedding hidden states via `self.decoder.k_linear`).
        *   Computes dot-product similarity: `logits = query @ key.T`.
        *   For each `C-tag` query, `torch.argmax(logits_for_physical_boxes, dim=1)` selects the specific text region index with the highest similarity.
        *   A check for whether any physical box was selected determines if the cell points to "empty" (index 0) or the selected physical box.

### 7. Final HTML String Construction (Subtask 23)

*   **Key Function:** `decode_OTSL_seq(otsl_token_seq, pointer_tensor, cell_text_data)` in `tflop/utils.py`.
*   **Inputs:**
    *   `otsl_token_seq`: The generated sequence of OTSL tokens.
    *   `pointer_tensor`: The output from `TFLOP.get_dr_point_pred`, indicating which text region index each `C-tag` points to.
    *   `cell_text_data`: A list of the actual text strings from all unique input detected regions.
*   **Process:**
    1.  **Parses OTSL:** Decodes `L-tag`, `U-tag`, `X-tag` to determine rowspan/colspan attributes for each `C-tag`.
    2.  **Retrieves Text:** For each `C-tag`, uses `pointer_tensor` to find the index `k` in `cell_text_data`. If `k=0` (empty), text is empty. If `k>0`, `cell_text_data[k-1]` is used. Multiple pointed texts are concatenated.
    3.  **Constructs HTML:** Iterates through the OTSL structure, converting `C-tag` to `<td>` (adding span attributes and retrieved text) and `NL-tag` to `</tr><tr>`, etc.
*   **Cleanup:** `custom_format_html` (in `tflop/utils.py`) performs final cleanup (removing special tokens) and wraps the table in `<html><body><table>...</table></body></html>`.

### 8. TEDS Evaluation (Subtask 19 & 24)

*   **Primary Script:** `evaluate_ted.py`. This script is run offline on saved model predictions.
*   **Metric Implementation:** The `TEDS` class in `tflop/evaluator.py`.
    *   **External Library:** It uses the `apted` library (`from apted import APTED`) for the core tree edit distance algorithm.
    *   **Custom Logic:**
        *   Parses HTML strings into `lxml` trees.
        *   Converts `lxml` trees to a custom `TableTree` structure suitable for `apted`.
        *   Uses `CustomConfig` to define node comparison costs for `apted`, considering HTML tags, `colspan`/`rowspan` attributes, and cell content similarity (via Levenshtein distance if not `structure_only`).
    *   **`TEDS.evaluate(pred_html, true_html)`:** This method orchestrates the process: parses HTML, converts to `TableTree`, calls `APTED(...).compute_edit_distance()`, and normalizes the result to get the TEDS score `1.0 - (distance / max_nodes)`.
*   `evaluate_ted.py` loads predicted and ground-truth HTML strings from JSON files and uses the `TEDS` class to compute and report scores, often in parallel. Full TEDS is generally not computed live during training validation loops due to its computational expense.

This concludes the detailed walkthrough.
