# Script for implementing Hierarchical Multi-Label Contrastive Learning in Tables (HiMulConET)
import torch


def get_rowwise_HiMulConET_Coeff(
    sliced_input_ids,
    tokenizer,
    bbox_token_cnt,
    is_data_tensor,
    tag2coord_map,
    rep_mode="OTSL",
    rowspan_coeff_mode="constant",
):
    """
    Function to get rowwise HiMulConET Coefficient Matrix

    Args:
        sliced_input_ids (torch.Tensor): [seq_len - bbox_token_cnt - 1],
        tokenizer (transformers.tokenization_bert.BertTokenizer): tokenizer
        bbox_token_cnt (int): number of bbox tokens used in the model
        tag2coord_map (dict): mapping of sliced_input_ids index to bbox token index
        rep_mode (str): representation mode, only "OTSL" is supported
        rowspan_coeff_mode (str): coefficient mode for rowspan, choice of ["constant", "proportional"]
    """
    # sanity check
    assert rep_mode in [
        "OTSL",
    ], "Non-OTSL modes are deprecated"

    rowwise_HiMulConET_Coeff = torch.zeros((bbox_token_cnt))
    rowspan_HiMulConET_Coeff = torch.zeros((bbox_token_cnt, bbox_token_cnt))
    table_breakdown = breakdown_otsl_seq(sliced_input_ids.tolist(), tokenizer)

    table_data_idx = (is_data_tensor).nonzero(as_tuple=True)[0]
    # First, associate each cell in table_breakdown with bbox_indices
    table_breakdown_row_id, table_breakdown_col_id = 0, 0
    for data_token_id in table_data_idx:
        while table_breakdown[table_breakdown_row_id][table_breakdown_col_id] is None:
            table_breakdown_col_id += 1
            if table_breakdown_col_id >= len(table_breakdown[table_breakdown_row_id]):
                table_breakdown_row_id += 1
                table_breakdown_col_id = 0

        if data_token_id.item() not in tag2coord_map:
            table_breakdown_col_id += 1
            if table_breakdown_col_id >= len(table_breakdown[table_breakdown_row_id]):
                table_breakdown_row_id += 1
                table_breakdown_col_id = 0
            continue

        bbox_indices = [
            x + 1 for x in tag2coord_map[data_token_id.item()]
        ]  # add 1 as the first bbox token is for empty cells
        table_breakdown[table_breakdown_row_id][table_breakdown_col_id].append(
            bbox_indices
        )

        table_breakdown_col_id += 1
        if table_breakdown_col_id >= len(table_breakdown[table_breakdown_row_id]):
            table_breakdown_row_id += 1
            table_breakdown_col_id = 0

    column_count = len(table_breakdown[0])
    if len(table_breakdown[-1]) < column_count:
        table_breakdown[-1].extend(
            [None for _ in range(column_count - len(table_breakdown[-1]))]
        )

    # Second, use table_breakdown to fill in rowwise_HiMulConET_Coeff
    for table_row_id, table_row in enumerate(table_breakdown):
        for table_col_id, table_cell in enumerate(table_row):
            if table_cell is None or len(table_cell) == 2:
                continue

            if table_cell[0] > 1:  # this is rowspan
                rowspan_cnt = table_cell[0]
                span_cell_bbox_indices = table_cell[2]
                rowspan_HiMulConET_Coeff[
                    torch.tensor(span_cell_bbox_indices).unsqueeze(1),
                    span_cell_bbox_indices,
                ] = 1

                for rowspan_id in range(rowspan_cnt):
                    tmp_table_row = table_breakdown[table_row_id + rowspan_id]
                    for tmp_col_id, tmp_cell in enumerate(tmp_table_row):
                        if tmp_cell is None or len(tmp_cell) == 2:
                            continue
                        if tmp_col_id == table_col_id:
                            continue
                        # cells in a row within the rowspan
                        num_rows_remaining = rowspan_cnt - rowspan_id
                        row_overlap_cnt = min(num_rows_remaining, tmp_cell[0])
                        src2tar_coeff, tar2src_coeff = eval_rowspan_coeff(
                            rowspan_id,
                            src_row_span_cnt=rowspan_cnt,
                            tar_row_span_cnt=tmp_cell[0],
                            row_overlap_cnt=row_overlap_cnt,
                            mode=rowspan_coeff_mode,
                        )

                        rowspan_HiMulConET_Coeff[
                            torch.tensor(span_cell_bbox_indices).unsqueeze(1),
                            tmp_cell[2],
                        ] = src2tar_coeff
                        rowspan_HiMulConET_Coeff[
                            torch.tensor(tmp_cell[2]).unsqueeze(1),
                            span_cell_bbox_indices,
                        ] = tar2src_coeff
            else:
                rowwise_HiMulConET_Coeff[table_cell[2]] = table_row_id + 1

    # Third, get coeff matrix
    rowwise_HiMulConET_Coeff = rowwise_HiMulConET_Coeff.unsqueeze(
        -1
    )  # [bbox_token_cnt, 1]
    match_index = torch.eq(
        rowwise_HiMulConET_Coeff, rowwise_HiMulConET_Coeff.T
    )  # [bbox_token_cnt, bbox_token_cnt]
    match_index[rowwise_HiMulConET_Coeff.squeeze(-1) == 0, :] = (
        0  # empty token should not be matched with any other tokens
    )
    match_index = match_index.type(torch.float)

    # Lastly, combine match_index and rowspan_HiMulConET_Coeff
    match_index = match_index + rowspan_HiMulConET_Coeff

    return match_index


def get_columnwise_HiMulConET_Coeff(
    sliced_input_ids,
    tokenizer,
    bbox_token_cnt,
    is_data_tensor,
    tag2coord_map,
    rep_mode="OTSL",
    colspan_coeff_mode="constant",
):
    """
    Function to get columnwise HiMulConET Coefficient Matrix

    Args:
        sliced_input_ids (torch.Tensor): [seq_len - bbox_token_cnt - 1],
        tokenizer (transformers.tokenization_bert.BertTokenizer): tokenizer
        bbox_token_cnt (int): number of bbox tokens used in the model
        tag2coord_map (dict): mapping of sliced_input_ids index to bbox token index
        rep_mode (str): representation mode, only "OTSL" is supported
        colspan_coeff_mode (str): coefficient mode for colspan, choice of ["constant", "proportional"]

    """

    # sanity check
    assert rep_mode in [
        "OTSL",
    ], "Non-OTSL modes are deprecated"

    columnwise_HiMulConET_Coeff = torch.zeros((bbox_token_cnt))
    colspan_HiMulConET_Coeff = torch.zeros((bbox_token_cnt, bbox_token_cnt))
    table_breakdown = breakdown_otsl_seq(sliced_input_ids.tolist(), tokenizer)

    table_data_idx = (is_data_tensor).nonzero(as_tuple=True)[0]
    # First, associate each cell in table_breakdown with bbox_indices
    table_breakdown_row_id, table_breakdown_col_id = 0, 0
    for data_token_id in table_data_idx:
        while table_breakdown[table_breakdown_row_id][table_breakdown_col_id] is None:
            table_breakdown_col_id += 1
            if table_breakdown_col_id >= len(table_breakdown[table_breakdown_row_id]):
                table_breakdown_row_id += 1
                table_breakdown_col_id = 0

        if data_token_id.item() not in tag2coord_map:
            table_breakdown_col_id += 1
            if table_breakdown_col_id >= len(table_breakdown[table_breakdown_row_id]):
                table_breakdown_row_id += 1
                table_breakdown_col_id = 0
            continue

        bbox_indices = [x + 1 for x in tag2coord_map[data_token_id.item()]]
        table_breakdown[table_breakdown_row_id][table_breakdown_col_id].append(
            bbox_indices
        )

        table_breakdown_col_id += 1
        if table_breakdown_col_id >= len(table_breakdown[table_breakdown_row_id]):
            table_breakdown_row_id += 1
            table_breakdown_col_id = 0

    # Second, transpose table_breakdown to get columnwise information
    column_count = len(table_breakdown[0])
    if len(table_breakdown[-1]) < column_count:
        table_breakdown[-1].extend(
            [None for _ in range(column_count - len(table_breakdown[-1]))]
        )
    table_breakdown = list(map(list, zip(*table_breakdown)))

    # Third, use table_breakdown to fill in columnwise_HiMulConET_Coeff
    for table_col_id, table_col in enumerate(table_breakdown):
        for table_row_id, table_cell in enumerate(table_col):
            if table_cell is None or len(table_cell) == 2:
                continue

            if table_cell[1] > 1:  # this is colspan
                colspan_cnt = table_cell[1]
                span_cell_bbox_indices = table_cell[2]
                colspan_HiMulConET_Coeff[
                    torch.tensor(span_cell_bbox_indices).unsqueeze(1),
                    span_cell_bbox_indices,
                ] = 1

                for colspan_id in range(colspan_cnt):
                    tmp_table_col = table_breakdown[table_col_id + colspan_id]
                    for tmp_row_id, tmp_cell in enumerate(tmp_table_col):
                        if tmp_cell is None or len(tmp_cell) == 2:
                            continue
                        if tmp_row_id == table_row_id:
                            continue
                        # cells in a column within the colspan
                        num_cols_remaining = colspan_cnt - colspan_id
                        col_overlap_cnt = min(num_cols_remaining, tmp_cell[1])
                        src2tar_coeff, tar2src_coeff = eval_colspan_coeff(
                            colspan_id,
                            src_col_span_cnt=colspan_cnt,
                            tar_col_span_cnt=tmp_cell[1],
                            col_overlap_cnt=col_overlap_cnt,
                            mode=colspan_coeff_mode,
                        )
                        colspan_HiMulConET_Coeff[
                            torch.tensor(span_cell_bbox_indices).unsqueeze(1),
                            tmp_cell[2],
                        ] = src2tar_coeff
                        colspan_HiMulConET_Coeff[
                            torch.tensor(tmp_cell[2]).unsqueeze(1),
                            span_cell_bbox_indices,
                        ] = tar2src_coeff
            else:
                columnwise_HiMulConET_Coeff[table_cell[2]] = table_col_id + 1

    # Fourth, get coeff matrix
    columnwise_HiMulConET_Coeff = columnwise_HiMulConET_Coeff.unsqueeze(-1)
    match_index = torch.eq(columnwise_HiMulConET_Coeff, columnwise_HiMulConET_Coeff.T)
    match_index[columnwise_HiMulConET_Coeff.squeeze(-1) == 0, :] = 0
    match_index = match_index.type(torch.float)

    # Lastly, combine match_index and colspan_HiMulConET_Coeff
    match_index = match_index + colspan_HiMulConET_Coeff

    return match_index


# ----------------- Auxiliary Functions -----------------#
def breakdown_otsl_seq(tok_input_ids, tokenizer):
    """Convert OTSL sequence into a list of lists, where each list is a row of the table

    NOTE:
        # of list matches number of rows, and # of elements in each list matches number of columns
        Each element is either None or [num_rows, num_cols]

    Args:
        tok_input_ids (list): list of token ids
        tokenizer (transformers.tokenization_bert.BertTokenizer): tokenizer

    Returns:
        otsl_full_compilation (list): list of lists
            - each list is a row of the table
            - each element is either None or [num_rows, num_cols]
    """
    otsl_token_list = tokenizer.convert_ids_to_tokens(tok_input_ids)
    otsl_full_compilation, otsl_row_compilation = [], []
    curr_column_index = 0
    for tok_ind, tok in enumerate(otsl_token_list):
        if tok == "C-tag":
            otsl_row_compilation.append([1, 1])  # [num_rows, num_cols]
            curr_column_index += 1
        elif tok == "NL-tag":
            otsl_full_compilation.append(otsl_row_compilation)
            otsl_row_compilation = []
            curr_column_index = 0
        elif tok == "L-tag":
            for col_i in range(len(otsl_row_compilation)):
                # traverse backwards
                if otsl_row_compilation[-1 - col_i] is not None:
                    otsl_row_compilation[-1 - col_i][1] += 1
                    break
            otsl_row_compilation.append(None)
            curr_column_index += 1
        elif tok == "U-tag":
            for row_i in range(len(otsl_full_compilation)):
                # traverse backwards
                if otsl_full_compilation[-1 - row_i][curr_column_index] is not None:
                    otsl_full_compilation[-1 - row_i][curr_column_index][0] += 1
                    break
            otsl_row_compilation.append(None)
            curr_column_index += 1
        elif tok == "X-tag":
            otsl_row_compilation.append(None)
            curr_column_index += 1
        else:
            continue

    if len(otsl_row_compilation) > 0:
        otsl_full_compilation.append(otsl_row_compilation)

    return otsl_full_compilation


def eval_rowspan_coeff(
    curr_row_id, src_row_span_cnt, tar_row_span_cnt, row_overlap_cnt, mode="constant"
):
    # sanity check
    assert mode in ["constant", "proportional"]

    if mode == "constant":
        return 1, 1
    elif mode == "proportional":
        src2tar_coeff = row_overlap_cnt / src_row_span_cnt
        tar2src_coeff = row_overlap_cnt / tar_row_span_cnt
        coeff_val = src2tar_coeff * tar2src_coeff
        return coeff_val, coeff_val

    else:
        raise NotImplementedError


def eval_colspan_coeff(
    curr_col_id, src_col_span_cnt, tar_col_span_cnt, col_overlap_cnt, mode="constant"
):
    # sanity check
    assert mode in ["constant", "proportional"]

    if mode == "constant":
        return 1, 1
    elif mode == "proportional":
        src2tar_coeff = col_overlap_cnt / src_col_span_cnt
        tar2src_coeff = col_overlap_cnt / tar_col_span_cnt
        coeff_val = src2tar_coeff * tar2src_coeff
        return coeff_val, coeff_val

    else:
        raise NotImplementedError
