def convert_html_to_otsl(html_seq, otsl_tag_maps):
    """
    Convert list of html tokens to OTSL format

    Args:
        html_seq List[str]: list of html tokens
            E.g. ['<thead>', '<tr>', '<td>', '</td>', ...]
        ost_tag_maps Dict[str, str]: mapping of otsl tag symbols
            E.g.{"C": "C-tag", "L": "L-tag", "U": "U-tag", "X": "X-tag", "NL": "NL-tag"}

    Returns:
        List[str]: Full OTSL sequence
        Int: Number of rows in the table
        Int: Number of columns in the table
    """

    # 1. Split list of HTML tokens into head and body
    end_of_head_index = html_seq.index("</thead>")
    thead_seq = html_seq[:end_of_head_index]
    thead_seq = [
        x for x in thead_seq if x not in ["<thead>", "<tbody>", "</thead>", "</tbody>"]
    ]
    tbody_seq = html_seq[(end_of_head_index + 1) :]
    tbody_seq = [
        x for x in tbody_seq if x not in ["<thead>", "<tbody>", "</thead>", "</tbody>"]
    ]

    # 2. Format HTML tags into row-wise list of tokens
    thead_row_wise_seq = get_row_wise(thead_seq)
    tbody_row_wise_seq = get_row_wise(tbody_seq)

    # 2.1 Check if the thead section is empty
    is_head_empty = False
    if len(thead_row_wise_seq) == 0:
        is_head_empty = True

    # 3. Convert row-wise list of tokens into OTSL array -> 4. Convert OTSL array into OTSL sequence
    thead_OTSL_array, num_head_rows, num_cols = None, 0, None
    thead_OTSL_seq = []
    if not is_head_empty:
        thead_OTSL_array, num_head_rows, num_cols = get_OTSL_array(
            thead_row_wise_seq, otsl_tag_maps
        )
        thead_OTSL_seq = convert_OTSL_array_to_OTSL_seq(
            thead_OTSL_array,
            num_rows=num_head_rows,
            num_cols=num_cols,
            otsl_tag_maps=otsl_tag_maps,
        )

    tbody_OTSL_array, num_body_rows, num_cols = get_OTSL_array(
        tbody_row_wise_seq, otsl_tag_maps, ref_num_cols=num_cols
    )
    tbody_OTSL_seq = convert_OTSL_array_to_OTSL_seq(
        tbody_OTSL_array,
        num_rows=num_body_rows,
        num_cols=num_cols,
        otsl_tag_maps=otsl_tag_maps,
    )

    # 5. Combine thead and tbody into one OTSL sequence
    combined_OTSL_seq = (
        ["<thead>"]
        + thead_OTSL_seq
        + ["</thead>", "<tbody>"]
        + tbody_OTSL_seq
        + ["</tbody>"]
    )
    num_rows = num_head_rows + num_body_rows

    return combined_OTSL_seq, num_rows, num_cols


def get_OTSL_array(row_wise_html_tags, otsl_tag_maps, ref_num_cols=None):
    """Generate OTSL array from row-wise html tags.

    Args:
        row_wise_html_tags List[List[str]]: list of list of html tags, where each inner list is a row
            E.g. [['<td>', '</td>'], ['<td>', '</td>']]
        otsl_tag_maps Dict[str, str]: mapping of otsl tag symbols
            E.g.{"C": "C-tag", "L": "L-tag", "U": "U-tag", "X": "X-tag", "NL": "NL-tag"}
        ref_num_cols int: reference number of columns to use. If None, will derive from row_wise_html_tags
            - Used to sanity check if tbody's num_cols match that of thead

    Returns:
        Tuple[List[List[str]], int, int]: OTSL array, number of rows, number of columns
    """
    num_rows, num_cols = get_num_rows_and_cols(row_wise_html_tags)
    if ref_num_cols is not None and ref_num_cols != num_cols:
        raise ValueError(
            "Number of columns in tbody does not match that of thead. Got %s but expected %s"
            % (num_cols, ref_num_cols)
        )

    # 1. Initialize OTSL array
    otsl_array = [list([None] * num_cols) for _ in range(num_rows)]

    # 2. Fill in OTSL array
    curr_row_ind, curr_col_ind = 0, 0
    current_data = {"standard": 0, "rowspan": 0, "colspan": 0}
    for row_tokens in row_wise_html_tags:
        for tok_i, tok in enumerate(row_tokens):
            # 2.1 sanity check token
            if tok not in ["<td>", "</td>", "<td", ">"] and (
                "rowspan" not in tok and "colspan" not in tok
            ):
                raise ValueError("Invalid HTML %s" % tok)

            # 2.2 iter over tokens in the row
            if tok in ["<td", ">"]:
                continue
            elif tok == "<td>":
                current_data["standard"] += 1
            elif "rowspan" in tok:
                current_data["rowspan"] += int(tok.split("=")[1].split('"')[1])
            elif "colspan" in tok:
                current_data["colspan"] += int(tok.split("=")[1].split('"')[1])
            elif tok == "</td>":
                # End of cell -> i.e. Time to start updating OTSL array with current_data
                # 2.2.1 Find row & col ind to insert data
                while otsl_array[curr_row_ind][curr_col_ind] is not None:
                    curr_col_ind += 1
                    if curr_col_ind >= num_cols:
                        curr_col_ind = 0
                        curr_row_ind += 1
                        assert (
                            curr_row_ind < num_rows
                        ), "curr_row_ind %s >= num_rows %s" % (curr_row_ind, num_rows)

                # 2.2.2 Sanity check current_data before insertion
                sanity_check_move(current_data)

                # 2.2.3 Insert data
                otsl_array = insert_data_into_OTSL(
                    current_data=current_data,
                    OTSL_array=otsl_array,
                    otsl_tag_maps=otsl_tag_maps,
                    curr_row_ind=curr_row_ind,
                    curr_col_ind=curr_col_ind,
                )

                # 2.2.4 reset current_data
                current_data = {"standard": 0, "rowspan": 0, "colspan": 0}

            else:
                raise ValueError("Invalid HTML %s" % tok)

    return otsl_array, num_rows, num_cols


def convert_OTSL_array_to_OTSL_seq(otsl_array, num_rows, num_cols, otsl_tag_maps):
    """Convert OTSL array to OTSL sequence.

    Args:
        otsl_array List[List[str]]: OTSL array
        num_rows int: number of rows in OTSL array
        num_cols int: number of columns in OTSL array
        otsl_tag_maps Dict[str, str]: mapping of otsl tag symbols
            E.g.{"C": "C-tag", "L": "L-tag", "U": "U-tag", "X": "X-tag", "NL": "NL-tag"}

    Returns:
        List[str]: OTSL sequence
    """
    OTSL_seq = []

    for row_ind in range(num_rows):
        for col_ind in range(num_cols):
            assert (
                otsl_array[row_ind][col_ind] is not None
            ), "row_ind %s, col_ind %s" % (row_ind, col_ind)
            OTSL_seq.append(otsl_array[row_ind][col_ind])

        OTSL_seq.append(otsl_tag_maps["NL"])

    return OTSL_seq


# -----Auxiliary Functions-----#
def get_row_wise(tok_list):
    """Given list of HTML tokens, group them into row-wise format.

    NOTE:
        Raises error if there are tokens not encapsulated by <tr></tr>

    Args:
        tok_list List[str]: list of html tokens
            E.g. ['<tr>', '<td>', '</td>', ...]

    Returns:
        List[List[str]]: list of list of tokens, where each inner list is a row
    """
    row_wise_tokens = []

    is_within_row = False
    for tok in tok_list:
        if tok == "<tr>":
            is_within_row = True
            tmp_row = []
        elif tok == "</tr>":
            is_within_row = False
            row_wise_tokens.append(tmp_row)
        else:
            assert is_within_row, "Token not encapsulated by <tr></tr>"
            tmp_row.append(tok)

    return row_wise_tokens


def get_num_rows_and_cols(row_wise_html_tags):
    """Given row-wise html tags, derive number of rows and columns.

    Args:
        row_wise_html_tags List[List[str]]: list of list of html tags, where each inner list is a row
            E.g. [['<td>', '</td>'], ['<td>', '</td>']]

    Returns:
        Tuple[int, int]: number of rows and columns
    """

    # Derive the number of rows in this table
    num_rows = len(row_wise_html_tags)
    num_cols = 0
    col_span_tracker = 0

    # Derive the number of columns in this table
    for first_row_tok in row_wise_html_tags[0]:
        if first_row_tok == "</td>":
            if col_span_tracker == 0:
                num_cols += 1
            else:
                num_cols += col_span_tracker
                col_span_tracker = 0
        else:
            if "colspan" in first_row_tok:
                col_span_tracker += int(first_row_tok.split("=")[1].split('"')[1])

    return num_rows, num_cols


def sanity_check_move(current_data):
    """Sanity checker of current move data prior to updating OTSL array.

    Args:
        current_data Dict: current data of move
            E.g. {'standard': 1, 'rowspan': 0, 'colspan': 0}

    Checks:
        1. If standard (i.e. single cell), then rowspan and colspan must be 0
        2. If not standard, then rowspan or colspan must be > 0
    """

    if current_data["standard"] == 0:
        assert sum([current_data["rowspan"], current_data["colspan"]]) > 0
    else:
        assert current_data["standard"] == 1
        assert sum([current_data["rowspan"], current_data["colspan"]]) == 0


def insert_data_into_OTSL(
    current_data, OTSL_array, otsl_tag_maps, curr_row_ind, curr_col_ind
):
    """Given current_data, insert data into OTSL array.

    Args:
        current_data Dict: current data of move
            E.g. {'standard': 1, 'rowspan': 0, 'colspan': 0}
        OTSL_array List[List[str]]: OTSL array
        otsl_tag_maps Dict: mapping of otsl tag symbols
        curr_row_ind int: current row index
        curr_col_ind int: current column index

    NOTE:
        This function updates the OTSL array based on the current_data.
        There are 4 cases in total:
            1. Standard cell (i.e. single cell, no rowspan or colspan)
            2. Colspan only
            3. Rowspan only
            4. Both rowspan and colspan

    Returns:
        List[List[str]]: updated OTSL array
    """

    if current_data["standard"] == 1:
        assert OTSL_array[curr_row_ind][curr_col_ind] is None
        OTSL_array[curr_row_ind][curr_col_ind] = otsl_tag_maps[
            "C"
        ]  # single cell mapped as 'C' in OTSL
    else:
        # Colspan only
        if current_data["rowspan"] == 0:
            assert OTSL_array[curr_row_ind][curr_col_ind] is None
            OTSL_array[curr_row_ind][curr_col_ind] = otsl_tag_maps["C"]
            for i in range(1, current_data["colspan"]):
                assert OTSL_array[curr_row_ind][curr_col_ind + i] is None
                OTSL_array[curr_row_ind][curr_col_ind + i] = otsl_tag_maps[
                    "L"
                ]  # All cells other than root for colspan mapped as 'L' in OTSL

        # Rowspan only
        elif current_data["colspan"] == 0:
            assert OTSL_array[curr_row_ind][curr_col_ind] is None
            OTSL_array[curr_row_ind][curr_col_ind] = otsl_tag_maps["C"]
            for i in range(1, current_data["rowspan"]):
                assert OTSL_array[curr_row_ind + i][curr_col_ind] is None
                OTSL_array[curr_row_ind + i][curr_col_ind] = otsl_tag_maps[
                    "U"
                ]  # All cells other than root for rowspan mapped as 'U' in OTSL

        # Both rowspan and colspan
        else:
            assert OTSL_array[curr_row_ind][curr_col_ind] is None
            OTSL_array[curr_row_ind][curr_col_ind] = otsl_tag_maps["C"]

            for i in range(1, current_data["colspan"]):
                assert OTSL_array[curr_row_ind][curr_col_ind + i] is None
                OTSL_array[curr_row_ind][curr_col_ind + i] = otsl_tag_maps["L"]

            for i in range(1, current_data["rowspan"]):
                assert OTSL_array[curr_row_ind + i][curr_col_ind] is None
                OTSL_array[curr_row_ind + i][curr_col_ind] = otsl_tag_maps["U"]

            for i in range(1, current_data["rowspan"]):
                for j in range(1, current_data["colspan"]):
                    assert OTSL_array[curr_row_ind + i][curr_col_ind + j] is None
                    OTSL_array[curr_row_ind + i][curr_col_ind + j] = otsl_tag_maps["X"]

    return OTSL_array


def calculate_pointer_index(
    curr_row_ind, curr_col_ind, row_offset, col_offset, is_table_body, num_cols
):
    """Given current row & col index, along with other info, calculate the index to point to for potsl."""

    # Apply offset values
    point_index = (curr_row_ind - row_offset) * num_cols + (curr_col_ind - col_offset)

    # Add number of rows to pointer as each row ends with NL tag
    point_index += curr_row_ind - row_offset

    # If current table is tbody, offset by 3 since <thead>, </thead>, <tbody> tags are added
    # else, offset by 1 since <thead>
    if is_table_body:
        point_index += 3
    else:
        point_index += 1

    return point_index
