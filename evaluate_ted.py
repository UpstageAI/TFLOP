import argparse
import json
import multiprocessing
import os
import time

from Levenshtein import distance
from tqdm import tqdm
from tflop.evaluator import TEDS


def strip_html_contents(html_string):
    """ Strips the content in the tabular data (i.e. remove preceding and trailing whitespaces in the content) """

    # get pre_thead
    if "<thead>" in html_string:
        pre_thead, remain_str = html_string.split("<thead>", 1)
    else:
        pre_thead = "<html><body><table>"
        remain_str = html_string[len(pre_thead) :]

    # get thead
    if "</thead>" in remain_str:
        thead, remain_str = remain_str.split("</thead>", 1)
        if not remain_str.startswith("<tbody>"):
            remain_str = "<tbody>" + remain_str
    else:
        if "<tbody>" in remain_str:
            thead, tmp_remain = remain_str.split("<tbody>", 1)
            remain_str = "<tbody>" + tmp_remain
        elif "</tbody>" in remain_str:
            thead, remain_str = remain_str.split("</tbody>", 1)
            remain_str = "<tbody></tbody>" + remain_str
        else:
            thead = remain_str.split("</table></body></html>")[0]
            remain_str = "<tbody></tbody></table></body></html>"

    # get tbody
    remain_str = remain_str.split("<tbody>", 1)[1]
    if "</tbody>" in remain_str:
        tbody, post_tbody = remain_str.split("</tbody>", 1)
    else:
        if remain_str == "</table></body></html>":
            tbody = ""
            post_tbody = remain_str
        else:
            tbody = remain_str.split("</table></body></html>")[0]
            post_tbody = "</table></body></html>"

    thead_stripped, tbody_stripped = [], []
    # thead handling
    thead_rows = thead.split("</tr>")
    for row in thead_rows:
        if row == "":
            continue
        if "<tr>" in row:
            row_contents = row.split("<tr>")[1].split("</td>")
        else:
            row_contents = row.split("</td>")

        row_stripped = []
        for row_content in row_contents:
            if row_content == "":
                continue
            td_header, td_content = row_content.split(">", 1)
            if td_content.startswith("<b>") and td_content.endswith("</b>"):
                td_content = td_content[3:-4].strip()
                td_content = "<b>" + td_content + "</b>"
            else:
                td_content = td_content.strip()
            new_td_entry = td_header + ">" + td_content + "</td>"
            row_stripped.append(new_td_entry)
        thead_stripped.append("<tr>" + "".join(row_stripped) + "</tr>")

    # tbody handling
    tbody_rows = tbody.split("</tr>")
    for row in tbody_rows:
        if row == "":
            continue
        if "<tr>" in row:
            row_contents = row.split("<tr>")[1].split("</td>")
        else:
            row_contents = row.split("</td>")

        row_stripped = []
        for row_content in row_contents:
            if row_content == "":
                continue
            td_header, td_content = row_content.split(">", 1)
            if td_content.startswith("<b>") and td_content.endswith("</b>"):
                td_content = td_content[3:-4].strip()
                td_content = "<b>" + td_content + "</b>"
            else:
                td_content = td_content.strip()
            new_td_entry = td_header + ">" + td_content + "</td>"
            row_stripped.append(new_td_entry)
        tbody_stripped.append("<tr>" + "".join(row_stripped) + "</tr>")

    new_html = (
        pre_thead
        + "<thead>"
        + "".join(thead_stripped)
        + "</thead><tbody>"
        + "".join(tbody_stripped)
        + "</tbody>"
        + post_tbody
    )

    return new_html


def evaluate_distance(data_tuple):
    """ Evaluation of TEDs and S-TEDs scores"""
    ted_evaluator_structure_only = TEDS(structure_only=True, n_jobs=1)
    ted_evaluator = TEDS(structure_only=False, n_jobs=1, ignore_nodes=["b"])

    file_name, pred_string, gold_string = data_tuple

    # edit-distance
    edit_distance = distance(pred_string, gold_string) / max(
        len(pred_string), len(gold_string)
    )

    # TED
    refined_pred = pred_string
    refined_gold = gold_string
    if pred_string.startswith("<table>") and pred_string.endswith("</table>"):
        refined_pred = "<html><body>" + pred_string + "</body></html>"
    elif not pred_string.startswith("<html><body><table>") and not pred_string.endswith(
        "</table></body></html>"
    ):
        refined_pred = "<html><body><table>" + refined_pred + "</table></body></html>"

    if gold_string.startswith("<table>") and gold_string.endswith("</table>"):
        refined_gold = "<html><body>" + gold_string + "</body></html>"
    elif not gold_string.startswith("<html><body><table>") and not gold_string.endswith(
        "</table></body></html>"
    ):
        refined_gold = "<html><body><table>" + refined_gold + "</table></body></html>"

    # strip content in table data
    refined_pred = strip_html_contents(refined_pred)
    refined_gold = strip_html_contents(refined_gold)

    # tree-edit-distance (structure only)
    try:
        ted_score_structure_only = ted_evaluator_structure_only.evaluate(
            refined_pred, refined_gold
        )
    except:
        ted_score_structure_only = 0.0

    # tree-edit-distance (structure + content)
    try:
        ted_score = ted_evaluator.evaluate(
            refined_pred, refined_gold
        )
    except:
        ted_score = 0.0

    return (
        file_name,
        pred_string,
        gold_string,
        edit_distance,
        ted_score_structure_only,
        ted_score,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_inference_pathdir", type=str, required=True)
    parser.add_argument("--output_savepath", type=str, required=True)
    args = parser.parse_args()

    if "full_model_inference.json" in os.listdir(args.model_inference_pathdir):
        with open(
            os.path.join(args.model_inference_pathdir, "full_model_inference.json"), "r"
        ) as f:
            model_inference = json.load(f)
    else:
        split_files = [
            os.path.join(args.model_inference_pathdir, f)
            for f in os.listdir(args.model_inference_pathdir)
            if (f.startswith("full_model_inference") and f.endswith(".json"))
        ]
        model_inference = {}
        for split_file in split_files:
            with open(split_file, "r") as f:
                model_inference.update(json.load(f))

    data_collection = [
        (
            k,
            v["pred_string"],
            v["answer_string"]
        )
        for k, v in model_inference.items()
    ]
    # sort data_collection by k, where k is the file name
    data_collection = sorted(data_collection, key=lambda x: x[0])

    batch_size = 200
    num_processes = 8
    if len(data_collection) % batch_size == 0:
        num_batches = len(data_collection) // batch_size
    else:
        num_batches = (len(data_collection) // batch_size) + 1

    result_collection = []
    for batch_idx in tqdm(
        range(num_batches),
        desc="Evaluating...",
        position=1,
        leave=False,
    ):
        batch_data_collection = data_collection[
            batch_idx * batch_size : (batch_idx + 1) * batch_size
        ]
        pool = multiprocessing.Pool(processes=num_processes)
        outputs = pool.map(evaluate_distance, batch_data_collection)
        pool.close()
        pool.join()
        result_collection.extend(outputs)

    teds_score = sum([x[-1] for x in result_collection]) / len(result_collection)
    print(f"TEDs score: {teds_score}")
    
    # /ted_score_output.json
    with open(
        os.path.join(
            args.output_savepath,
            "ted_score_output.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result_collection, f, ensure_ascii=False)
