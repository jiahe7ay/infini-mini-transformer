import os
from unicodedata import normalize

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import ujson
from rich import progress


def split_txt_cropus_to_chunk_data(
    texts: list, batch_size: int = 768**2, max_len: int = 768, window_size: int = 2
) -> list:

    buffer, buffer_len = [], 0
    chunk_data = []

    for i, line in enumerate(texts):
        print("chunk_data")
        print(i)
        buffer_len += len(line)
        buffer.append(line)

        if buffer_len >= batch_size or i == len(texts) - 1:
            buffer_txt = "".join(buffer)

            # - window_size为滑动窗口，这样每个窗口都包含有window_size个上文
            for i in range(0, len(buffer_txt), max_len - window_size):

                chunk_data.append("".join(buffer_txt[i : i + max_len]))

            buffer, buffer_len = [], 0

    return chunk_data




def process_none(s: str) -> str:
    if s:
        return s
    return ""




def gen_wiki_filter(origin_file, output_file="../train_datasets/wiki_fi.parquet"):
    lines = []
    with open(origin_file, "r", encoding="utf-8") as f:
        items = ujson.load(f)
       # for item in items:
        lines.append(items["completion"] + "<|endoftext|>")
    chunk_data = split_txt_cropus_to_chunk_data(lines)
    tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
    pq.write_table(
        table=tb,
        where=output_file,
        row_group_size=50000,
        data_page_size=50000,
    )
gen_wiki_filter("../datasets/wikipedia-cn-20230720-filtered.json")


