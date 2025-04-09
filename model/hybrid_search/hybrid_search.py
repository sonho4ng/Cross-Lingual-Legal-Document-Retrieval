
def hybrid_search(query, bm25_func, dense_func, corpus):
    bm25_res = bm25_func(query, corpus)
    dense_res = dense_func(query)

    # Convert BM25 từ DataFrame sang list[dict]
    bm25_list = bm25_res.to_dict(orient="records")

    # Hợp nhất + khử trùng lặp theo cid
    merged = {item["cid"]: item for item in bm25_list + dense_res}

    # đang ưu tiên bm25
    return list(merged.values())[:25]
