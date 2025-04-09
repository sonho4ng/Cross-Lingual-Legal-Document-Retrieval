from model.hybrid_search import hybrid_search
from model.hybrid_search.bm25 import retrieve_bm25
from model.hybrid_search.bi_encoder.predict import dense_func
from model.rerank.predict import rerank_with_cross_encoder

if __name__ == '__main__':
    query = "Phó Tổng Giám đốc Ngân hàng Chính sách xã hội được xếp lương theo bảng lương như thế nào?"
    corpus = "/kaggle/working/corpus_hybrid_search.csv"
    model_name = "vinai/phobert-base"
    best_ckpt_path = trainer.state.best_model_checkpoint

    hybrid_search_results = hybrid_search(query, retrieve_bm25, dense_func, corpus )
    # Một example
    question = "Phó Tổng Giám đốc Ngân hàng Chính sách xã hội được xếp lương theo bảng lương như thế nào?"
    contexts = [item['text'] for item in hybrid_search_results]

    reranked = rerank_with_cross_encoder(
        question=question,
        contexts=contexts,
        model_path=best_ckpt_path,
        model_name=model_name,
        top_k=3
    )

    # In kết quả
    for i, (ctx, score) in enumerate(reranked, 1):
        print(f"#{i} - score: {score:.4f}")
        print(ctx)
        print("-" * 50)
