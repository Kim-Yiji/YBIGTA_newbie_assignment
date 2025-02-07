# 구현하세요!
import datasets

def load_corpus() -> list[str]:
    """
    Hugging Face Datasets 라이브러리를 사용하여 corpus 로드
    """
    corpus: list[str] = []
    # 구현하세요!
    dataset = datasets.load_dataset("google-research-datasets/poem_sentiment", split="train")
    corpus = [example["verse_text"] for example in dataset]
    return corpus