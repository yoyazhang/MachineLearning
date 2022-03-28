from typing import List
import torch


class Solution:
    def crf_predict(self, sentences: List[str]) -> List[str]:
        model = torch.load("models/modelCRF-1.pth")
        results = model.conduct(sentences)
        return results

    def dnn_predict(self, sentences: List[str]) -> List[str]:
        model = torch.load("models/modelDNN-1.pth")
        results = model.conduct(sentences)
        return results
