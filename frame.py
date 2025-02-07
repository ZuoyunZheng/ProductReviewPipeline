from dataclasses import dataclass

import torch
from pipecat.frames.frames import Frame


@dataclass
class TokenFrame(Frame):
    text: str
    tensor: torch.Tensor

    def __str__(self):
        return f"{self.name}(pts: {self.pts}, txt: {self.text}, tensor: {self.tensor})"


@dataclass
class SentimentFrame(Frame):
    text: str
    stmt: str

    def __str__(self):
        return f"{self.name}(pts: {self.pts}, txt: {self.text}, sentiment:{self.stmt})"
