from dataclasses import dataclass
from functools import partial
import numpy as np

import torch
from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class TensorFrame(Frame):
    tensor: torch.Tensor

    def __str__(self):
        return f"{self.name}(pts: {self.pts}, tensor: [{self.tensor}])"


class HuggingFaceTokenizer(FrameProcessor):
    """This processor calls the given hugging face tokenizer on text.

    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         print(frame.text)

    >>> aggregator = HuggingFaceTokenizer(TODO)
    >>> asyncio.run(print_frames(aggregator, TextFrame("Hello")))
    HELLO
    """

    def __init__(self, tokenizer):
        super().__init__()
        self._tokenizer = partial(tokenizer, return_tensors="pt")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            try:
                result = self._tokenizer(frame.text)["input_ids"]
            except Exception as e:
                print(e)
                # logger.error(f"Tokenizer error {frame.text}")
                await self.push_frame(frame, direction)
            else:
                print(f"Tokenized {frame.text} into {result}")
                await self.push_frame(TensorFrame(tensor=result), direction)
        else:
            await self.push_frame(frame, direction)


class HuggingFaceSAModel(FrameProcessor):
    """This processor calls the given hugging face sentiment analysis model on preprocessed text.

    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         print(frame.text)

    >>> aggregator = HuggingFaceSAProcessor(TODO)
    >>> asyncio.run(print_frames(aggregator, TextFrame("Hello")))
    HELLO
    """

    def __init__(self, model):
        super().__init__()
        self._model = model
        self._labels = self._model.config.id2label

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TensorFrame):
            result = self._model(frame.tensor).logits.detach().numpy()
            print(
                f"Forward pass result: {[self._labels[np.argmax(r)] for r in result]}"
            )
            await self.push_frame(TensorFrame(tensor=result), direction)
        else:
            await self.push_frame(frame, direction)
