import asyncio
import random
from functools import partial

import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from frame import SentimentFrame, TokenFrame


class HuggingFaceTokenizer(FrameProcessor):
    """This processor calls the given hugging face tokenizer on text."""

    def __init__(self, tokenizer):
        super().__init__()
        self._tokenizer = partial(tokenizer, return_tensors="pt")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            try:
                # Uncomment to see some async behavior
                await asyncio.sleep(random.random() * 0.005)
                result = self._tokenizer(frame.text)["input_ids"]
                if random.random() > 0.9:
                    raise Exception
            except Exception as e:
                logger.error(f"Tokenizer error for {frame.text[:30]}...\n{e}")
                # await self.push_frame(frame, direction)
            else:
                logger.info(f"Tokenized: {frame.text[:30]}...")
                await self.push_frame(
                    TokenFrame(text=frame.text, tensor=result), direction
                )
        else:
            await self.push_frame(frame, direction)


class HuggingFaceSAModel(FrameProcessor):
    """This processor calls the given hugging face sentiment analysis model on preprocessed text."""

    def __init__(self, model):
        super().__init__()
        self._model = model
        self._labels = self._model.config.id2label

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TokenFrame):
            # Uncomment to see some async behavior
            await asyncio.sleep(random.random() * 0.005)
            logits = self._model(frame.tensor).logits.detach().numpy()
            stmts = [self._labels[np.argmax(lo)] for lo in logits]
            logger.info(f"Forward pass: {stmts}")
            await self.push_frame(
                SentimentFrame(text=frame.text, stmt=stmts), direction
            )
        else:
            await self.push_frame(frame, direction)
