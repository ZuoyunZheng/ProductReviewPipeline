import asyncio
import json
import random
from pathlib import Path

import aiofiles
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    SystemFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from frame import SentimentFrame


class JSONReader(FrameProcessor):
    def __init__(self, path):
        super().__init__()
        self._path = path

    async def start(self):
        # Init asyncio Queue
        self._q = asyncio.Queue()
        # (Call run_coroutine(...) to) read file and put on Queue
        # asyncio.run_coroutine_threadsafe(self._read_file(), self.get_event_loop())
        self._read_task = asyncio.create_task(self._read_file())
        # create_task() to push/consume frame
        self._push_task = asyncio.create_task(self._push_handler())

    async def cancel(self):
        await self.cancel_task(self._read_task)
        self._read_task = None
        await self.cancel_task(self._push_task)
        self._push_task = None

    async def stop(self):
        await self.cancel_task(self._read_task)
        self._read_task = None
        await self.cancel_task(self._push_task)
        self._push_task = None

    async def _read_file(self):
        async with aiofiles.open(self._path, "r") as f:
            async for line in f:
                try:
                    line = json.loads(line)["text"]
                except KeyError as e:
                    logger.exception(f"KeyError {e} for {line}")
                    continue
                else:
                    logger.info(f"Read: {line[:30]}...")
                    await self._q.put(line)

    async def _push_handler(self):
        while True:
            # Uncomment to see some async behavior
            await asyncio.sleep(random.random() * 0.005)
            line = await self._q.get()
            logger.info(f"Pushed: {line[:30]}...")
            await self.push_frame(TextFrame(text=line))
            self._q.task_done()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Specific system frames
        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self.start()
        elif isinstance(frame, CancelFrame):
            await self.cancel()
            await self.push_frame(frame, direction)
        # All other system frames
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames
        elif isinstance(frame, EndFrame):
            # Push EndFrame before stop(), because stop() waits on the task to
            # finish and the task finishes when EndFrame is processed.
            await self.push_frame(frame, direction)
            await self.stop()
        # Other frames
        else:
            await self.push_frame(frame, direction)


class JSONWriter(FrameProcessor):
    def __init__(self, path):
        super().__init__()
        self._path = path
        Path(self._path).unlink(missing_ok=True)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SentimentFrame):
            line = {"text": frame.text, "sentiment": frame.stmt}
            logger.info(f"Outputted: {line}")
            with open(self._path, "a+") as f:
                f.write(json.dumps(line) + "\n")
        else:
            await self.push_frame(frame, direction)
