from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    StartFrame,
    EndFrame,
    SystemFrame,
    CancelFrame,
)
import json
import aiofiles
import asyncio
import random


class ReviewScrapper(FrameProcessor):
    def __init__(self, url):
        super().__init__()
        self._url = url


class JSONReader(FrameProcessor):
    def __init__(self, path):
        super().__init__()
        self._path = path

    async def start(self):
        # Init asyncio Queue
        print("Init queue")
        self._q = asyncio.Queue()
        # Call run_coroutine(...) to read file and put on Queue
        print("Launch read")
        asyncio.run_coroutine_threadsafe(self._read_file(), self.get_event_loop())
        # create_task() to push/consume frame
        print("Created push data task")
        self._data_task = asyncio.create_task(self._data_handler())

    async def cancel(self):
        await self.cancel_task(self._data_task)
        self._data_task = None

    async def stop(self):
        await self.cancel_task(self._data_task)
        self._data_task = None

    async def _read_file(self):
        async with aiofiles.open(self._path, "r") as f:
            async for line in f:
                await asyncio.sleep(random.random())
                try:
                    line = json.loads(line)["text"]
                except KeyError as e:
                    print(f"KeyError {e} for {line}")
                    continue
                else:
                    print(f"Enqueued: {line}")
                    await self._q.put(line)

    async def _data_handler(self):
        while True:
            line = await self._q.get()
            print(f"Dequeued {line}")
            await self.push_frame(TextFrame(text=line))
            self._q.task_done()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Specific system frames
        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            print("StartFrame")
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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
