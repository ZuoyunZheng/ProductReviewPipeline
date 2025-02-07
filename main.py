import asyncio

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hf import HuggingFaceSAModel, HuggingFaceTokenizer
from js import JSONReader, JSONWriter


async def main():
    json_reader = JSONReader("assets/All_Beauty.jsonl")

    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )
    hf_tokenizer = HuggingFaceTokenizer(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )
    hf_model = HuggingFaceSAModel(model)

    json_writer = JSONWriter("sentiment.jsonl")
    runner = PipelineRunner()
    task = PipelineTask(  # noqa
        Pipeline(
            [
                json_reader,
                hf_tokenizer,
                hf_model,
                json_writer,
            ]
        )
    )
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
