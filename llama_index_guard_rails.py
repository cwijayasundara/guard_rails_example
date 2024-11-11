import warnings
from dotenv import load_dotenv
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.output_parsers.guardrails import GuardrailsOutputParser
from pydantic import BaseModel, Field
from typing import List
import guardrails as gd
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
)

warnings.filterwarnings('ignore')
_ = load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# load documents
documents = SimpleDirectoryReader("./docs/").load_data()

index = VectorStoreIndex.from_documents(documents, chunk_size=512)

class BulletPoints(BaseModel):
    # In all the fields below, you can define validators as well
    # Left out for brevity
    explanation: str = Field()
    explanation2: str = Field()
    explanation3: str = Field()


class Explanation(BaseModel):
    points: BulletPoints = Field(
        description="Bullet points regarding events in the author's life."
    )

# Define the prompt
prompt = """
Query string here.

${gr.xml_prefix_prompt}

${output_schema}

${gr.json_suffix_prompt_v2_wo_none}
"""

# Create a guard object
guard = gd.Guard.from_pydantic(output_class=Explanation, prompt=prompt)

# Create output parse object
output_parser = GuardrailsOutputParser(guard)

# attach to an llm object
llm = OpenAI(model='gpt-4o-mini', output_parser=output_parser)

# take a look at the new QA template!
fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
print(fmt_qa_tmpl)

query_engine = index.as_query_engine(
    llm=llm,
)
response = query_engine.query(
    "What are the three items the author did growing up?",
)

print(response)

# View a summary of what the guard did
guard.history.last.tree
