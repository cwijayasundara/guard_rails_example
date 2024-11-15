from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from guardrails import Guard
from guardrails.hub import CompetitorCheck
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
_ = load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

competitors_list = ["delta", "american airlines", "united"]

guard = Guard().use_many(
    CompetitorCheck(competitors=competitors_list, on_fail="fix")
)

prompt = ChatPromptTemplate.from_template("Answer this question {question}")

output_parser = StrOutputParser()

chain = prompt | model | guard.to_runnable() | output_parser

result = chain.invoke({"question": "What are the top five airlines for domestic travel in the US?"})
print(result)