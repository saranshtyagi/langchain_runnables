from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about {topic}.", 
    input_variables=["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkedin_post": RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({"topic": "AI"})

# print(result)
print("Tweet:", result["tweet"])
print("LinkedIn Post:", result["linkedin_post"])