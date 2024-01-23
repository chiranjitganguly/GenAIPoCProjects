from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import textract
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import openai
import os
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


# load the open AI key
load_dotenv()
# set up the openai key
openai.api_key = os.getenv('OPENAI_API_KEY')

client = openai.OpenAI()


def summarise_cr_stuffchain():
    # Define prompt
    prompt_template = """Write a concise summary for each of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    loader = UnstructuredWordDocumentLoader(
        "/AlliancePoC/CRSummarization/CR1.docx")
    docs1 = loader.load()
    loader = UnstructuredWordDocumentLoader(
        "/AlliancePoC/CRSummarization/CR-2.docx")
    docs2 = loader.load()
    docs = docs1 + docs2
    print(stuff_chain.run(docs))


def chat_with_cr():
    chat = ChatOpenAI()
    loader = UnstructuredWordDocumentLoader(
        "/AlliancePoC/CRSummarization/CR1.docx")
    docs1 = loader.load()
    loader = UnstructuredWordDocumentLoader(
        "/AlliancePoC/CRSummarization/CR-2.docx")
    docs2 = loader.load()
    docs = docs1 + docs2

    system_template = "You are an AI chatbot. Refer to {docs} to answer questions."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{input_message}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    request = (chat_prompt.format_prompt(docs=docs, input_message="What is the downtime for the affected systems?")
               .to_messages())

    result = chat(request)
    print(result.content)


if __name__ == "__main__":
    summarise_cr_stuffchain()
    #chat_with_cr()
