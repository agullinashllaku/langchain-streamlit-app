import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from extract_MD import url_dict
import openai
import os




# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = "/tmp/chroma"


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context but do not mention in the answer that it is mentioned in the context: {question}
"""


def main():
    st.image("image/1685967547827.jpeg", use_column_width=True)
    st.title("Databricks ML Associate Study Buddy With RAG")
    st.write(
        "Enter your question for any topic related to Databricks ML Associate certification:"
    )

    # User input
    query_text = st.text_area("Question", "")

    if st.button("Get Answer"):
        if query_text:
            # Prepare the DB
            embedding_function = OpenAIEmbeddings()
            db = Chroma(
                persist_directory=CHROMA_PATH, embedding_function=embedding_function
            )

            results = db.similarity_search_with_relevance_scores(query_text, k=4)
            if len(results) == 0:
                st.write("There are no docs available")
            elif results[0][1] < 0.6:
                st.write("Unable to find matching results.")
            else:
                context_text = "\n\n---\n\n".join(
                    [doc.page_content for doc, _score in results]
                )
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                prompt = prompt_template.format(
                    context=context_text, question=query_text
                )

                # Get response from the model
                model = ChatOpenAI()
                response_text = model.predict(prompt)
                sources = set()
                for doc, _score in results:
                    source_key = doc.metadata.get("source", None)

                if source_key in url_dict:
                    sources.add(url_dict[source_key])
                else:
                    sources = ["Test Bank"]

                sources = [source for source in sources if source is not None]
                sources_string = ",".join(sources)
                formatted_response = (
                    f"**Response:** {response_text}\n\n**Sources:** {sources_string}"
                )

                st.markdown(formatted_response)
        else:
            st.write("Please enter a question to get an answer.")

if __name__ == "__main__":
    main()
