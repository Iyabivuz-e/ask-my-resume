from ingestion_pipeline.embeddings import Embeddings, VectorStore
from retrieval_pipeline.retrieval import RetrievalPipeline, LLMRetrieval, RetrievalWithCitations
import streamlit as st
import os

def main():

    ## Page configs
    st.set_page_config(page_title="Ask My CV", page_icon=":robot_face:")
    st.title("Ask My CV")

    ##Then wwe load the resources cached(embeddings, vector store) so that they run only once
    @st.cache_resource
    def load_resources():
        vector_store = VectorStore()
        myembeddings = Embeddings()
        retrieval = RetrievalPipeline(vector_store, myembeddings)
        llm_retrieval = RetrievalWithCitations()

        return retrieval, llm_retrieval

    ## Then we initialise the chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    ## We display the chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    ## We then handle the user's input
    if prompt := st.chat_input("Ask a question about the resume(cv)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing the resume..."):
                retrieval, llm_retrieval = load_resources()
                retrieved_docs = retrieval.retrieve(prompt)[:3]
                response = llm_retrieval.generate_response(prompt, retrieved_docs)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.sidebar:
            st.header("Debug Info")
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

    

                
    
    # vector_store = VectorStore()
    # myembeddings = Embeddings()

    # retrieval = RetrievalPipeline(vector_store, myembeddings)
    # llm_retrieval = LLMRetrieval()
    # llm_retrieval = RetrievalWithCitations()

    ## Interaction loop
#     while True:
#         user_query = input("\nAsk (type 'q' to quit): ")
#         if user_query.lower() == "q" or user_query.lower() == "quit":
#             break
        
#         retrieved_docs = retrieval.retrieve(user_query)[:3]
#         response = llm_retrieval.generate_response(user_query, retrieved_docs)
#         print("\nAssistant:", response)


if __name__ == "__main__":
    main()


