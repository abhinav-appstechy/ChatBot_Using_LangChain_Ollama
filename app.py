
import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
import utils
import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.title("Chatbot Demo With Ollama Gemma LLM")

st.image("https://cdn.analyticsvidhya.com/wp-content/uploads/2024/02/gemma-cover.png", "gemma llm")
class BasicChatbot:

    def __init__(self):

        pass
    
    def setup_chain(self):
        # Create a memory object which will store the conversation history.
        
        llm = Ollama(model="gemma:2b")
        memory = ConversationSummaryMemory(llm=llm)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Please respond to the user's questions."),
            ("user", "Question: {input}")
        ])
        output_parser = StrOutputParser()
        chain = ConversationChain(
            llm=llm,
            verbose=True,
            memory=memory,
            output_parser=output_parser,
            
        )
        return chain
    
    @utils.enable_chat_history
    def main(self):
        if 'chain' not in st.session_state:
            st.session_state.chain = self.setup_chain()
            st.session_state.messages = []

        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            # result = st.session_state.chain.invoke({"input": user_query})
            result = st.session_state.chain.predict(input= user_query)
            print(result)
            response = result
            utils.display_msg(response, 'assistant')

if __name__ == "__main__":
    obj = BasicChatbot()
    obj.main()
