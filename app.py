import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="Math Problem Solver", page_icon="🧮")

st.title("Math Problem Solver")

groq_api_key = st.sidebar.text_input("GROQ API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your GROQ API key.")
    st.stop()

llm = ChatGroq(model = "Gemma2-9b-It", groq_api_key = groq_api_key)

wikipedia_wrapper = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Search Wikipedia for information."
)

math_chain = LLMMathChain.from_llm(llm)

calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Perform mathematical calculations."
)

prompt = """
You are a helpful assistant. Your task is to solve user's math problems.
Logically arrive at the solution and provide a detailed explanation and
display it point wise for the question below: 
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(input_variables=["question"], template=prompt)

chain = LLMChain(llm = llm, prompt = prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="Solve math problems."
)

assistant_agent = initialize_agent(
    tools = [wikipedia_tool, calculator, reasoning_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handle_parse_errors = True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role" : "assistant", "content" : "Hi, I'm your math problem solving assistant. How can I help you today?"}
    ]


for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

def generate_response(question):
    response = assistant_agent.invoke({"question" : question})

question = st.text_area("Enter your question")

if st.button("Submit"):
    if question:
        with st.spinner("Generating..."):
            st.session_state.messages.append({"role" : "user", "content" : question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({"role" : "assistant", "content" : response})
            st.chat_message("assistant").write(response)

    else:
        st.warning("Please enter a question.")