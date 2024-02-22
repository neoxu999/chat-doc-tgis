import os

import streamlit as st
import utils.caikit_tgis_langchain as caikit_tgis_langchain

from gui.history import ChatHistory
from gui.layout import Layout
from gui.sidebar import Sidebar, Utilities
from snowflake import SnowflakeGenerator

username = os.environ.get("REDIS_USERNAME", "default")
password = os.environ.get("REDIS_PASSWORD", "default")
host = os.environ.get("REDIS_HOST", "127.0.0.1")
redis_url = f"redis://{username}:{password}@{host}:6379"
certificate_chain_file = os.environ.get("CERTIFICATE_CHAIN_FILE")

inference_server_url=os.environ.get('INFERENCE_SERVER_URL',
  'https://llama-2-7b-chat-hf-fine-tuned-predictor-rhone-chatbot-demo.apps.rosa-zpp6h.zjoc.p1.openshiftapps.com/')
model_id = os.environ.get("MODEL_ID", "llama-2-7b-chat-hf-fine-tuned")

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_icon="💬", page_title="ChatPDF")
    layout, sidebar, utils = Layout(), Sidebar(), Utilities()

    layout.show_header()
    sidebar.show_logo()
    login_config = utils.load_login_details()

    if not login_config:
        layout.show_loging_details_missing()
    else:
        sidebar.show_login(login_config)
        pdf = utils.handle_upload()

        if pdf:
            sidebar.show_options()

            try:
                if 'chatbot' not in st.session_state:
                    llm = caikit_tgis_langchain.CaikitLLM(
                        inference_server_url=inference_server_url,
                        model_id=model_id,
                        certificate_chain=certificate_chain_file,
                        streaming=False
                    )

                    indexGenerator = SnowflakeGenerator(42)
                    index_name = str(next(indexGenerator))

                    chatbot = utils.setup_chatbot(pdf, llm, redis_url, index_name, "redis_schema.yaml")
                    st.session_state["chatbot"] = chatbot

                if st.session_state["ready"]:
                    history = ChatHistory()
                    history.initialize(pdf.name)

                    response_container, prompt_container = st.container(), st.container()

                    with prompt_container:
                        is_ready, user_input = layout.prompt_form()

                        if st.session_state["reset_chat"]:
                            history.reset()

                        if is_ready:
                            with st.spinner("Processing query..."):
                                output = st.session_state["chatbot"].conversational_chat(user_input)

                    history.generate_messages(response_container)

            except Exception as e:
                st.error(f"{e}")
                st.stop()

    sidebar.about()
