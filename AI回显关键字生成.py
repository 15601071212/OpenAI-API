# import streamlit as st
# import os
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'LRM.settings')

# import django
# django.setup()
# from srm.views import if_show_command
# import json

# # Add a selectbox to the sidebar:
# add_selectbox = st.sidebar.selectbox(
#     '选择模型名称',
#     ('Qwen-72B-Chat','Qwen_14B_Chat')
# )
# # Add a slider to the sidebar:
# add_slider = st.sidebar.slider(
#     '选择随机性指数',
#     0.0, 1.0, 0.5
# )

# model=add_selectbox
# temperature=add_slider

# keyword_name='获取Interface信息'
# st.text_input("请输入你的问题：", key="name")

# messages=st.session_state.name
# if messages:
#     answer=if_show_command(messages,keyword_name,model,temperature)
#     st.write('AI回复的答案：')
#     st.code(answer, language='robotframework')
import streamlit as st
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'LRM.settings')

import django
django.setup()
from srm.views import (
                        if_show_command,
                        split_show_command,
                        chat_knowledge_for_show_command,
                        get_keywords_multithread
                         )
import json

st.set_page_config(
        page_title="AI show命令行关键字生成",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "### design by 刘瑞10340416 2024.01.11"
        }
    )

st.sidebar.title('AI show命令行关键字生成')
# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    '选择模型名称',
    ('Qwen-72B-Chat','Qwen_14B_Chat')
)
# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    '选择随机性指数',
    0.0, 1.0, 0.2
)

model = add_selectbox
temperature = add_slider

#keyword_name = '获取Interface信息'

# Get conversation history from session state
conversation_history = st.session_state.get('conversation_history', [])

# Display the conversation history
#counter=1
for conv in conversation_history:
    with st.chat_message("human"):
        st.text(conv["question"])
    #st.text(f'你的问题{counter}：{conv["question"]}\n')
    with st.chat_message("ai"):
        st.code(conv["answer"], language='python')
    #st.code(conv["answer"], language='robotframework',line_numbers=True)
    #counter=counter+1

# Create a text input at the bottom of the page
# Do not use the previous value, just set it to an empty string
#question = st.text_input("请输入你的问题，例如：如何通过show arp命令获取ARP信息中的The count数值？：", value='', key="name"

if len(conversation_history)==0:
    st.text("问答示例：")
    question_example="如何通过show arp命令获取ARP信息中的The count数值?"
    answer_example="${result}    获取ARP信息    device=${zxr0}    get_info=arp_count"
    with st.chat_message("human"):
            st.text(question_example)
    with st.chat_message("ai"):
            st.code(answer_example, language='python')
question = st.chat_input("请输入你的问题，例如：如何通过show arp命令获取ARP信息中的The count数值？ 换行请使用Shift+Enter")

# keyword_dict={
#     "show arp": "获取ARP信息",
#     "show interface": "获取Interface信息"
#     }

# if st.button('提交问题'):
if question:
    if question.startswith('!<'):
        keyword_list,config_dict = get_keywords_multithread(question, model=model, temperature=temperature)
        config_dict_convert = {k: [f"加{v}" if v.startswith("配置") else v for v in value] for k, value in config_dict.items()}
        config_keyword_dict = {}
        for k,v in config_dict_convert.items():
            config_keyword_dict[k]={'关键字':v[1]}
            
            
        st.json(config_keyword_dict)
    else:
        #answer1,answer2 = split_show_command(question, model, temperature)
        #show_keyword_list, show_command_list, execution_time = chat_knowledge_for_show_command(answer1.lower(),'h11-23-show', top_k=1)
        show_keyword_list, show_command_list, execution_time = chat_knowledge_for_show_command(question.lower(),'h11-23-show', top_k=1)
        answer = if_show_command(question, show_keyword_list[0], model, temperature)
        #answer = if_show_command(answer2, show_keyword_list[0], model, temperature)
        # Append the new conversation to the history
        conversation_history.append({'question': question, 'answer': answer})
        #conversation_history.append({'question': question, 'answer': '%s:%s' % (answer1,answer2)})
        # Update the session state with the new conversation history
        st.session_state.conversation_history = conversation_history
        # Reset the temporary variable for the input value
        st.session_state.input_value = ''
        # Rerun the app to update the conversation display
        st.rerun()
    
