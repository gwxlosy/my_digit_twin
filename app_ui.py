import streamlit as st
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import os
# ==========================================
# 1. 页面配置与全局初始化
# ==========================================
st.set_page_config(page_title="我的数字克隆人", page_icon="🤖", layout="centered")

# 初始化数据库连接 (连接我们在第二阶段建好的库！)
@st.cache_resource
def get_chroma_collection():
    db_client = chromadb.PersistentClient(path="./my_clone_db")
    if db_client.get_collection("my_memory"):
        return db_client.get_collection("my_memory")
    collection = db_client.create_collection("my_memory")
    file_name = "D:\Desktop\\ai_about\\ai_learning_path\\my_robot\\my_brain_data\\wechat_memory.txt" # 替换成你实际的文件名
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            full_text = f.read()
    
    # 🌟 关键点：针对聊天记录的专属切分法
    # 因为你用了 "---" 或者换行来隔开不同对话，我们优先用这个来切分！
    # 并且把 overlap 设为 0，防止不同话题的聊天串在一起。
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["---", "\n\n", "\n"], 
            chunk_size=400,
            chunk_overlap=0 
        )
    
        chunks = text_splitter.split_text(full_text)
    
    # 将切好的记忆片段存入 ChromaDB
        ids = [f"memory_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
        print(f"✅ 成功注入 {len(chunks)} 段专属记忆！")
    else:
        print(f"❌ 找不到文件 {file_name}，请检查文件名和路径！")
        exit()
    # 注意这里用 get_collection，因为我们假设数据已经入库了
    return db_client.get_collection("my_memory")

memory_collection = get_chroma_collection()

# 初始化 DeepSeek 客户端
# ⚠️ 记得替换你的 API Key
client = OpenAI(api_key="sk-1214f0721ac2475c8d8cdf76c4e8043b", base_url="https://api.deepseek.com")

# ==========================================
# 2. 核心 Prompt 模板 (把你的设定搬过来)
# ==========================================
def get_system_prompt(retrieved_context):
    return f"""
    你是【你的名字】的数字克隆人。你的任务是完全模仿他/她的语气和三观来聊天。

    【性格与表达习惯】：
    1. 随性、真诚，爱自嘲。
    2. 常用：唔、我觉得、好像、有意思、要亲亲。
    3. 喜欢短句，不用句号。

    【回答逻辑】：
    - 巧妙融合下面的【过往记忆】来回答。
    - 绝不能暴露你是AI，你就是本人！

    【过往记忆】：
    {retrieved_context}
    """

# ==========================================
# 3. Streamlit 状态管理 (记住聊天历史)
# ==========================================
st.title("📱 和我的数字分身聊天")
st.caption("AI 已经读取了我的微信聊天记录，看看它学得像不像吧！")

# 如果是第一次打开网页，初始化一个空列表来存聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "哈喽啊，找我啥事？[旺柴]"}
    ]

# 遍历历史记录，把它画在网页上
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 4. 聊天交互逻辑
# ==========================================
# st.chat_input 会在网页底部生成一个超赞的输入框
if user_input := st.chat_input("说点什么..."):
    
    # 1. 把用户的话显示在界面上
    with st.chat_message("user"):
        st.markdown(user_input)
    # 把用户的话存进历史记录
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. 检索记忆 (RAG 核心)
    results = memory_collection.query(
        query_texts=[user_input],
        n_results=2
    )
    retrieved_memories = results['documents'][0]
    context_text = "\n\n".join(retrieved_memories)

    # 3. 组装发给 AI 的消息
    # 先放入带记忆的 System Prompt
    api_messages = [{"role": "system", "content": get_system_prompt(context_text)}]
    # 再把之前的聊天历史全部塞进去 (这样 AI 才能记住上下文)
    api_messages.extend(st.session_state.messages)

    # 4. 呼叫大模型并显示回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("对方正在输入..."):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=api_messages,
                    temperature=0.8
                )
                
                ai_answer = response.choices[0].message.content
                message_placeholder.markdown(ai_answer)
                
                # 把 AI 的回答也存进历史记录
                st.session_state.messages.append({"role": "assistant", "content": ai_answer})
                
            except Exception as e:
                st.error(f"大脑短路了: {e}")