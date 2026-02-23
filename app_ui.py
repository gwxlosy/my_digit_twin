import streamlit as st
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import uuid
# ==========================================
# 1. 页面配置与全局初始化
# ==========================================
st.set_page_config(page_title="我的数字克隆人", page_icon="🤖", layout="centered")

# 初始化数据库连接 (连接我们在第二阶段建好的库！)
import os
# (确保你已经 import 了 RecursiveCharacterTextSplitter)

@st.cache_resource
def get_chroma_collection():
    # 连接数据库
    db_client = chromadb.PersistentClient(path="./my_clone_db_v2")
    # 这里要用 get_or_create，防止报错
    collection = db_client.get_or_create_collection("my_memory")
    
    # 🌟 核心修复逻辑：如果发现数据库是空的，就当场读取 txt 重新灌入基础记忆
    if collection.count() == 0:
        print("⏳ 首次在云端启动，正在重建基础记忆库...")
        
        # 你的基础语料文件名
        file_name = "wechat_memory.txt" 
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="utf-8") as f:
                full_text = f.read()
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["---", "\n\n", "\n"], 
                chunk_size=400,
                chunk_overlap=0 
            )
            chunks = text_splitter.split_text(full_text)
            ids = [f"base_memory_{i}" for i in range(len(chunks))]
            
            collection.add(documents=chunks, ids=ids)
            print(f"云端基础记忆注入成功，共 {len(chunks)} 条！")
        else:
            print("⚠️ 警告：找不到 wechat_memory.txt，克隆人将处于失忆状态！")
            
    return collection
memory_collection = get_chroma_collection()

# 初始化 DeepSeek 客户端
# ⚠️ 记得替换你的 API Key
# 让 Streamlit 从保险箱里读取 Key
client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
# ==========================================

with st.sidebar:
    st.header("🧠 记忆注入区 (仅主人可用)")
    
    # 用 expander 折叠起来，保持界面整洁
    with st.expander("➕ 添加新记忆"):
        admin_pwd = st.text_input("请输入主人密码：", type="password")
        new_memory = st.text_area("今天发生了什么值得记住的事？", placeholder="例如：今天中午去吃了顿爆辣火锅，肚子疼死了，以后再也不吃了！")
        
        if st.button("注入大脑", use_container_width=True):
            if admin_pwd == st.secrets["ADMIN_PASSWORD"]:
                if new_memory.strip():
                    with st.spinner("正在写入神经元..."):
                        # 1. 切分新记忆 (万一你写了一大段小作文)
                        text_splitter = RecursiveCharacterTextSplitter(
                            separators=["\n\n", "\n", "。", "！", "？"], 
                            chunk_size=400,
                            chunk_overlap=0 
                        )
                        new_chunks = text_splitter.split_text(new_memory)
                        
                        # 2. 生成随机的 ID (UUID保证绝不重复)
                        new_ids = [str(uuid.uuid4()) for _ in new_chunks]
                        
                        # 3. 存入 ChromaDB
                        memory_collection.add(documents=new_chunks, ids=new_ids)
                        
                        st.success(f"✅ 成功注入 {len(new_chunks)} 段新记忆！你的克隆人已经变聪明了。")
                else:
                    st.warning("总得写点什么吧？")
            else:
                st.error("🚫 密码错误！你是谁？")
    
    st.divider()
    st.caption("提示：在左侧注入新记忆后，直接在右侧提问测试。")
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