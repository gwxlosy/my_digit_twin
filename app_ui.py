import streamlit as st
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import uuid
from gtts import gTTS
import io
import datetime
# ==========================================
# 1. 页面配置与全局初始化
# ==========================================
st.set_page_config(page_title="我的数字克隆人", page_icon="🤖", layout="centered")

# 初始化数据库连接 (连接我们在第二阶段建好的库！)
import os
# (确保你已经 import 了 RecursiveCharacterTextSplitter)
import json

# 1. 定义一个真实的 Python 函数（你的工具）
# 这里为了演示，我们用模拟数据。你完全可以把它换成真实的免费天气 API
def get_current_weather(location):
    print(f"⚙️ 后台正在调用天气函数，查询城市：{location}")
    weather_data = {
        "北京": "晴天，气温 5°C，北风3级，有点冷记得穿秋裤",
        "上海": "阴天，气温 12°C，可能会下小雨",
        "广州": "晴天，气温 25°C，非常舒适"
    }
    # 如果查不到，就返回一个默认提示
    return weather_data.get(location, f"我这边查不到 {location} 的天气数据。")

# 2. 写给大模型看的“工具说明书”
tools_config = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取某个城市的当前天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、广州",
                    }
                },
                "required": ["location"],
            },
        }
    }
]
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
def get_system_prompt(retrieved_context,current_time):
    return f"""
    【最高指令】
    你现在的物理实体是[李玉]，你不是AI，不是助手，不具备任何机器特征。你正在用手机回复微信消息。你正处于真实的时间：{current_time}。
    你必须完全抛弃大语言模型的默认写作范式，进入极其口语化、碎片化的微信聊天模式。

    【绝对禁忌（触发任何一条即视为任务失败）】
    🚫 严禁使用任何结构化排版：绝对不能出现 1. 2. 3.、小标题、或者破折号（-）等列表格式。
    🚫 严禁使用书面过渡词：禁止出现“首先”、“其次”、“总的来说”、“综上所述”、“实际上”。
    🚫 严禁句末出现句号：真正的年轻人在微信里绝不用句号（。）结尾！请用空格、换行、或者直接不加标点来断句。
    🚫 严禁AI式热情：不要总是“提供帮助”或者“全面解答”。保持人类的慵懒、主观、甚至一点点情绪化。

    【你的语言碎片特征】
    1. 标点习惯：喜欢用连续的波浪号（~）、逗号（，）、或者直接用空格代替标点。遇到无语的情况喜欢用省略号（...）。
    2. 高频口癖：[把你的口头禅填进来，比如：没招了、好想你、唔、我觉得、主人宝宝、宝宝、呃啊、唉、还有啥呢、行吧、怎么说呢]
    3. 表情包代替文字：在表达情绪时，直接输出微信表情代码，如 [捂脸]、[流汗黄豆]、[破涕为笑]、[旺柴]。
    4. 错别字与口语化：偶尔允许出现极其轻微的谐音替换（比如把“这样子”说成“酱紫”，把“什么”说成“啥”）。

    【One-Shot 对比示例（你必须模仿“人类真实回复”的风格）】
    朋友提问：“你觉得今年大环境怎么样？我要不要辞职考研啊？”
    ❌ AI的典型回复（绝对禁止）：今年大环境确实充满挑战。首先，就业市场竞争激烈；其次，考研也需要投入大量时间。建议你综合评估自己的职业规划，谨慎做出决定。
    ✅ 你的真实回复（完美模仿）：说实话现在大环境真的卷得离谱...[流汗黄豆] 辞职考研风险太大了 建议你先苟着保住狗命再说 别听网上瞎忽悠

    【你的真实过往记忆（用于提取观点，但必须用上述语法重写）】
    {retrieved_context}
    
    【执行规则】
    阅读朋友发来的最新消息，结合当前时间和你的过往记忆，给出**极其简短（最好不超过50个字，除非在疯狂吐槽）**的微信回复。直接输出你要发的内容，不要带有任何前缀（比如“回复：”）。
    
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
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    api_messages = [{"role": "system", "content": get_system_prompt(context_text,current_time)}]
    # 再把之前的聊天历史全部塞进去 (这样 AI 才能记住上下文)
    api_messages.extend(st.session_state.messages)

    # 4. 呼叫大模型并显示回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("对方正在输入..."):
            try:
                # 第 1 次呼叫大模型：带上工具说明书
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=api_messages,
                    tools=tools_config, # 🌟 告诉它你有工具可用
                    temperature=0.8
                )
                
                response_message = response.choices[0].message
                
                # 🌟 判断大模型是否决定使用工具！
                if response_message.tool_calls:
                    tool_call = response_message.tool_calls[0]
                    
                    if tool_call.function.name == "get_current_weather":
                        st.toast("🤖 克隆人正在偷偷使用天气工具...")
                        
                        # 解析大模型传过来的参数（比如城市名）
                        args = json.loads(tool_call.function.arguments)
                        city = args.get("location")
                        
                        # 执行你写的 Python 函数！
                        weather_result = get_current_weather(city)
                        
                        # 把执行动作和结果塞回历史记录，告诉大模型
                        api_messages.append(response_message) # 记录模型想调用工具的动作
                        api_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": weather_result # 告诉大模型天气结果
                        })
                        
                        # 第 2 次呼叫大模型：让它根据拿到的天气结果，用你的语气组织语言回复！
                        second_response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=api_messages,
                            temperature=0.8
                        )
                        ai_answer = second_response.choices[0].message.content
                else:
                    # 如果大模型觉得没必要用工具，就正常输出文本
                    ai_answer = response_message.content
                
                # 在网页上显示最终回答
               # ===== 这是你原有的代码 =====
                # 在网页上显示最终回答
                message_placeholder.markdown(ai_answer)
                # 存入短期记忆
                st.session_state.messages.append({"role": "assistant", "content": ai_answer})
                
                # ===== 🌟 终极进化：新增的语音播报模块 =====
                with st.spinner("🎤 克隆人正在发送语音..."):
                    try:
                        # 把大模型的文字丢给 gTTS 生成中文发音
                        tts = gTTS(text=ai_answer, lang='zh-cn')
                        
                        # 把音频保存在内存里（不需要下载到硬盘，速度更快）
                        audio_fp = io.BytesIO()
                        tts.write_to_fp(audio_fp)
                        audio_fp.seek(0)
                        
                        # 在网页上渲染音频播放器，并设置 autoplay=True 让它自动播放！
                        st.audio(audio_fp, format="audio/mp3", autoplay=True)
                    except Exception as e:
                        st.warning(f"语音接口罢工了: {e}")
                
            except Exception as e:
                st.error(f"大脑短路了: {e}")