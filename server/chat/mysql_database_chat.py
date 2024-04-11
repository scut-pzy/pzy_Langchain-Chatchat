from fastapi import Body
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from configs import LLM_MODELS, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI, get_table_info

from typing import List, Optional, Union
from server.chat.utils import History
from langchain.prompts import PromptTemplate
from server.utils import get_prompt_template

from fastapi.responses import JSONResponse
from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
from server.db.repository import add_message_to_db
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler


# async def mysql_database_chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
#                 history: List[History] = Body([],
#                                        description="历史对话",
#                                        examples=[[
#                                            {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
#                                            {"role": "assistant", "content": "虎头虎脑"}]]
#                                        ),
#                 stream: bool = Body(False, description="流式输出"),
#                 model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
#                 temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
#                 # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
#                 prompt_name: str = Body("llm_chat", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
#          ):
#     history = [History.from_data(h) for h in history]
#
#     async def mysql_database_iterator(query: str,
#                             history: List[History] = [],
#                             model_name: str = LLM_MODEL,
#                             prompt_name: str = prompt_name,
#                             ) -> AsyncIterable[str]:
#         callback = AsyncIteratorCallbackHandler()
#         model = get_ChatOpenAI(
#             model_name=model_name,
#             temperature=temperature,
#             callbacks=[callback],
#         )
#
#         prompt_template = get_prompt_template(prompt_name)
#         input_msg = History(role="user", content=prompt_template).to_msg_template(False)
#         chat_prompt = ChatPromptTemplate.from_messages(
#             [i.to_msg_template() for i in history] + [input_msg])
#         ##todo 加入数据库
#
#         db = SQLDatabase.from_uri("mysql://root:123456@127.0.0.1:3306/sys")
#         llm_chain = LLMChain(prompt=chat_prompt, llm=model)
#         chain = SQLDatabaseChain(llm_chain=llm_chain,database=db)
#
#         # Begin a task that runs in the background.
#         task = asyncio.create_task(wrap_done(
#             chain.acall({"query": query}),
#             callback.done),
#         )
#
#         if stream:
#             async for token in callback.aiter():
#                 # Use server-sent-events to stream the response
#                 yield token
#         else:
#             answer = ""
#             async for token in callback.aiter():
#                 answer += token
#             yield answer
#
#         await task
#
#     a=StreamingResponse(mysql_database_iterator(query=query,
#                                            history=history,
#                                            model_name=model_name,
#                                            prompt_name=prompt_name),
#                              media_type="text/event-stream")
def mysql_database_chat(query: str = Body(..., description="用户输入", examples=[
    "查询发布年份在2021年之前，发布机构是含有中医的发布过，且指南领域与心血管有关，并给我其中的一条完整记录，需要包含表字段"]),
                        conversation_id: str = Body("", description="对话框ID"),
                        history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
                        history: Union[int, List[History]] = Body([],
                                                                  description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                                                  examples=[]
                                                                  ),
                        stream: bool = Body(False, description="流式输出"),
                        model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                        temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
                        max_tokens: Optional[int] = Body(None,
                                                         description="限制LLM生成Token数量，默认None代表模型最大值"),
                        prompt_name: str = Body("Guide_Model_Prompt",
                                                description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                        run_way: str = Body("direct", description="执行方式：stpe：步骤展开|direct:直接打出结果"),
                        table_name: List[str] = Body(["guide_modul"], description="模块名")
                        ):
    # 获取历史
    history = [History.from_data(h) for h in history]
    # 获取prompt
    PROMPT = PromptTemplate(
        input_variables=["input", "table_info", "dialect"], template=get_prompt_template('sql_chat', prompt_name)
    )
    # 获取表结构
    custom_table_info = get_table_info("guide_model_table")  ###todo 改成高可用性

    db = SQLDatabase.from_uri("mysql://root:123456@127.0.0.1:3306/data_display",
                              include_tables=table_name,
                              custom_table_info=custom_table_info
                              )
    model = get_ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    match run_way:
        case "step":
            db_chain = SQLDatabaseChain.from_llm(model, db, prompt=PROMPT, verbose=True,
                                                 return_intermediate_steps=True, use_query_checker=True)
            result = db_chain(query)
            response_data = {"intermediate_steps": result["interm"
                                                          "ediate_steps"]}
        case "direct":
            db_chain = SQLDatabaseChain.from_llm(model, db, prompt=PROMPT, verbose=True, return_intermediate_steps=False,
                                                 use_query_checker=True)
            response_data = db_chain.run(query)
        case _:
            response_data = {"ERROR MESSAGE": "ERROR KEY"}

    # 创建 JSONResponse 并返回
    return JSONResponse(content=response_data)
