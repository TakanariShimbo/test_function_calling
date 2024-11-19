import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


def initialize_openai_client() -> OpenAI:
    """OpenAIクライアントの初期化"""
    load_dotenv()
    api_key = os.environ["OPENAI_API_KEY"]
    return OpenAI(api_key=api_key)


def add_numbers(num1: float, num2: float) -> float:
    """2つの数を足し算して結果を返す"""
    return num1 + num2


def get_tools_definition() -> list[dict[str, Any]]:
    """関数定義を返す"""
    return [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "2つの数を足します。ユーザーが2つの数の合計を求めたときにこの関数を使用してください。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num1": {"type": "number", "description": "最初に足す数。"},
                        "num2": {"type": "number", "description": "次に足す数。"},
                    },
                    "required": ["num1", "num2"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def get_system_message() -> dict[str, Any]:
    """初期メッセージを返す"""
    return {
        "role": "system",
        "content": "あなたは2つの数の足し算ができる役立つアシスタントです。提供されたツールを使ってユーザーをサポートしてください。",
    }


def first_query(client: OpenAI, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    """ツール呼び出しを含むメッセージを返す"""
    response = client.chat.completions.create(model=model, messages=messages, tools=tools)
    response_message = response.choices[0].message

    # ツール呼び出しがない場合
    if not hasattr(response_message, "tool_calls"):
        return (False, [response_message])
    if not response_message.tool_calls:
        return (False, [response_message])

    # ツール呼び出しがある場合
    tool_call = response_message.tool_calls[0]
    arguments = json.loads(s=tool_call.function.arguments)
    num1 = arguments.get("num1")
    num2 = arguments.get("num2")
    result = add_numbers(num1=num1, num2=num2)

    function_call_result_message = {
        "role": "tool",
        "content": json.dumps({"num1": num1, "num2": num2, "result": result}),
        "tool_call_id": tool_call.id,
    }

    return (True, [response_message, function_call_result_message])


def second_query(client: OpenAI, model: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """計算結果を含む最終応答を生成し、メッセージを返す"""
    response = client.chat.completions.create(model=model, messages=messages)
    response_message = response.choices[0].message
    return [response_message]


def main() -> None:
    """メイン処理"""
    client = initialize_openai_client()
    tools = get_tools_definition()
    system_message = get_system_message()

    first_query_messages = [
        system_message,
        # {"role": "user", "content": "こんにちは"},
        {"role": "user", "content": "123456789+987654321は？"},
    ]

    is_tool_called, first_response_message = first_query(
        client=client,
        model="gpt-4o",
        messages=first_query_messages,
        tools=tools,
    )

    if not is_tool_called:
        print("ツール呼び出しなし :", first_response_message[0].content)
        return

    final_message = second_query(
        client=client,
        model="gpt-4o",
        messages=first_query_messages + first_response_message,
    )

    print("ツール呼び出しあり :", final_message[0].content)
    return


if __name__ == "__main__":
    main()
