import json
import os
from typing import Any, List, Dict

from dotenv import load_dotenv
from openai import OpenAI


# === 環境設定とクライアントの初期化 ===
def initialize_openai_client() -> OpenAI:
    """OpenAIクライアントを初期化します。"""
    load_dotenv()
    api_key = os.environ["OPENAI_API_KEY"]
    return OpenAI(api_key=api_key)


# === 足し算関数 ===
def add_numbers(num1: float, num2: float) -> float:
    """2つの数を足し算して結果を返します。"""
    return num1 + num2


# === ツール定義 ===
def get_tools_definition() -> List[Dict[str, Any]]:
    """関数定義を返します。"""
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


# === 初期メッセージ定義 ===
def get_initial_messages() -> List[Dict[str, str]]:
    """初期メッセージを返します。"""
    return [
        {"role": "system", "content": "あなたは2つの数の足し算ができる役立つアシスタントです。提供されたツールを使ってユーザーをサポートしてください。"},
        {"role": "user", "content": "7と13を足してもらえますか？"},
    ]


# === OpenAI API 呼び出し ===
def call_chat_completion(client: OpenAI, model: str, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Any:
    """OpenAIのChat Completion APIを呼び出し、メッセージを返します。"""
    response = client.chat.completions.create(model=model, messages=messages, tools=tools)
    return response.choices[0].message


def call_final_completion(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> Any:
    """計算結果を含む最終応答を生成し、メッセージを返します。"""
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message


# === レスポンスの処理 ===
def process_tool_calls(response_message: Any, client: OpenAI, messages: List[Dict[str, str]]) -> str:
    """ツール呼び出しを処理し、計算結果を生成します。"""
    tool_call = response_message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    num1 = arguments.get("num1")
    num2 = arguments.get("num2")

    # 足し算を実行
    result = add_numbers(num1, num2)

    # 計算結果を送信
    function_call_result_message = {
        "role": "tool",
        "content": json.dumps({"num1": num1, "num2": num2, "result": result}),
        "tool_call_id": tool_call.id,
    }

    # 最終応答を生成
    final_response = call_final_completion(client, model="gpt-4o", messages=messages + [response_message, function_call_result_message])
    return final_response.content


def process_response(response_message: Any, client: OpenAI, messages: List[Dict[str, str]]) -> str:
    """レスポンスメッセージを処理し、最終的な出力を生成します。"""
    if hasattr(response_message, "tool_calls") and response_message.tool_calls:
        return process_tool_calls(response_message, client, messages)
    return response_message.content


# === メイン処理 ===
def main() -> None:
    """プログラムのエントリーポイント。"""
    client = initialize_openai_client()
    tools = get_tools_definition()
    messages = get_initial_messages()

    # 初回のAPI呼び出し
    response_message = call_chat_completion(client, model="gpt-4o", messages=messages, tools=tools)

    # デバッグ用: レスポンス全体を出力
    print("レスポンス JSON:")
    print(response_message)

    # レスポンスの処理
    final_message = process_response(response_message, client, messages)

    # 結果の表示
    print(final_message)


if __name__ == "__main__":
    main()
