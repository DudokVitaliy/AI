import boto3
import json
import textwrap

client = boto3.client(
    service_name="bedrock-runtime",
    region_name="eu-north-1"
)

def call_bedrock(prompt: str) -> str:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 400,
            "temperature": 0.5
        }
    }

    response = client.invoke_model(
        modelId="amazon.nova-lite-v1:0",
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())

    return result["output"]["message"]["content"][0]["text"]


def print_answer(title: str, text: str):
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)
    print(textwrap.fill(text, width=80))
    print("=" * 70 + "\n")


def menu():
    while True:
        print("\nAI ДЕБАГЕР (AWS Bedrock)")
        print("1 - Пояснити помилку Python")
        print("2 - Виправити код")
        print("3 - Написати чистий код")
        print("4 - Власне питання")
        print("0 - Вихід")

        choice = input("\nОбери опцію: ")

        if choice == "1":
            error = input("Введи текст помилки:\n")
            prompt = f"""
Поясни українською цю помилку Python:

{error}

Опиши:
- причина
- чому це сталося
- як виправити
"""
            answer = call_bedrock(prompt)
            print_answer("Пояснення помилки", answer)

        elif choice == "2":
            code = input("Встав свій код:\n")
            prompt = f"""
Виправ цей Python код і поясни зміни українською:

{code}
"""
            answer = call_bedrock(prompt)
            print_answer("Виправлення коду", answer)

        elif choice == "3":
            task = input("Що потрібно зробити?: ")
            prompt = f"""
Напиши чистий Python код українською пояснюючи:

Завдання: {task}

Дотримуйся best practices.
"""
            answer = call_bedrock(prompt)
            print_answer("Чистий код", answer)

        elif choice == "4":
            q = input("Твоє питання: ")
            prompt = f"Відповідай українською: {q}"
            answer = call_bedrock(prompt)
            print_answer("Відповідь AI", answer)

        elif choice == "0":
            print("Вихід. Гарного дня!")
            break

        else:
            print("Невірний вибір")


if __name__ == "__main__":
    menu()