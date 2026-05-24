from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="openai.gpt-oss-120b",
    input=[
        {
            "role": "user",
            "content": "Дай інформацію про популяцію жирафів у Африці"}
    ]
)

print(response.output_text)