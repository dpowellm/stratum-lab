import os
os.environ["OPENAI_API_KEY"] = "sk-placeholder"

from openai import OpenAI

client = OpenAI(
    api_key="sk-placeholder",
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
)

print("=== Test 1: Raw OpenAI call ===")
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=50
)
print("Response:", response.choices[0].message.content)

print("=== Test 2: Second call ===")
response2 = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role": "user", "content": "Name three colors."}],
    max_tokens=50
)
print("Response:", response2.choices[0].message.content)

print("=== Test 3: Check if patcher captured events ===")
events_path = os.environ.get("STRATUM_EVENTS_PATH", "/app/output/stratum_events.jsonl")
if os.path.exists(events_path):
    with open(events_path) as f:
        lines = f.readlines()
    print(f"SUCCESS: {len(lines)} events captured")
    for line in lines:
        print(line.strip())
else:
    print("NO EVENTS FILE — patcher did not fire")
