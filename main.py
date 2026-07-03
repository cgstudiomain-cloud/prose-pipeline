from dotenv import load_dotenv
from anthropic import Anthropic
from rich import print

load_dotenv()
client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    messages=[{"role": "user", "content": "Reply with exactly: pipeline online"}],
)
print(f"[green]{response.content[0].text}[/green]")
print(f"tokens in: {response.usage.input_tokens}, out: {response.usage.output_tokens}")