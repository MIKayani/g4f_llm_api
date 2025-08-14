from langchain_core.language_models.llms import LLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from pydantic import Field
import g4f
from typing import Optional, List, Any, Mapping
from langchain_core.tools import tool

# ------------------ Step 1: Custom G4F LLM ---------------------
class G4FLLM(LLM):
    model: Any = Field(default=g4f.models.deepseek_r1)
    provider: Any = Field(default=g4f.Provider.LambdaChat)
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "g4f-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = g4f.ChatCompletion.create(
                model=self.model,
                provider=self.provider,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return str(response)
        except Exception as e:
            return f"Error: {e}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": str(self.model),
            "provider": str(self.provider),
            "temperature": self.temperature,
        }

# ------------------ Step 2: Define Tools ---------------------
@tool
def get_weather(city: str) -> str:
    """Return fake weather for a city."""
    return f"The weather in {city} is sunny with 25°C."

tools = [
    Tool(
        name="GetWeather",
        func=get_weather.run,
        description="Useful to get weather for a specific city"
    )
]

# ------------------ Step 3: Setup Agent ---------------------
llm = G4FLLM()

# Note: The prompt template is crucial for the new agent structure.
# It tells the agent how to reason about the tools and conversation history.
prompt = PromptTemplate.from_template(f"""
Assistant is a large language model trained by Google.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text in response to a wide range of prompts and questions, allowing it to engage in natural-sounding conversations and provide responses that are both informative and engaging.

TOOLS:
-------
Assistant has access to the following tools:
{{tools}}

To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [the final answer to the original input question]
```

Begin!

Previous conversation history:
{{chat_history}}

New input: {{input}}
{{agent_scratchpad}}
""")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ------------------ Step 4: Chat Loop ---------------------
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    print("Bot:", response['output'])
    
    # Update chat history
    chat_history.append(f"Human: {user_input}")
    chat_history.append(f"Assistant: {response['output']}")