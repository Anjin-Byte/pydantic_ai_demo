from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent
from pydantic import BaseModel
from anytree import Node, RenderTree
from anytree.exporter import JsonExporter
from anytree.render import ContRoundStyle
from dotenv import load_dotenv

load_dotenv()
import os

model = OpenAIModel(
    "gpt-4o",
    provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")),
)


class LessSimple(BaseModel):
    taskNum: int
    task: str
    reasoning: str


class Simple(BaseModel):
    tasks: list[LessSimple]


agent: Agent[None, Simple] = Agent(
    model,
    result_type=Simple,
    system_prompt=(
        "You are a birthday party planner."
        "You specialize in giving step-by-step plans to accomplish the customer's birthday demands."
        "Provide a list of high-level tasks."
    ),
)

result = agent.run_sync(
    "Make a plan to throw a birthday party!"
    "I want to grow a garden in which to throw the party. "
    "It is currently June and I want to throw the party in 12 months."
    "I currently reside in the country and have plenty of fertile land."
)

for task in result.data.tasks:
    print(str(task) + "\n")

subdivision_agent: Agent[str, Simple] = Agent(
    model,
    result_type=Simple,
    system_prompt=(
        "You are an efficient day-planner. For a given task, determine if it can be accomplished in one day. "
        "If it can, respond with a single task that is exactly 'NO SUBDIVISION'. "
        #"Do not be timid in ceasing to subdivide further. If you lack information or knowledge of context to know how long something will take, consider no subdivision." #experimental
        "'reasoning' summary explaining why no subdivision is needed. If you lack information or knowledge of context to know how long something will take, take special care to explain this point."
        "If it cannot, break the task into exactly 2 sub-tasks, each of which can be accomplished in one day. "
        "Provide the result as a JSON object with the key 'tasks' mapping to a list of sub-task objects, "
        "each having 'taskNum' and 'task'. "
        "'reasoning' field (no more than 20 words) that explains why the sub-task is necessary. "
        "Return the result as JSON with a key 'tasks' mapping to a list of objects."
    ),
)


class TaskNode(Node):
    def __init__(self, name, reasoning: str = "", parent=None, **kwargs):
        super().__init__(name, parent=parent, **kwargs)
        self.reasoning = reasoning


def subdivide_node(
    node: TaskNode, depth: int = 0, max_depth: int = 3
) -> None:  # recursive
    if depth >= max_depth:
        return

    response = subdivision_agent.run_sync(node.name)

    if (
        len(response.data.tasks) == 1
        and response.data.tasks[0].task.strip().upper() == "NO SUBDIVISION"
    ):
        child = TaskNode(
            response.data.tasks[0].task.strip().upper(),
            reasoning=response.data.tasks[0].reasoning,
            parent=node,
        )
        return
    else:
        for sub in response.data.tasks:
            print(sub.reasoning)
            child = TaskNode(sub.task, reasoning=sub.reasoning, parent=node)
            subdivide_node(child, depth + 1, max_depth)


root = TaskNode("Birthday Party Tasks")

for task in result.data.tasks:
    task_node = TaskNode(task.task, reasoning=task.reasoning, parent=root)
    subdivide_node(task_node, depth=0, max_depth=3)

for pre, fill, node in RenderTree(root, style=ContRoundStyle()):
    print(f"{pre}{node.name} | Reason: {node.reasoning}")

exporter = JsonExporter(indent=2, sort_keys=True)
json_str = exporter.export(root)

path = "task_tree.json"
with open(path, "w") as file:
    file.write(json_str)

print(f"Task tree has been saved at {path}")
