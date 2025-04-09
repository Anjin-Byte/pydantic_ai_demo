import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List, Set

from anytree import Node, RenderTree
from anytree.render import ContRoundStyle
from anytree.exporter import JsonExporter

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent

load_dotenv()
model = OpenAIModel(
    "gpt-4o",
    provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")),
)

class LessSimple(BaseModel):
    taskNum: int
    task: str
    reasoning: str

class Simple(BaseModel):
    tasks: List[LessSimple]
    
class Specialist(BaseModel):
    name: str
    role: str
    background: str
    strengths: List[str]
    preferred_projects: List[str]

class AllSpecialists(BaseModel):
    specialists: List[Specialist]
    
class LLMAssignabilityResponse(BaseModel):
    is_assignable: bool
    assigned_specialist: Optional[str] 
    can_be_done_in_one_day: bool
    reasoning: str
    
assignability_agent: Agent[str, LLMAssignabilityResponse] = Agent(
    model,
    result_type=LLMAssignabilityResponse,
    system_prompt=(
        "You are a task assignment and duration evaluator. "
        "Given a set of specialists (in JSON), and a specific task, decide:\n"
        "1) If exactly one specialist can handle the entire task.\n"
        "2) If the task can be done in one day.\n"
        "3) Provide a concise reasoning.\n\n"
        "Please return your answer as JSON with the following fields:\n"
        "  is_assignable (bool): whether exactly one specialist can do the entire task\n"
        "  assigned_specialist (str|null): name of that one specialist, if any\n"
        "  can_be_done_in_one_day (bool): if the entire task can be done in one day\n"
        "  reasoning (str): short explanation\n\n"
        "Your output must be valid JSON (no extra keys)."
    ),
)

task_generation_agent: Agent[None, Simple] = Agent(
    model,
    result_type=Simple,
    system_prompt=(
        "You are a birthday party planner."
        "You specialize in giving step-by-step plans to accomplish the customer's birthday demands."
        "Provide a list of high-level tasks."
    ),
)

task_generation_prompt = (
    "Make a plan to throw a birthday party!"
    "I want to grow a garden in which to throw the party. "
    "It is currently June and I want to throw the party in 12 months."
    "I currently reside in the country and have plenty of fertile land."
)

result = task_generation_agent.run_sync(task_generation_prompt)
print("High-level Tasks Returned:")
for task in result.data.tasks:
    print(f" - {task.task} (reasoning: {task.reasoning})")

subdivision_agent: Agent[str, Simple] = Agent(
    model,
    result_type=Simple,
    system_prompt=(
        "You are an efficient day-planner. For a given task, determine if it "
        "can be accomplished in one day. A global context is provided in JSON "
        "under 'processed_tasks', listing tasks handled in other branches. "
        "If the current task or a similar one is already processed, respond "
        "with a single task that is exactly 'NO SUBDIVISION' and reasoning. "
        "If the task cannot be done in one day, break it into exactly 2 sub-tasks "
        "that can each be completed in one day. Return a JSON object with key 'tasks' "
        "mapping to a list of objects with keys 'taskNum', 'task', and 'reasoning'."
    ),
)

specialists_agent: Agent[str, AllSpecialists] = Agent(
    model,
    result_type=AllSpecialists,
    system_prompt=(
        "You are a recruiter generating detailed dossiers for various event-planning specialists.\n\n"
        "When given a user prompt about how many specialists to create, "
        "you will produce valid JSON with the following structure:\n"
        "{\n"
        "  'specialists': [\n"
        "    {\n"
        "      'name': str,\n"
        "      'role': str,\n"
        "      'background': str,\n"
        "      'strengths': list of strings,\n"
        "      'preferred_projects': list of strings\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Return ONLY valid JSON. No extra keys or text.\n"
    ),
)

specialists_user_prompt = (
    "Generate 10 unique event-planning specialists. "
    "Include name, role, background, a few strengths, and preferred projects. "
    "Be creative and ensure each specialist has distinct skills."
)


specialists_result = specialists_agent.run_sync(specialists_user_prompt)

print("Generated specialists:")
for idx, spec in enumerate(specialists_result.data.specialists, start=0):
    print(f"\n#{idx} - {spec.name} ({spec.role})")
    print(f"  Background: {spec.background}")
    print(f"  Strengths: {spec.strengths}")
    print(f"  Preferred Projects: {spec.preferred_projects}")
    
specialists: List[Specialist] = specialists_result.data.specialists

class TaskNode(Node):
    def __init__(self, name: str, reasoning: str = "", parent=None, **kwargs):
        super().__init__(name, parent=parent, **kwargs)
        self.reasoning = reasoning

class TaskAssignabilityEvaluation(BaseModel):
    task: str
    is_assignable: bool = False
    assigned_specialist: Optional[str] = None
    reasoning: Optional[str] = None

""" specialists = [
    Specialist(name="Cook", description="Expert in cooking."),
    Specialist(name="Baker", description="Expert in baking."),
    Specialist(name="Interior Designer", description="Expert in decor and design."),
    Specialist(name="Gardener", description="Expert in landscaping."),
] """

def evaluate_assignability(task: str, specialists: List[Specialist]) -> TaskAssignabilityEvaluation:
    """
    Use an LLM to decide if exactly one specialist can handle the task
    AND whether the task can be completed in one day.
    Returns a TaskAssignabilityEvaluation object (for consistency with existing code).
    """
    specialists_json = json.dumps([s.model_dump() for s in specialists], indent=2)
    user_prompt = (
        f"Specialists:\n{specialists_json}\n\n"
        f"Task: {task}\n\n"
        "Decide if exactly one specialist can handle it, "
        "whether it can be done in one day, and give short reasoning."
    )

    llm_response = assignability_agent.run_sync(user_prompt)
    return TaskAssignabilityEvaluation(
        task=task,
        is_assignable=llm_response.data.is_assignable,
        assigned_specialist=llm_response.data.assigned_specialist,
        reasoning=(
            f"{llm_response.data.reasoning} "
            f"One-day feasibility: {llm_response.data.can_be_done_in_one_day}"
        )
    )

def subdivide_node(node: TaskNode, global_context: Set[str], depth: int = 0, max_depth: int = 3) -> None:
    """
      1) Checks if the task is already in global_context => 'NO SUBDIVISION' if so.
      2) Checks if the task can be assigned to a single specialist => mark as assigned if yes.
      3) Otherwise calls subdivision_agent to see if we subdivide or not.
    """

    if depth >= max_depth:
        return

    # check if task is repetitive
    normalized_task = node.name.strip().lower()
    if normalized_task in global_context:
        TaskNode("NO SUBDIVISION", reasoning="Duplicate task. No further action needed.", parent=node)
        return
    else:
        global_context.add(normalized_task)

    # check if task is assignable
    evaluation = evaluate_assignability(node.name, specialists)
    if evaluation.is_assignable:
        assigned_label = f"ASSIGNED: {evaluation.assigned_specialist}"
        TaskNode(assigned_label, reasoning=evaluation.reasoning, parent=node)
        return

    # check if it can be done in one day or needs subdivision
    context_dict = {
        "processed_tasks": list(global_context)
    }
    prompt = (
        f"{{global_context}}: {json.dumps(context_dict)}\n"
        f"Current Task: {node.name}"
    )
    response = subdivision_agent.run_sync(prompt)
    tasks_data = response.data.tasks

    if len(tasks_data) == 1 and tasks_data[0].task.strip().upper() == "NO SUBDIVISION":
        TaskNode(
            tasks_data[0].task.strip().upper(),
            reasoning=tasks_data[0].reasoning,
            parent=node
        )
    else:
        for subtask_obj in tasks_data:
            child_node = TaskNode(subtask_obj.task, reasoning=subtask_obj.reasoning, parent=node)
            subdivide_node(child_node, global_context, depth + 1, max_depth)


root = TaskNode("Birthday Party Tasks")
global_context_set = set()

for t in result.data.tasks:
    task_node = TaskNode(t.task, reasoning=t.reasoning, parent=root)
    subdivide_node(task_node, global_context_set, depth=0, max_depth=3)

print("\nFinal Task Tree:\n")
for pre, fill, node in RenderTree(root, style=ContRoundStyle()):
    print(f"{pre}{node.name} | Reason: {node.reasoning}")

exporter = JsonExporter(indent=2, sort_keys=True)
tree_json = exporter.export(root)

with open("combined_task_tree.json", "w") as f:
    f.write(tree_json)

print("\nTask tree saved to combined_task_tree.json\n")

evaluation_prompt = f"""
You are a quality evaluator for a hierarchical task decomposition system.
Evaluate the following task tree based on:
1. Structural Coherence
2. Non-Redundancy
3. Appropriate Granularity
4. Termination and Convergence
5. Context Sensitivity

Task Tree:
{tree_json}

Provide a summary of the system's performance and recommendations.
"""

evaluation_agent = Agent(
    model,
    result_type=str,
    system_prompt=evaluation_prompt,
)

evaluation_result = evaluation_agent.run_sync("")
print("Evaluation Result:\n")
print(evaluation_result)
