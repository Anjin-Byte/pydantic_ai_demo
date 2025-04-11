import os
import json
from pydantic import BaseModel
from typing import Optional, List

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from anytree import Node, RenderTree
from anytree.render import ContRoundStyle
from anytree.exporter import JsonExporter

load_dotenv()
model = OpenAIModel(
    "gpt-4o",
    provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")),
)


class Task(BaseModel):
    task: str
    reasoning: str


class TaskList(BaseModel):
    tasks: List[Task]


task_generation_agent: Agent[None, TaskList] = Agent(
    model,
    result_type=TaskList,
    system_prompt=(
        "You are a birthday party planner.\n"
        "You specialize in giving step-by-step plans to accomplish the customer's birthday demands.\n"
        "Provide a list of high-level tasks as JSON with a 'tasks' field that contains an array of objects.\n"
        "Here is an example of the expected JSON format:\n"
        "{\n"
        '  "tasks": [\n'
        "    {\n"
        '      "task": "Plan the party theme and guest list",\n'
        '      "reasoning": "Deciding the theme and guest list sets the tone for the event."\n'
        "    },\n"
        "    {\n"
        '      "task": "Design and set up the garden area",\n'
        '      "reasoning": "The garden must be planned well in advance for a successful party."\n'
        "    }\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
    ),
)

party_prompt = (
    "Make a plan to throw a birthday party!"
    "I want to grow a garden in which to throw the party. "
    "It is currently June and I want to throw the party in 12 months."
    "I currently reside in the country and have plenty of fertile land."
)

initial_task_list = task_generation_agent.run_sync(party_prompt)
for task in initial_task_list.data.tasks:
    print(task)


class Specialist(BaseModel):
    """
    Represents an event-planning specialist with a role, background,
    strengths, and preferred projects.
    """

    name: str
    role: str
    background: str
    strengths: List[str]
    preferred_projects: List[str]


class AssignmentAndSubDiv(BaseModel):
    is_assignable: bool
    assigned_specialist: Optional[Specialist]
    sub_tasks: List[Task]
    reasoning: str


AGGRESSIVNESS = f"{50}-{60}"
BRANCH_FACTOR = 2
dual_purpose_agent: Agent[str, AssignmentAndSubDiv] = Agent(
    model,
    result_type=AssignmentAndSubDiv,
    system_prompt=(
        "You are a task assignment and subdivision evaluator agent. Analyze the given task description and decide whether it is "
        "sufficiently granular to be directly assigned to a single specialist or should be subdivided into exactly two subtasks. \n\n"
        "Guidelines:\n"
        f"1. If approximately {AGGRESSIVNESS} percent or more of the task clearly aligns with one specialist's domain, even if minor elements fall outside, respond with:\n"
        "   - is_assignable: true\n"
        "   - assigned_specialist: the specialist's name\n"
        "   - sub_tasks: an empty array\n"
        "2. If the task involves two or more distinctly different areas with no clear dominant focus such that it cannot be handled by a single specialist, respond with:\n"
        "   - is_assignable: false\n"
        "   - assigned_specialist: null\n"
        f"   - sub_tasks: exactly {BRANCH_FACTOR} subtask descriptions that together cover the entire task (plain text, without any specialist assignments).\n"
        "3. Provide a brief reasoning (up to 50 words) explaining your decision.\n\n"
        "Return your answer as valid JSON with exactly these keys: is_assignable, assigned_specialist, sub_tasks, and reasoning.\n\n"
    ),
)


force_best_fit: Agent[str, AssignmentAndSubDiv] = Agent(
    model,
    result_type=AssignmentAndSubDiv,
    system_prompt=(
        "You are a task assignment evaluator agent. Analyze the given task description and choose the best-fit specialist from the list provided.\n\n"
        "You must always select one specialist from the list, even if the fit is imperfect.\n"
        "Evaluate based on how well the task aligns with their role, background, strengths, and preferred projects.\n\n"
        "Guidelines:\n"
        "1. Return a JSON object with these exact fields:\n"
        "   - is_assignable: true\n"
        "   - assigned_specialist: the selected specialist (as JSON, including name, role, background, strengths, preferred_projects)\n"
        "   - sub_tasks: always an empty array\n"
        "   - reasoning: a short explanation (no more than 50 words) of why this specialist is the best fit.\n\n"
        "2. Do NOT subdivide the task. Your job is to match a whole task to a single specialist.\n\n"
        "Return ONLY valid JSON with exactly those keys. No text outside the JSON."
    ),
)


class AllSpecialists(BaseModel):
    specialists: List[Specialist]


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

NUM_SPECIALISTS = 10
specialists_user_prompt = (
    f"Generate {NUM_SPECIALISTS} unique event-planning specialists. "
    "Include name, role, background, a few strengths, and preferred projects. "
    "Be creative and ensure each specialist has distinct skills."
)

specialists_result = specialists_agent.run_sync(specialists_user_prompt)
specialists_list = specialists_result.data.specialists
specialists_json = json.dumps([s.model_dump() for s in specialists_list], indent=2)
print(f"Work crew: \n {specialists_json}")

""" 
for task in initial_task_list.data.tasks:
    task_prompt = f"{task.task}\nAvailable Specialists: {specialists_json}"
    result = dual_purpose_agent.run_sync(task_prompt)
    print(json.dumps(result.data.model_dump(), indent=2)) 
"""


class DuplicateCheckResponse(BaseModel):
    """
    Model representing the duplication-check response. Tells whether
    a new task is a near-duplicate of previously processed tasks,
    along with a short rationale.
    """

    is_duplicate: bool
    reasoning: str


duplicate_check_agent: Agent[str, DuplicateCheckResponse] = Agent(
    model,
    result_type=DuplicateCheckResponse,
    system_prompt=(
        "You are a duplication detection system. "
        "You receive a list of tasks that have already been processed, and a new task. "
        "Decide only if the new task is functionally redundant — meaning it would involve substantially the same work or output — as any task in the list. Similar themes or phrasing are not enough.\n"
        "You MUST return valid JSON with exactly:\n"
        "  is_duplicate (bool)\n"
        "  reasoning (str)\n"
        "No additional keys.\n"
    ),
)


def is_repetitive(new_task: str, processed_tasks: List[str]) -> bool:
    """
    Checks if a given new task is too similar (duplicate) to any tasks in
    `processed_tasks`.

    :param new_task: The task to check for duplication
    :param processed_tasks: A list of previously processed task strings
    :return: A boolean indicating duplication (True if duplicate)
    """
    tasks_json = json.dumps(list(processed_tasks), indent=2)
    user_prompt = (
        f"Already Processed Tasks:\n{tasks_json}\n\n"
        f"New Task:\n{new_task}\n\n"
        "Decide if the new task is a near-duplicate or effectively the same "
        "as any of the old tasks."
    )

    response = duplicate_check_agent.run_sync(user_prompt)
    return response.data


class TaskNode(Node):
    def __init__(self, name: str, reasoning: str = "", parent=None, **kwargs):
        super().__init__(name, parent=parent, **kwargs)
        self.reasoning = reasoning

    def to_dict(self) -> dict:
        return {
            "type": "TaskNode",
            "name": self.name,
            "reasoning": self.reasoning,
            "children": [child.to_dict() for child in self.children],
        }


class AssignedTaskNode(TaskNode):
    def __init__(
        self,
        name: str,
        reasoning: str = "",
        assigned_specialist: dict = None,
        parent=None,
        **kwargs,
    ):
        super().__init__(name, reasoning=reasoning, parent=parent, **kwargs)
        self.assigned_specialist = assigned_specialist

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["type"] = "AssignedTaskNode"
        data["assigned_specialist"] = self.assigned_specialist
        return data


def build_task_tree(tasks: List[Task], max_depth: int = 3) -> TaskNode:
    root = TaskNode("Birthday Party Plan", "Root")
    processed = []

    for task in tasks:
        top_node = TaskNode(name=task.task, reasoning=task.reasoning, parent=root)
        subdivide_node(top_node, processed, depth=1, max_depth=max_depth)

    return root

REPETITIVE_CHECK = False
def subdivide_node(
    node: TaskNode, processed_tasks: List[str], depth: int = 0, max_depth: int = 3
) -> None:
    if depth >= max_depth:
        prompt = f"{node.name}\nAvailable Specialists: {specialists_json}"
        result = force_best_fit.run_sync(prompt)
        data = result.data

        assigned = (
            data.assigned_specialist.model_dump() if data.assigned_specialist else None
        )

        parent = node.parent
        node.parent = None

        AssignedTaskNode(
            name=node.name,
            reasoning=data.reasoning,
            assigned_specialist=assigned,
            parent=parent,
        )
        return

    prompt = f"{node.name}\nAvailable Specialists: {specialists_json}"
    result = dual_purpose_agent.run_sync(prompt)
    data = result.data

    if data.is_assignable:
        assigned = (
            data.assigned_specialist.model_dump() if data.assigned_specialist else None
        )

        parent = node.parent
        node.parent = None

        AssignedTaskNode(
            name=node.name,
            reasoning=data.reasoning,
            assigned_specialist=assigned,
            parent=parent,
        )
        return
    else:
        if not REPETITIVE_CHECK:
            for sub_task in data.sub_tasks:
                task_name = sub_task.task.strip()
                processed_tasks.append(task_name)

                child_node = TaskNode(
                    name=task_name, reasoning=sub_task.reasoning, parent=node
                )
                subdivide_node(child_node, processed_tasks, depth + 1, max_depth)
                
        else:   
            added_children = 0
            sub_nodes = []

            for sub_task in data.sub_tasks:
                task_name = sub_task.task.strip()
                if is_repetitive(task_name, processed_tasks):
                    continue

                processed_tasks.append(task_name)

                child_node = TaskNode(
                    name=task_name, reasoning=sub_task.reasoning, parent=node
                )
                sub_nodes.append(child_node)
                added_children += 1
                
            for child in sub_nodes:
                subdivide_node(child, processed_tasks, depth + 1, max_depth)

            if added_children == 0:
                prompt = f"{node.name}\nAvailable Specialists: {specialists_json}"
                result = force_best_fit.run_sync(prompt)
                data = result.data

                assigned = (
                    data.assigned_specialist.model_dump() if data.assigned_specialist else None
                )

                parent = node.parent
                node.parent = None

                AssignedTaskNode(
                    name=node.name,
                    reasoning=data.reasoning,
                    assigned_specialist=assigned,
                    parent=parent,
                )


root_node = build_task_tree(initial_task_list.data.tasks)

print("\nTask Hierarchy Tree:\n")
for pre, fill, node in RenderTree(root_node, style=ContRoundStyle()):
    line = f"{pre}{node.name}"
    if isinstance(node, AssignedTaskNode):
        line += f" [ASSIGNED: {node.assigned_specialist['name']}]"
    print(line)

exporter = JsonExporter(indent=2, sort_keys=True)
with open("task_tree.json", "w") as f:
    f.write(exporter.export(root_node))
