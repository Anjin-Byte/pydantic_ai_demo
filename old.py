"""
Module: main.py

This module demonstrates a workflow that:
1) Generates tasks for a birthday party (using an LLM-based "agent").
2) Subdivides tasks into manageable steps (or concludes they can be done in a single day).
3) Assigns tasks to appropriate specialists if exactly one can do the job.
4) Checks for duplicates among tasks.
5) Builds a hierarchical representation (tree) of these tasks.
6) Exports the tree to JSON and evaluates the quality of the entire task decomposition.

Overall Goals:
- Show how to orchestrate multiple LLM "agents" (defined by Pydantic-based classes in `pydantic_ai`) to handle
  different subtasks (e.g., plan generation, subdivision, assignment, duplication check).
- Construct a hierarchical tree of tasks and sub-tasks using the `anytree` library.
- Demonstrate how to integrate the LLM's responses into a structured approach with Pydantic models.

Dependencies used:
- pydantic: for defining data models to enforce structure in the LLM responses.
- anytree: for building and exporting the tree structure of tasks.
- dotenv: for loading environment variables (like OpenAI API keys).
- openai: LLM functionality (through a custom interface pydantic_ai).

Usage:
- The user runs this script to automatically:
  1) Generate tasks for a birthday party.
  2) Potentially subdivide each task if needed.
  3) Assign each subdivided task to one specialist if exactly one can handle it.
  4) Check for duplication.
  5) Export a final hierarchical task tree to JSON.
  6) Print an evaluation of the process.
"""

import os  # Provides a portable way of using operating system dependent functionality
import json  # Used for serialization and deserialization of JSON
from dotenv import load_dotenv  # Loads environment variables from a .env file
from pydantic import BaseModel  # Base class for creating Pydantic models
from typing import Optional, List, Set  # For type annotations

from anytree import Node, RenderTree  # Tools for building and rendering trees
from anytree.render import ContRoundStyle  # A style option for rendering the tree
from anytree.exporter import JsonExporter  # Exports an anytree tree to JSON

from pydantic_ai.models.openai import OpenAIModel  # Custom LLM model wrapper
from pydantic_ai.providers.openai import OpenAIProvider  # Provider to plug in OpenAI
from pydantic_ai import Agent  # Agent class that ties prompt, model, and result type together

# Load environment variables (e.g., OPENAI_API_KEY) from .env
load_dotenv()

# Instantiate an OpenAI-based model. "gpt-4o" is a label being used for demonstration.
model = OpenAIModel(
    "gpt-4o",  # Model identifier/name
    provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")),  # The provider to handle API calls
)

# ----------------------#
# Pydantic Models      #
# ----------------------#

class LessSimple(BaseModel):
    """
    Represents a single task with an associated task number,
    description, and reasoning.
    """
    taskNum: int  # The index or ID of the task
    task: str  # Description or name of the task
    reasoning: str  # Short explanation or rationale for the task


class Simple(BaseModel):
    """
    Holds a list of LessSimple tasks, effectively a wrapper for
    multiple tasks.
    """
    tasks: List[LessSimple]


class Specialist(BaseModel):
    """
    Represents an event-planning specialist with a role, background,
    strengths, and preferred projects.
    """
    name: str  # Full name of the specialist
    role: str  # Job title or specialization
    background: str  # Brief description of their professional background
    strengths: List[str]  # A list of key strengths
    preferred_projects: List[str]  # Project types the specialist prefers to work on


class AllSpecialists(BaseModel):
    """
    Wraps a list of Specialist objects.
    """
    specialists: List[Specialist]


class LLMAssignabilityResponse(BaseModel):
    """
    Represents an LLM-based response regarding whether a single specialist
    can do a specific task, and if the task can be done in one day.
    """
    is_assignable: bool  # Whether exactly one specialist can handle the task
    assigned_specialist: Optional[str]  # If only one is suitable, their name
    can_be_done_in_one_day: bool  # Whether the entire task can be done in a single day
    reasoning: str  # Short explanation from the LLM

class LLMBestFitResponse(BaseModel):
    """
    Represents an LLM-based response indicating:
      1) The single 'best fit' specialist for the task.
      2) A short rationale.
      3) Whether it can be done in one day.
    """
    best_specialist: str  # The name of the single chosen specialist
    reasoning: str        # Brief explanation
    can_be_done_in_one_day: bool


# ----------------------#
# Agents (LLM Orchestration)
# ----------------------#

# Create an Agent for assessing if exactly one specialist can handle a given task
# and if it can be done in one day.
assignability_agent: Agent[str, LLMAssignabilityResponse] = Agent(
    model,  # The LLM model configured above
    result_type=LLMAssignabilityResponse,  # The expected response model
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

# Create a new "best fit" agent that picks exactly one specialist no matter what.
# You might place this next to your other agent definitions:
best_fit_agent: Agent[str, LLMBestFitResponse] = Agent(
    model,  # The same LLM model you already created
    result_type=LLMBestFitResponse,
    system_prompt=(
        "You are a lenient assignment system. You MUST always select EXACTLY ONE 'best' specialist "
        "from a given list of specialists for the provided task. You do not reject any task.\n\n"
        "1) Output must contain exactly three keys:\n"
        "     best_specialist: (str) Name of the single chosen specialist.\n"
        "     reasoning: (str) short explanation.\n"
        "     can_be_done_in_one_day: (bool) is the entire task feasible in one day.\n"
        "2) If multiple specialists could do the task, you must pick the one who most closely matches "
        "the required skills or the user's preferences.\n"
        "3) If no specialist obviously matches, pick the 'closest' in your judgment.\n"
        "4) Your output MUST be valid JSON with exactly those three fields, nothing more.\n"
    ),
)

# Create an Agent to generate a set of tasks (Simple) for a scenario
task_generation_agent: Agent[None, Simple] = Agent(
    model,  # The LLM model
    result_type=Simple,  # The expected response model (Simple)
    system_prompt=(
        "You are a birthday party planner."
        "You specialize in giving step-by-step plans to accomplish the customer's birthday demands."
        "Provide a list of high-level tasks."
    ),
)

# Create a prompt for the task_generation_agent to generate tasks for a birthday party
task_generation_prompt = (
    "Make a plan to throw a birthday party!"
    "I want to grow a garden in which to throw the party. "
    "It is currently June and I want to throw the party in 12 months."
    "I currently reside in the country and have plenty of fertile land."
)

# Run the task generation agent synchronously, generating high-level tasks
result = task_generation_agent.run_sync(task_generation_prompt)

# Print out the tasks returned by the agent
print("High-level Tasks Returned:")
for task in result.data.tasks:
    print(f" - {task.task} (reasoning: {task.reasoning})")

# Agent that decides if a single day is enough to complete a given task
# If not, it subdivides the task into exactly two sub-tasks.
subdivision_agent: Agent[str, Simple] = Agent(
    model,
    result_type=Simple,
    system_prompt=(
        "You are an efficient day-planner. For a given task, determine if it can be accomplished in one day. "
        "If it can, respond with a single task that is exactly 'NO SUBDIVISION SUB_SYSTEM'. "
        "'reasoning' summary explaining why no subdivision is needed. If you lack information or knowledge of context to know how long something will take, take special care to explain this point."
        "If it cannot, break the task into exactly 2 sub-tasks, each of which can be accomplished in one day. "
        "Provide the result as a JSON object with the key 'tasks' mapping to a list of sub-task objects, "
        "each having 'taskNum' and 'task'. "
        "'reasoning' field (no more than 20 words) that explains why the sub-task is necessary. "
        "Return the result as JSON with a key 'tasks' mapping to a list of objects."
    ),
)

# Agent to generate a set of 10 specialists
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

# Prompt that instructs the specialists_agent to create 10 unique event-planning specialists
specialists_user_prompt = (
    "Generate 10 unique event-planning specialists. "
    "Include name, role, background, a few strengths, and preferred projects. "
    "Be creative and ensure each specialist has distinct skills."
)

# Generate the specialists with the agent
specialists_result = specialists_agent.run_sync(specialists_user_prompt)

# Print the generated specialists
print("Generated specialists:")
for idx, spec in enumerate(specialists_result.data.specialists, start=0):
    print(f"\n#{idx} - {spec.name} ({spec.role})")
    print(f"  Background: {spec.background}")
    print(f"  Strengths: {spec.strengths}")
    print(f"  Preferred Projects: {spec.preferred_projects}")

# Store the list of Specialist objects
specialists: List[Specialist] = specialists_result.data.specialists


class DupCheckResponse(BaseModel):
    """
    Model representing the duplication-check response. Tells whether
    a new task is a near-duplicate of previously processed tasks,
    along with a short rationale.
    """
    is_duplicate: bool
    reasoning: str


# Agent that checks if a new task is considered a duplicate of existing tasks
dup_check_agent: Agent[str, DupCheckResponse] = Agent(
    model,
    result_type=DupCheckResponse,
    system_prompt=(
        "You are a duplication detection system. "
        "You receive a list of tasks that have already been processed, and a new task. "
        "You must decide if the new task is effectively the 'same' or 'too similar' to any of the old tasks.\n\n"
        "You MUST return valid JSON with exactly:\n"
        "  is_duplicate (bool)\n"
        "  reasoning (str)\n"
        "No additional keys.\n"
    ),
)


class TaskNode(Node):
    """
    Extension of anytree's Node class. Represents a task node in the hierarchy,
    including an optional 'reasoning' attribute to store short textual explanations
    about the node.
    """
    def __init__(self, name: str, reasoning: str = "", parent=None, **kwargs):
        """
        :param name: Name of the node, typically the task description
        :param reasoning: Short explanation or rationale (used in the tree display)
        :param parent: Parent node in the tree (None if this is the root)
        :param kwargs: Additional arguments for the base Node
        """
        super().__init__(name, parent=parent, **kwargs)
        self.reasoning = reasoning


class TaskAssignabilityEvaluation(BaseModel):
    """
    Contains information about whether a task is assignable to a single
    specialist, which specialist, and a summary of the reasoning.
    """
    task: str  # The task description
    is_assignable: bool = False  # Indicates if a single specialist can handle the task
    assigned_specialist: Optional[str] = None  # The name of that specialist
    reasoning: Optional[str] = None  # Explanation or notes about the decision


def evaluate_assignability(
    task: str, specialists: List[Specialist]
) -> TaskAssignabilityEvaluation:
    """
    Given a task and a list of Specialist objects, uses the assignability_agent
    to determine if exactly one specialist can handle the task, whether it can
    be done in one day, and provides a short reasoning.

    :param task: The task description to evaluate
    :param specialists: List of Specialist objects
    :return: TaskAssignabilityEvaluation object containing the results
    """
    # Convert specialist data to JSON for the LLM
    specialists_json = json.dumps([s.model_dump() for s in specialists], indent=2)
    user_prompt = (
        f"Specialists:\n{specialists_json}\n\n"
        f"Task: {task}\n\n"
        "Decide if exactly one specialist can handle it, "
        "whether it can be done in one day, and give short reasoning."
    )

    # Run the agent with the user prompt
    llm_response = assignability_agent.run_sync(user_prompt)

    # Build and return a TaskAssignabilityEvaluation from the LLM response
    return TaskAssignabilityEvaluation(
        task=task,
        is_assignable=llm_response.data.is_assignable,
        assigned_specialist=llm_response.data.assigned_specialist,
        reasoning=(
            f"{llm_response.data.reasoning} "
            f"One-day feasibility: {llm_response.data.can_be_done_in_one_day}"
        ),
    )
    
def evaluate_assignability_best_fit(
    task: str, specialists: List[Specialist]
) -> TaskAssignabilityEvaluation:
    """
    A more lenient version of the assignability logic. It forces a
    single best-fit specialist to be selected, rather than checking if
    exactly one person can do it. The LLM will:
      1) Evaluate which specialist is the best match.
      2) Decide if it can be completed in one day.
      3) Return an explanation (reasoning).
    
    :param task: The task description
    :param specialists: A list of Specialist objects
    :return: TaskAssignabilityEvaluation with a single assigned specialist
             and the LLM's explanation.
    """
    # Convert specialists to JSON for the LLM
    specialists_json = json.dumps([s.model_dump() for s in specialists], indent=2)
    user_prompt = (
        f"Specialists:\n{specialists_json}\n\n"
        f"Task: {task}\n\n"
        "You must return exactly one best_specialist, a short reasoning, and can_be_done_in_one_day (bool)."
    )

    # Run our new 'best fit' agent
    llm_response = best_fit_agent.run_sync(user_prompt)

    # Always assign the single best_fit specialist from the LLM response
    return TaskAssignabilityEvaluation(
        task=task,
        is_assignable=True,  # We force an assignment here
        assigned_specialist=llm_response.data.best_specialist,
        reasoning=(
            f"{llm_response.data.reasoning} "
            f"One-day feasibility: {llm_response.data.can_be_done_in_one_day}"
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
    # Convert the processed tasks to JSON for the LLM
    tasks_json = json.dumps(list(processed_tasks), indent=2)
    user_prompt = (
        f"Already Processed Tasks:\n{tasks_json}\n\n"
        f"New Task:\n{new_task}\n\n"
        "Decide if the new task is a near-duplicate or effectively the same "
        "as any of the old tasks."
    )

    # Run the duplication check agent and return True/False based on is_duplicate
    response = dup_check_agent.run_sync(user_prompt)
    return response.data


def subdivide_node(
    node: TaskNode, processed_tasks: List[str], depth: int = 0, max_depth: int = 3
) -> None:
    """
    Subdivides tasks represented as TaskNodes. If the task can be done in a day
    or is assigned to a single specialist, it doesn't subdivide further.
    Otherwise, it breaks the task into sub-tasks.

    :param node: Current task node
    :param processed_tasks: List of tasks already processed (for duplication checks)
    :param depth: Current depth of recursion in the tree
    :param max_depth: Maximum depth to prevent infinite or excessive recursion
    """
    # Stop if we've reached the max allowable depth
    if depth >= max_depth:
        return

    # Potential duplication check logic (commented out):
    #   new_task_str = node.name.strip()
    #   is_rep = is_repetitive(new_task_str, processed_tasks)
    #   if is_rep.is_duplicate:
    #       TaskNode("IS REPETITIVE", reasoning=is_rep.reasoning, parent=node)
    #       return
    #   else:
    #       processed_tasks.append(new_task_str)

    # Evaluate if the task can be assigned to exactly one specialist
    evaluation = evaluate_assignability(node.name, specialists)
    if evaluation.is_assignable:
        assigned_label = f"ASSIGNED: {evaluation.assigned_specialist}"
        TaskNode(assigned_label, reasoning=evaluation.reasoning, parent=node)
        return

    # If no single specialist can handle it exclusively, decide if the task can be done in one day.
    # If not, subdivide into 2 sub-tasks.
    new_task_str = node.name.strip()
    context_dict = {"processed_tasks": processed_tasks}
    prompt = (
        f"{{global_context}}: {json.dumps(context_dict, indent=2)}\n"
        f"Current Task: {new_task_str}"
    )

    response = subdivision_agent.run_sync(prompt)
    tasks_data = response.data.tasks

    # If the returned sub-task is "NO SUBDIVISION SUB_SYSTEM", we forcibly assign
    # to a specialist using the BEST-FIT approach (guarantees exactly one specialist).
    if len(tasks_data) == 1 and tasks_data[0].task.strip().upper() == "NO SUBDIVISION SUB_SYSTEM":
        # Using the best-fit function instead of the strict approach
        force_specialist_assign = evaluate_assignability_best_fit(node.name, specialists)

        # We'll always have exactly one assigned specialist
        assigned_label = f"ASSIGNED: {force_specialist_assign.assigned_specialist}"
        TaskNode(
            assigned_label,
            reasoning=force_specialist_assign.reasoning,
            parent=node
        )
        return
    else:
        # If it needs subdivision, create two child nodes and subdivide further
        for subtask_obj in tasks_data:
            child_node = TaskNode(
                subtask_obj.task,  # Subtask name
                reasoning=subtask_obj.reasoning,  # Brief reasoning
                parent=node
            )
            # Recursively subdivide the child node
            subdivide_node(child_node, processed_tasks, depth + 1, max_depth)


# Create the root node for the entire birthday party tasks
root = TaskNode("Birthday Party Tasks")

# A list to keep track of tasks we've processed
global_context_set = list()

# For each of the high-level tasks, build the hierarchical structure
for t in result.data.tasks:
    task_node = TaskNode(t.task, reasoning=t.reasoning, parent=root)
    subdivide_node(task_node, global_context_set, depth=0, max_depth=3)

# Render the final task tree to the console
print("\nFinal Task Tree:\n")
for pre, fill, node in RenderTree(root, style=ContRoundStyle()):
    print(f"{pre}{node.name} | Reason: {node.reasoning}")

# Export the tree to JSON format
exporter = JsonExporter(indent=2, sort_keys=True)
tree_json = exporter.export(root)

# Write the JSON to a file
with open("combined_task_tree.json", "w") as f:
    f.write(tree_json)

print("\nTask tree saved to combined_task_tree.json\n")

# Build a prompt to evaluate the quality of the task tree
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

# Create an Agent for evaluating the final task tree
evaluation_agent = Agent(
    model,
    result_type=str,  # We only expect a string response for the evaluation
    system_prompt=evaluation_prompt,
)

# Run the evaluation and print out the result
evaluation_result = evaluation_agent.run_sync("")
print("Evaluation Result:\n")
print(evaluation_result)
