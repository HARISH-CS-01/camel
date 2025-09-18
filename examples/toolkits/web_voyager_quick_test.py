# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

"""
Quick WebVoyager Test - Python Version with Minimal Logging

This script provides a lightweight way to test the Python version of the
Hybrid Browser Toolkit with reflection capabilities on a few WebVoyager
tasks. Features minimal logging that only shows:
- The question/task
- Tool calls and reflection decisions
- Final answer
"""

import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits.hybrid_browser_toolkit_py.hybrid_browser_toolkit import (
    HybridBrowserToolkit,
)
from camel.types import ModelPlatformType, ModelType

load_dotenv()

# Configure logging to show INFO level for browser toolkit only
logging.basicConfig(
    level=logging.ERROR, format='%(levelname)s: %(message)s', force=True
)

# Enable INFO logging specifically for the browser toolkit
browser_logger = logging.getLogger(
    'camel.camel.toolkits.hybrid_browser_toolkit_py.hybrid_browser_toolkit'
)
browser_logger.setLevel(logging.INFO)


class MinimalLogger:
    """Minimal logger for question, tool calls, reflections, and final answer"""  # noqa: E501

    def __init__(self):
        self.tool_count = 0
        self.current_task = None

    def start_task(self, task_num, question):
        self.current_task = task_num
        self.tool_count = 0
        print(f"\n📋 TASK {task_num}: {question}")
        print("─" * 60)

    def log_tool_call(self, tool_name):
        self.tool_count += 1
        print(f"  {self.tool_count}. 🔧 {tool_name}")

    def log_reflection(self, decision):
        if decision == "SKIP":
            print("     ⚡ Reflection: Skipped (simple action)")
        else:
            print(f"     🤔 Reflection: {decision}")

    def log_final_answer(self, answer):
        print("\n💬 FINAL ANSWER:")
        print(answer)
        print("─" * 60)


# Global minimal logger
minimal_logger = MinimalLogger()


class BrowserLogHandler(logging.Handler):
    """Intercept browser toolkit logs and convert to minimal format"""

    def emit(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            msg = record.msg.lower()

            # Extract tool calls
            if 'executing browser action' in msg:
                parts = record.msg.split('executing browser action')
                if len(parts) > 1:
                    action_part = parts[1].strip()
                    action_name = (
                        action_part.split()[0]
                        .replace("'", "")
                        .replace('"', '')
                    )
                    minimal_logger.log_tool_call(action_name)

            # Extract reflection decisions - Enhanced logging
            elif (
                'skipping reflection' in msg
                or 'bypass for simple actions' in msg
            ):
                minimal_logger.log_reflection("SKIP (simple action)")

            elif 'planning model' in msg and 'evaluating' in msg:
                print("     🤔 Planning: Evaluating action...")

            elif 'should_proceed' in msg:
                if 'true' in msg:
                    minimal_logger.log_reflection("PROCEED")
                elif 'false' in msg:
                    minimal_logger.log_reflection("MODIFY/SKIP")

            elif 'planning assessment' in msg or 'planning result' in msg:
                print(f"     🤔 Planning Result: {record.msg}")

            elif 'alternative action' in msg:
                print(f"     🔄 Alternative: {record.msg}")

            # Show all planning-related logs
            elif any(
                keyword in msg
                for keyword in [
                    'planning',
                    'reflection',
                    'should_proceed',
                    'alternative',
                ]
            ):
                print(f"     🤔 {record.msg}")

            # Show any JSON planning responses
            elif '{' in record.msg and (
                'should_proceed' in record.msg or 'confidence' in record.msg
            ):
                print(f"     📋 Planning JSON: {record.msg}")


# Add handler to browser toolkit logger
browser_handler = BrowserLogHandler()
browser_logger.addHandler(browser_handler)


async def quick_test(num_tasks: int = 3):
    """Run test on WebVoyager tasks."""

    print("🚀 Starting WebVoyager Test")

    # Load dataset
    dataset_path = Path(
        "/Users/waleedalzarooni/Desktop/CAMEL PR's/Browser_research/"
        "browser-action-wrapper/camel/WebVoyager_data.jsonl"
    )
    tasks = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_tasks:
                break
            if line.strip():
                task = json.loads(line.strip())
                tasks.append(task)

    # Initialize toolkit and agent - modeled after hybrid example
    USER_DATA_DIR = "User_Data"

    # Custom tools selection for better performance
    custom_tools = [
        "browser_open",
        "browser_close",
        "browser_visit_page",
        "browser_back",
        "browser_forward",
        "browser_click",
        "browser_type",
        "browser_switch_tab",
        "browser_enter",
        "browser_get_page_snapshot",
        "browser_scroll",
        "browser_select",
        "browser_get_som_screenshot",
        "browser_press_key",
        "browser_console_view",
        "browser_console_exec",
        "browser_mouse_drag",
    ]

    # Create a lightweight external planning model (no tools)
    planning_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0.0, "max_tokens": 500},
    )

    toolkit = HybridBrowserToolkit(
        headless=False,
        user_data_dir=USER_DATA_DIR,
        planning_model=planning_model,  # enable external pre-planning
        enabled_tools=custom_tools,
        browser_log_to_file=True,  # generate detailed log file
        stealth=True,  # Using stealth mode during browser operation
        viewport_limit=True,  # Limit snapshot to current viewport
    )

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0.0, "top_p": 1},
    )

    agent = ChatAgent(
        model=model,
        system_message=BaseMessage.make_assistant_message(
            role_name="Web Navigator",
            content="""You are a web automation expert. Your job is to complete web tasks autonomously using browser tools.  # noqa: E501

IMPORTANT: You must work independently and complete the entire task without asking for further instructions. Do not stop after one action - continue working until you have fully completed the task and found the complete answer.  # noqa: E501

When given a task:
1. Navigate to the required website
2. Search for and find the requested information
3. Extract all the specific details requested
4. Provide a comprehensive final answer

Do not ask "What would you like to do next?" - instead, continue working until the task is complete.""",  # noqa: E501,
        ),
        tools=[*toolkit.get_tools()],  # Give agent access to browser tools
        toolkits_to_register_agent=[
            toolkit
        ],  # Register agent with toolkit for reflection
        max_iteration=10,
    )

    # Process tasks
    results = []

    for i, task in enumerate(tasks, 1):
        minimal_logger.start_task(i, task['ques'])

        try:
            # Create specific, actionable task prompt (following hybrid pattern)  # noqa: E501
            task_prompt = f"""
Use the browser to complete this task: {task['ques']}

Website: {task['web']}

You must:
1. Navigate to {task['web']}
2. Search for the requested information using the search functionality
3. Click on relevant results to find the specific details
4. Extract all the information requested
5. Provide a complete answer with all the details found

Continue working through all steps until you have found 
and can provide the complete answer.
Do not stop after one action.  
"""

            # Provide task context to the planning model
            toolkit.set_task_context(
                f"Task: {task['ques']} | Target site: {task['web']}"
            )

            # Single astep call - let CAMEL handle iterations internally
            response = await agent.astep(task_prompt)

            if response.msgs:
                answer = response.msgs[-1].content
                minimal_logger.log_final_answer(answer)

                results.append(
                    {'task_id': task['id'], 'success': True, 'answer': answer}
                )
            else:
                minimal_logger.log_final_answer("No response from agent")
                results.append(
                    {
                        'task_id': task['id'],
                        'success': False,
                        'error': 'No response',
                    }
                )

        except Exception as e:
            minimal_logger.log_final_answer(f"Task failed: {e!s}")
            results.append(
                {'task_id': task['id'], 'success': False, 'error': str(e)}
            )
        finally:
            # Clear planner context for the next task
            try:
                toolkit.clear_task_context()
            except Exception:
                pass

    # Clean up browser at the end
    try:
        await toolkit.browser_close()
    except Exception:
        pass  # Ignore cleanup errors

    # Simple summary
    successful = sum(1 for r in results if r['success'])
    print(
        f"\n📊 SUMMARY: {successful}/{len(results)} tasks completed successfully"  # noqa: E501
    )

    return results


if __name__ == "__main__":
    print("Script starting...")
    try:
        asyncio.run(quick_test())
        print("Script completed successfully")
    except Exception as e:
        print(f"Script failed with error: {e}")
        import traceback

        traceback.print_exc()
