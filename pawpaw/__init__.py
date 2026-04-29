"""pawpaw — build and run .paw neural programs locally.

Two-step usage for agentic workflows:

# Once, offline:
pawpaw.build(
    "Classify the user message as 'trivial' or 'substantive'.",
    save_to="programs/triage.paw",
)

# In your service / agent:
triage = pawpaw.load("programs/triage.paw")
mood = pawpaw.load("programs/mood.paw")
intent = pawpaw.load("programs/intent.paw")
# All three programs share one base model in memory.

label = triage(user_message)
if label == "substantive":
    ...
"""
from pawpaw.api import build, clear_cache, load
from pawpaw.pipeline import CompileResult
from pawpaw.runtime import Program

compile = build

__all__ = [
    "build",
    "load",
    "Program",
    "CompileResult",
    "clear_cache",
]
