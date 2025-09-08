#!/usr/bin/env bash
# @yaml
# signature: |-
#   ask_supervisor <question>
#   end_of_supervisor
# docstring: |
#   Pause execution, update the supervisor on your progress and ask the supervisor for guidance.
# end_name: end_of_supervisor
# arguments:
#   question:
#     type: string
#     description: The summary of what you have done so far and the question you want to ask the supervisor.
#     required: false

echo "<ASK_SUPERVISOR>$*</ASK_SUPERVISOR>"