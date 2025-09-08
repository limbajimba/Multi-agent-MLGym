from __future__ import annotations
from typing import TYPE_CHECKING

import queue, threading, sys, time
from mlgym.agent.base import BaseAgent, TrajectoryStep
from rich.logging import RichHandler

if TYPE_CHECKING:
    from mlgym.backend.base import APIStats
   
# live interrupt queue for supervisor messages
#_interrupt_q: queue.Queue[str] = queue.Queue()
#def _stdin_listener():
#    for line in iter(sys.stdin.readline, ''):
#        _interrupt_q.put(line.rstrip())

#threading.Thread(target=_stdin_listener, daemon=True).start()

class SupervisorAwareAgent(BaseAgent):
    SUB_CMD = "submit_request"          
    ASK_CMD = "ask_supervisor"
    
    def __init__(self, name, args):
        super().__init__(name, args)
        
        for h in self.logger.handlers:
            if isinstance(h, RichHandler):
                h.markup = True

        # single in-thread interrupt buffer 
        self._pending_interrupt: str | None = None

    def _run_step(self, observation: str | None):
        log = self.logger

        assert self._env is not None 
        assert self.config is not None

        # LLM step
        state = (self._env.communicate(self.tools.state_command.name)
                 if self.tools.state_command else "")
        thought, action, output = self.forward(
            observation, self._env.get_available_actions(), state
        )

        run_action: str = self.tools.guard_multiline_input(action).strip()

        done = False
        observation = None
        execution_t0 = time.perf_counter()

        assert self._env is not None
        assert self.config is not None

        #log.info("step=%s run_action=%s", self._env.current_step, action.splitlines()[0])

        # interrupt handling
        #if not _interrupt_q.empty():
         #   observation = f"[SUPERVISOR INTERRUPT] {_interrupt_q.get()}"
          #  log.info(observation)
           # execution_time = time.perf_counter() - execution_t0
            #done = False
        
        # if there are other actions
        if run_action.startswith(self.SUB_CMD):
            summary = run_action[len(self.SUB_CMD):].strip()
            log.info("[magenta]SUBMIT REQUEST:[/magenta] %s", summary)
            approve = input("[SUPERVISOR approve?] (yes/no + feedback) > ").strip()
            if approve.lower().startswith("y"):
                observation,_ , done, _info  = self._env.step(self.tools.submit_command)
                self.info.update(_info)
                done = True
                execution_time = time.perf_counter() - execution_t0
                #self._append_history({"role": "user", "content": observation, "agent":"SUPERVISOR"})
            else:
                observation = approve
                log.info(observation)
                done = False
                execution_time = time.perf_counter() - execution_t0
                #self._append_history({"role": "user", "content": observation, "agent":"SUPERVISOR"})

        elif run_action.startswith(self.ASK_CMD):
            question = run_action[len(self.ASK_CMD):].strip()
            log.info("[cyan]AGENT ASKS:[/cyan] %s", question)
            feedback = input("[SUPERVISOR] > ")
            observation = f"[SUPERVISOR FEEDBACK] {feedback}"
            log.info(observation)
            done = False
            execution_time = time.perf_counter() - execution_t0
            #self._append_history({"role": "user", "content": observation, "agent":"SUPERVISOR"})
        else:
            observation, _, done, _info = self._env.step(run_action)
            self.info.update(_info)
            execution_time = time.perf_counter() - execution_t0

        # Use BaseAgent's trajectory handling (already done in forward())
        # No need to duplicate trajectory step creation here
        return observation, done