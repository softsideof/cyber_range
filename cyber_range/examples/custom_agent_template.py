"""Example of building a custom agent for CyberRange."""

from cyber_range import CyberRangeEnv


class MyAgent:
    def decide(self, observation, step):
        """Return (tool_name, args_dict) for the next action."""
        if step == 0:
            return "observe_network", {}
        # TODO: your logic here
        return "observe_network", {}


def main():
    env = CyberRangeEnv(scenario="script_kiddie")
    tools = env.get_tools()
    agent = MyAgent()

    obs = env.reset()
    step = 0
    while not obs.done:
        tool_name, args = agent.decide(obs, step)
        obs = env.step(tool_name=tool_name, arguments=args)
        step += 1

    grader = getattr(env.state, "grader_result", {})
    print(f"Score: {grader.get('final_score', 0.0)}")


if __name__ == "__main__":
    main()
