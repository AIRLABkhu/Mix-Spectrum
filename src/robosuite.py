from env.robosuite.core import make_robosuite
#ALL_TASKS = ["Door", "TwoArmPegInHole", "NutAssemblyRound", "TwoArmLift"]
env = make_robosuite(task="Door",
                     mode="train",
                     scene_id=0)
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)