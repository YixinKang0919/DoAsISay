# coding=utf-8
'''
@Author: Peizhen Li
@Desc: task specific configurations, e.g., environment config, and user input or task (command to the system)
Feel free to add/modify these configs to accommodate more tasks.
'''
LLM_ENGINE = "text-davinci-002"  # "text-ada-001"

# environment configuration
ENV_CONF = {'pick':  ['yellow block', 'green block', 'blue block'],  # 'red block'
          'place': ['yellow bowl', 'green bowl', 'blue bowl',]}  # 'red bowl'

# mimic user input
RAW_INPUT = 'move the light colored block to the middle'
# 'move the block to the bowl' 