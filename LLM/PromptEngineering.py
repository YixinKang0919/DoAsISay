# coding=utf-8
'''
@Author: Peizhen Li
@Desc: None
'''


TERMINATION_STRING = 'done'

GPT3_CONTEXT = """
objects = [red block, yellow block, blue block, green bowl]
# put the yellow one to the green thing.
robot.pick_and_place(yellow block, green bowl)
done()

objects = [yellow block, blue block, red block]
# move the light colored block to the middle.
robot.pick_and_place(yellow block, middle)
done()

objects = [blue block, green bowl, red block, yellow bowl, green block]
# stack the blocks.
robot.pick_and_place(green block, blue block)
robot.pick_and_place(red block, green block)
done()
"""
