from tangrams import *

s1 = \
'{"pieces": [["large triangle2", "270", "0 0"], ["medium triangle", "90", "1 1"], ["square", "0", "1 0"], ["small triangle2", "0", "0 0"], ["small triangle1", "270", "2 0"], ["large triangle1", "180", "2 0"], ["parrallelogram", "0", "2 0"]], "size": "5 5"}'
task = Task()
task.create_from_json(s1)
task.show()
