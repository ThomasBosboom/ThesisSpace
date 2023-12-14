import os
import sys

print(os.path.realpath(__file__))

print(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# print(sys.path)

