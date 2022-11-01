"""
This is a basic example of how to run the snnbuilder tool using a python script
"""

from snnbuilder.models import mnist

# uses default output path SNN_Toolbox/outputs/
x = mnist.CNN_Mnist(samples=100, epochs=5)
x.train()
x.parse()
x.sim()
x.graph()


