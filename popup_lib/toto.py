
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


xs=np.linspace(0,1,10)
for x in xs:
    plt.scatter(x,x,color=cm.jet(x))


plt.show()

