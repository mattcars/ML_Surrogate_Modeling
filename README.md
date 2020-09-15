# Truss_Surrogate_Model

Applies a surrogate modeling technique using a neural network to approximate gradients described in the paper a paper published by MIT (see citation below).
This method takes a structural design (a truss for example) and chooses certain nodes and certain design values to be the degrees of freedom to be optimized. 
In the truss example, the degrees of freedom are the height (y-distance) of the nodes and the x-distance from the endpoints. Minimizing the weight of the truss is the optimizatoin function.
Since this problem has 2 dimensions, a 2D gradient is created to show what the actual design space looks like, with the optimizing point being the minimum. 

A neural network is then created, which takes in a sample of points from the design gradient and their objective value. The network approximates the gradient
by learning it through those points, then the optimization is run on this neural network. This saves time because the finite element analysis approach
only needs to be run on the sample points and then the neural network neds to be trained. 

For a 2D design space like this, it is easy and fast to compute the entire gradient; however, as the degrees of freedom increase and the structures get larger,
a neural network approximation can replace the costly FEA process. 




Citation:
Tseranidis, S., Brown, N. C., & Mueller, C. T. (2016). Data-driven approximation algorithms for rapid performance evaluation and optimization of civil structures. Automation in Construction, 72, 279–293. https://doi.org/10.1016/j.autcon.2016.02.002

# TODO
- Explore how many design variables can be used with existing neural net architecture
- Figure out minimum number of epochs necessary for reliable results
- Implement gaussian blur on the neural network design gradient to smooth results
