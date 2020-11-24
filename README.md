# ViSOM

Python implementation of the algorithm presented in the work of Yin, H. (2002), "Data visualization and manifold mapping using the ViSOM" in Neural Networks, 15, 1005-1016. This algorithm is a variation of the well-known self-organizing map (SOM) technique for dimensionality reduction and data visualization.

The code in "ViSOM.py" file implements this algorithm on the small dataset of object's photographs contained in the "images" folder.

Dataset information:
- 53 objects
- 5 sample images per object
- samples are RBG photographs 320 x 240

**SOM**

SOM is an unsupervised learning algorithm, which uses a finite grid or lattice of neurons to fill and frame the input data. Nodes are usually arranged in a 2D rectangular grid. In the SOM, a neighbourhood learning is adopted to form topological ordering among the neurons in the map. The close data points are likely to be projected to nearby nodes. Thus the map can be used to show the relative relationships among data points.
The ViSOM algorithm is a variation of SOM that tries to learn a map in a way that the distances between neurons in the data space are in proportion to those in the map space.

**ViSOM algorithm steps**

1. Initialise the weights of the neurons to small random values.
2. At time step t, given a randomly chosen input vector ***x***(t), find the winner neuron ***v***.
3. Update the winner’s weights.
4. Update the weights of neighbouring neurons to ***v***.
5. Repeat steps 2–4 until the map converges.
