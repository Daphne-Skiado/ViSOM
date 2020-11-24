# ViSOM

Python implementation of the algorithm presented in the work of Yin, H. (2002), "Data visualization and manifold mapping using the ViSOM" in Neural Networks, 15, 1005-1016. This algorithm is a variation of the well-known self-organizing map (SOM) technique for dimensionality reduction and data visualization.

The code in "ViSOM.py" file implements this algorithm on the small dataset of object's photographs contained in the "images" folder.

Dataset information:
- 53 objects
- 5 sample images per object
- samples are RBG photographs 320 x 240

**ViSOM algorithm steps**

1. Initialise the weights of the neurons to small random values.
2. At time step t, given a randomly chosen input vector ***x***(t), find the winner neuron ***v***.
3. Update the winner’s weights.
4. Update the weights of neighbouring neurons to ***v***.
5. Repeat steps 2 –5 until the map converges.
