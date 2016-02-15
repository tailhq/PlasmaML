## Welcome to spaceweather.wiki!
---

### What is it?
Spaceweather.wiki is an open platform for the space weather scientific community.
The scope is to allow scientists to compare Space Weather models, with very little effort.

### How does it work?
Spaceweather.wiki is interactive and user-friendly.
When fully operational, you will be able to choose different models to compare, run them on a given time interval,plot the results side-by-side, and calculate some standard metrics on their prediction efficiency.
The raw data resides on the server, and most of the model can run on the fly. Anybody will be able to add their own model to the website.

### What is available right now?
For now, the website is on beta version, with the purpose of showing the idea and to attract the interest of the community.

In the `Test Models` tab, you can run a Neural Network model for the prediction of the geomagnetic index Dst.
You can train and test the models either on 2006 or 2007 OMNI data (a full year). You can plot an Elman, or a Jordan network (or both), superposed to the real data. You can check how the prediction changes, by playing with the following parameters (moving the slide buttons):
 * Training data fraction: what is the fraction of dat aused for training (the remaining is used for testing)
 * The step width used in the gradient descent algorithm
 * Maximum growth parameter (regularization term)
 * Weight decay term

Please note that anytime you change a parameter, the neural network is re-trained, and the website freezes for a few seconds.

Do you want to be involved or to be updated on the progress of spaceweather.wiki?
Send me an email at e.camporeale@cwi.nl
