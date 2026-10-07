# Noter Week 5 

DTU doesnt collect data right now. 

Calculate the correlation between the ar params 

If you put a new state in, it MUST have intepretability

Think about what you include in the covarites - makes it hard to predict


Use the window opening indicator in some of the rooms to see and how it relates to the states. 

Incooporate the indicatos in the model such that you dont trust them completely. 

Try with AR(1) with a few more states. 

$$
dc/t = -\alpha c + p 
$$
ODE , and you can make it a SDE with 

$$
dC = (-\alpha C + p)dt + \sigma dW
$$

Try to incorporate huminity as a extra observation in the model 
Try to incorporate the solar radians for a feedback model. 

Pick a state that reflects feedback or window opening. Use a distribution of the states. 

The physical parameters should be the same across the rooms. 

Use different rooms for to infer physical params. 


Look at the SDE and write up the solution as a AR(1) process. 
(Simulation and inference with stochastil differential eqautions, chapter 1.13)


