# Outline for thesis

## Formalia checklist for thesis
- [ ] Names of supervisors ( & signature?)
- [ ] References on all images. 
- [ ] Abstract (resume)
- [ ] AI declaration. 
- [ ] Consistent reference on sources. 
- [ ] All figures are referenced and used in the text. 
- [ ] All tables are referenced and used in the text.
- [ ] All figures and tables results can be understand from the caption. 
- [ ] Each chapter has a short introduction and conclusion.

Write out formalia from Jans. 




Outline of the thesis:

1. Introduction
    - What is the generel problem of the project?
    And what is the known challenges of the problem 
    - What is the specific problem of the project? 
    - What is the challenges of the specific problem? And how do we  adress these challenges\ (maybe also how does the specfic challenges relate to the generel challenges)
    -  What is the research question and objectives of the project?
    - How does this project give a tourough explanation of the problem and the solution? (thesis outline explained in words)

2. Data: 
    - What data was used in the project? And how was the raw data collected? How was the data preprocessed? Is there any known isssues with the collected data? How was this adressed? 

3. Theoretical background: 
    - Why were the type of statistcal model chosen? And how does this model relate to the problem? (Talk about diffusions and argument for why ARHMM is a good model for this problem)
    - Inference types and why was the inference type chosen? 
    (Feedforward for likelihood, Forward-Backward for inference of true state distributions, Backward Sampling for simulations of paths, EM vs Autograd for parameter estimation) 

4. Methodology
    - How was the models implemented? And why was it implemented in this way? (maybe talk about autograd vs. EM algorithm) why did I choose this and not just a known library? 
    - What packages and libraries were used in the project? And why were these packages and libraries chosen? What coding languages 
    - How was the models trained and validated? 
    - How can other people reproduce the results of the project? 

5. Findings and Results
    - Statistical difference berween models. What model best explain the data? And why could that be? Why doesnt the other models explain as well? (Parameters of 1 or 2 models for comparison)
    - Predictive metrics of the models. How well do the models predict the data? And why could that be? Why doesnt the other models predict as well? 
    - What model are the best for the specific problem? And why could that be? 
    - Results of the best models when simulating feedback paths.
    - How data would we need? 
6. Discussion:
    - What are the constraints of the diffusion model, i.e. what counterproofs of results ? And how does the model 
