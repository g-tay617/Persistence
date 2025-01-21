For the simulations of continuous constant dose treatments and drug holidays, use ConstantConc_and_Holidays_8.m

For finding the critical value for one specific constant dose, run InitialAnalysis_3.m first and then run Find_xcrit_3.m. The former will output Vars_x1s01.mat. 

For finding the ciritcal values for a range of constant doses, one can load Vars_x1s01.mat and then run Find_xcrit_4.m. 

For the stochastic simulation of the x-dynamics prior to treatment using Gillespie Algorithm, use Gillespie_xDynamics_3.m. 
