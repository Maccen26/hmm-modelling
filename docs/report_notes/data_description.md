# Motivation & Data Description

## Motivation
Overall:
Occupancy detection is a key component in building energy management systems. Accurate occupancy detection can lead to significant energy savings by optimizing heating, ventilation, and air conditioning (HVAC) systems based on actual usage patterns. Its widely regonized that occupant behovour is random, uncertain and effects the energy consumption of buildings. Therefore, it is important to have a reliable method for detecting occupancy in order to improve energy efficiency and reduce costs.

This project: 
This projects aims to develop statistical models to predict occupancy behaviour in rooms regarding when occupants are present or absent and when they open windows or close them again. The overall goal is to improve energy efficiency in buildings by predicting the optimal comfort thermal conditions for occupants. 



Data:
1. Data is CO^2 from a single bedroom in a residential building. 
Link: 
https://www2.imm.dtu.dk/~jkmo/ 


2. Data recievned from DTU. 
Link is: https://gitlab.compute.dtu.dk/users/sign_in
Link to holiday data for both 2023 and 2024: https://www.generalblue.com/calendar/denmark/denmark-holidays-2024

Description: 
Data is from rooms at DTU. The data is collected from k (1 or 2?) sensors in each working rooms for the professors. It has been collected in 2023 and there is a weather data set. The average time interval between the measurements is 10 minutes (around 600 seconds). 
The holiday data is used to identify the holidays in Denmark, which can affect occupancy patterns in the rooms.
Both weather data and holiday data is from 2023-11-01 to 2024-02-09 and so is the $CO_2$ data. 

This data differentiate itselfs from the bedroom data in such a sense that there is now multiple rooms and mutiple states. The following states should be considered: 

1. No one has been present in the room for a long time. (steady state)
2. No one has been present in the room for a long time and someone has entered the room during the 30 minutes interval and left it again. (short spike)
3. No one has been present in the room for a long time and someone has entered the room during the 30 minutes interval and is present (up going spike)
4. Someone has been present in the room for a long time and has left the room during the 30 minutes interval. (down going spike)
5. Someone has been present in the room for a long time and is still present in the room. (steady state)
6. Severel people has been present in the room for a long time and is still present in the room. (steady state)
