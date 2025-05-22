# Emotion Recognition through Speech

> A machine learning approach to detect emotion of user through speech using Long-Short Term Memory Model 

## Table of Contents
- [About](#about)
- [Data Collection](#data-collection)
- [Experiment](#experiment)
- [Improvements](#improvements)

## About
Purpose of this project is to utilize a machine learning approach to classify features within human speech that dictates emotional state.

## Data Collection
There are three methods to collect labelled audio data for training (in order of emotional authenticity):
1. Speech recreation through participants - Participants who volunteered to recreate emotion with their voices.
2. Speech recreation through actors/actresses - Skilled participants who volunteered to recreate emotion with their voices.
3. __Uninformed participants - Participants who volunteered for speech data collection but unaware for what purpose.__
***
This project utilizes uninformed participants for most accurate and raw features that are likely to be present in real life situations. 

## Experiment
**Requirements:**
* 2 Computers
* ZOOM Meeting (Mobile application)
* Keep Talking and Nobody Explodes (Video game)
* Keep Talking and Nobody Explodes Defusal Manual (PDF Document)
* 1 Participant
* 1 Moderator

**Procedure:**
1. Participant and moderator enters a ZOOM meeting.
2. Participant opens the video game and moderator opens the PDF document.
3. Participant cannot look at moderator's PDF document and moderator cannot look at participant's monitor. Optimally, they both should be in different rooms to minimize audio feedback and improve recording audio clarity.
4. Moderator records the ZOOM Meeting.
5. Participant and Moderator plays a trial run where participant will get accustomed to the rules of the game.
6. Participant and Moderator plays three runs of the game, each 5 minutes long.
7. Moderator informs participant of the experiment's objective and asks for consent.
8. If participant provides consent, their speech data will be stored and divided per sentence.
9. Each sentence is manually labelled by three investigators.
10. Each sentence will be labelled according to a majority by the investigators. If a majority is not reached, the sentence will be discarded. 

## Improvements
**Greater sample size**
The experiment was conducted hastily due to a limited time. There were also difficulty in finding participants for the experiment. As a result, the collected data is unsufficient to train a machine learning model. The model was unable to reliably identify the features for each class (emotion).

**Equal training data for each class**
Due to the nature of the experiment, there is an unproportionally large training data for the neutral class, leading to an imbalance training data for other classes such as anger or frustration. As a result, the model was unable to correctly identify the features that correspond to the classes with limited data.
