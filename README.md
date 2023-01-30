# Tennis Game Tracker

The project aims to track the movements of two players and the ball in a tennis match. The code reads a video file of the match and outputs the score of each player and the direction of the ball. 

## Dependencies
- cv2
- numpy
- KalmanFilter

## Video Input
The code takes as input a .mp4 video file. The default file is "tennis_match.mp4". To use a different file, change the value of the `filename` variable.

## Code structure
- The video file is read and processed frame by frame. 
- A foreground/background subtraction algorithm is applied to isolate the players and the ball.
- Mean shift algorithm is used to track the players.
- Kalman Filter is used to track the tennis ball.
- The movement of the players and the ball is monitored and used to update the score.
- The score and the direction of the ball is displayed on the output video.

## Output
The output of the code is a video with the score and the tracking of the ball and the players displayed. The score is updated as the game progresses and at the end of the game the winner is announced.

