import cv2
import numpy as np
from KalmanFilter import KalmanFilter

# Select the file
filename = 'tennis_match'
cap = cv2.VideoCapture(filename + '.mp4')

#variable keeps who's turn it is
turn = 0

if (filename == 'tennis_match'):
    first_player = (1030, 170, 120, 180)
    second_player = (400, 650, 140, 220)
    turn = 1
elif (filename == 'tennis_match2'):
    first_player = (800, 100, 80, 140)
    second_player = (550, 650, 140, 220)
    turn = 2
    
def create_mask_and_hist(frame, x, y, w, h):
    """
    Create a mask and histogram for a given region of interest in the frame.
    """
    roi = frame[x:x+w, y:y+h]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((255., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


def process_frame(frame, roi_hists, fgbg, term_crit):
    """
    Process a frame of the video to track objects.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    region_of_interest = np.zeros((height, width, 3))
    points = np.array([[726,76], [292,963], [1634, 953],[1172,76]])
    cv2.fillPoly(region_of_interest, pts=[points], color=(255, 255, 255))
    region_of_interest = cv2.cvtColor(region_of_interest.astype('uint8'), cv2.COLOR_BGR2GRAY) 
    ret,region_of_interest = cv2.threshold(region_of_interest,250,255,0)
    region_of_interest = cv2.bitwise_and(frame, region_of_interest)
    
    # Remove the background
    dst = fgbg.apply(region_of_interest)

    track_windows_out = []

    for i, roi_hist in enumerate(roi_hists):
     
        dst = cv2.medianBlur(dst, 5)

        # Perform morphological opening
        kernel = np.ones((3,3), np.uint8)  
        dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)

        # Perform morphological closing
        kernel = np.ones((3,3), np.uint8)  
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
        dst = cv2.dilate(dst,kernel,iterations = 3)
       
        cv2.imshow('dst', dst)
        
        # Find contours in the image
        contours, hier = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Append contours that meet size requirements
        cList = []
        for c in contours:
            if cv2.contourArea(c) > 40 and cv2.contourArea(c) < 600:
                (x, y), radius = cv2.minEnclosingCircle(c)
                cList.append(c)
                
        # Apply meanshift to get new location
        ret, track_window = cv2.meanShift(dst, track_windows[i], term_crit)
        track_windows_out.append(track_window)

    return [track_windows_out, cList]

player1 = 0 #player1 score
player2 = 0 #player2 score
direction = 1 #direction of the ball
player1Contact = False
player2Contact = False
isGameOver = False

def writeScore(yList,frame):
    global player1,player2
    global player1Contact,player2Contact
    global direction, turn, isGameOver
    dir = 0
    
    #calculating dir value for direction estimation
    for i in range(0,int(fps/3)):
       dir += yList[-1-i] - yList[-1-i-1] 
    
    if dir > 0 and direction != 1:
        direction = 1
        if player1Contact and turn==1:
            player1 += 1
            turn =2
    elif dir <= 0 and direction != -1:
        direction = -1
        if player2Contact and turn==2:
            player2 += 1
            turn = 1
            
    # Player scores text
    score = "Player 1 Hit: " + str(player1) +"    /     Player 2 Hit: " + str(player2)
    
    # Define the position of the text
    orgScore = (1200, 1000)

    # Define the font type, scale, color, and thickness
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    colorScore = (255, 255, 255)
    thickness = 4
    
    # Draw the text on the image
    cv2.putText(frame, score, orgScore, fontFace, fontScale, colorScore, thickness)

# get frame rate
fps = cap.get(cv2.CAP_PROP_FPS)

# Take first frame of the video
ret, frame = cap.read()

# get width and height
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the initial location and size of the players
track_windows = [first_player,  second_player]

roi_hists = [create_mask_and_hist(frame, x, y, w, h) for x, y, w, h in track_windows]

# Create the background subtractor
fgbg = cv2.createBackgroundSubtractorKNN()

# Set up the termination criteria
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1 )

# Save result video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("result_" + filename + ".mp4", fourcc, 20.0, (frame.shape[1], frame.shape[0]))

yList = [80] * (int(fps/3)+1)

KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)

lastFrame = frame.copy()

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        # Process the current frame
        pr= process_frame(frame, roi_hists, fgbg, term_crit)
        track_windows = pr[0]
        cList = pr[1]
        
        #for the players
        x1, y1, w1, h1 = track_windows[0]
        x2, y2, w2, h2 = track_windows[1]
        margin = 5
        inter = 40 #interaction distance
        centers=[]

        for c in cList:
            x, y, w, h = cv2.boundingRect(c)
            if  (not (x+margin > x1 and x < x1+w1+margin and y+margin > y1 and y < y1+h1+margin) and 
                    not (x+margin > x2 and x < x2+w2+margin and y+margin > y2 and y < y2+h2+margin)
                    and x>500 and x<1500 and y>160 and y<900):
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                yList.append(y)
                centers.append(np.array([[x+ w/2], [y+h/2]]))
                if (x+inter > x1 and x < x1+w1+inter and y+inter > y1 and y < y1+h1+inter):
                    player1Contact = True
                    player2Contact = False
                if (x+inter > x2 and x < x2+w2+inter and y+inter > y2 and y < y2+h2+inter):
                    player2Contact = True
                    player1Contact = False
                    
                
        if (len(centers) > 0):        
            # Draw the detected circle
            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)

            # Predict
            (x, y) = KF.predict()
            # Draw a rectangle as the predicted object position
            cv2.rectangle(frame, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (255, 0, 0), 2)

            # Update
            (x1, y1) = KF.update((centers[0]))

            # Draw a rectangle as the estimated object position
            cv2.rectangle(frame, (int(x1 - 15), int(y1 - 15)), (int(x1 + 15), int(y1 + 15)), (0, 0, 255), 2)

            cv2.putText(frame, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", (int(centers[0][0] + 15), int(centers[0][1] - 15)), 0, 0.5, (0,191,255), 2)

        writeScore(yList,frame)
        
        # Draw rectangles for the players
        for i, t in enumerate(track_windows):
            x, y, w, h = t
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 220), 2)
            img = cv2.putText(frame, "Player " + str(i+1), (int(x + 15), int(y + 10)), 0, 0.5, (255, 255, 255), 2)
            cv2.imshow('img', img)

        
        out.write(frame)
        lastFrame = frame.copy()
        
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
    else:
        break

# END OF THE GAME
if turn == 1:
    winner = 2
else:
    winner = 1
    
# Announce winner
winText = "Player " + str(winner) + " wins!" 
orgWin = (600, 500)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
colorWin = (0, 255, 255)
cv2.putText(lastFrame, winText, orgWin, fontFace, 3, colorWin, 6)
cv2.imshow('img', lastFrame)
cv2.waitKey(6000) # 6 seconds stop for winner announce!
out.write(lastFrame)
out.release()
cv2.destroyAllWindows()

