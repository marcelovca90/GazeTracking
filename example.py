import time
import cv2
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gaze_tracking import GazeTracking

coords = []

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

time_results = []
res = 0
contador = 0
temp_ini = time.time()#tempo que começa o programa
while True:
    ini = time.time()  # inicia tempo dentro do while
    res = ini - temp_ini  # diferença do tempo inicial e o tempo dentro do while
    time_results.append(res) #colocar res no vetor
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    if res >=0.1:
        contador = contador+10
        coords.append([left_pupil, right_pupil, pd.Timedelta(milliseconds=contador)])
        temp_ini = ini
        print(f'contador: {contador}')
    print(f'res:{res} temp_ini:{temp_ini} ini:{ini}')
    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
    # time.sleep (1)#delay de 1s

# find empty rows
empty_rows = []
for i in range(len(coords)):
    if coords[i][0] is None or coords[i][1] is None:
        empty_rows.append(i)

# prepare data to be saved
df = pd.DataFrame(coords, columns=['left_pupil','right_pupil', 'time'])
# df = pd.DataFrame(time_results, columns= ['tempo'])

# remove empty rows
df.drop(empty_rows, inplace=True)

colors = np.random.rand(len(df.index))

x_right,y_right = zip(*df['left_pupil'])
plt.scatter(x_right,y_right,c=colors,alpha=0.5)
plt.title('left_pupil')
plt.show()

x_left,y_left = zip(*df['right_pupil'])
plt.scatter(x_left,y_left,c=colors,alpha=0.5)
plt.title('right_pupil')
plt.show()

# save data to filesystem
df.to_csv('coords.csv', index=False)