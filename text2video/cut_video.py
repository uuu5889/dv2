import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2

threshold = 4000.
fault_detect_len = 50


def cut_detect(h1, h2, thresh=threshold):
    """Detect if its a shot cut between h1 and h2."""
    e_d = np.sqrt(np.sum(np.square(h1 - h2)))
    if e_d >= thresh:
        return True
    else:
        return False

def load_video(video_path):
    """Read the video and find the shot cuts."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    # scene_id = 0
    hist_history = []
    cut_points = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            hist = np.squeeze(cv2.calcHist([blur], [0], None, [256], [0, 256]),1)
            b, a = signal.butter(3, 0.05)
            hist = signal.filtfilt(b, a, hist)

            if i == 0:
                prev_hist = hist

            if cut_detect(hist, prev_hist):
                cut_points.append(i)
                hist_history.append(hist)

            prev_hist = hist
            i += 1
        else:
            # Reach the end of the video.
            break

    cap.release()
    # cv2.destroyAllWindows()

    return cut_points, hist_history, fps

def remove_fault_detection(cut_points, hist_history, fps):
    """Remove wrongly detected shot cuts."""
    suspicious = []
    period = []
    for i in range(1, len(cut_points)-1):
        if cut_points[i] - cut_points[i-1] <= fps:
            if len(period) == 0:
                period.append(i-1)
            period.append(i)
            if cut_points[i+1] - cut_points[i] > fps:
                suspicious.append(period)
                period = []

    values = []
    for period in suspicious:
        for i in period:
            values.append(cut_points[i])
    confirmed = [point for point in cut_points if point not in values]
    print "Confirmed frames:", confirmed

    # For each suspicious period, compare its previous scene and its next scene.
    for period in suspicious:
        history = zip(cut_points, hist_history)
        first = period[0]
        last = period[-1]
        curr_scene_point = cut_points[first]
        prev_scene_point, prev_scene_hist = history[first - 1]
        next_scene_point, next_scene_hist = history[last + 1]
        if cut_detect(prev_scene_hist, next_scene_hist, thresh=7000.):
            # Different scene,curr_scene_point can be a cut point
            confirmed.append(curr_scene_point)
            print "Point added to confirmed set:", curr_scene_point
        else:
            # The scene doesn't changes. Detect a fault cut point.
            print "Abandoned point: ", curr_scene_point

    return confirmed

def visualization(video_path, true_cuts):
    cap = cv2.VideoCapture(video_path)
    i = 0
    scene_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if i in true_cuts:
                scene_id += 1
            if i>=1000:  # Start from a certain frame
                frame = cv2.putText(frame, "Scene " + str(scene_id), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('v', frame)
                #plt.pause(0.05) # To slow down the playing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            i += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


video_path = 'test.mp4'
cuts, hists, fps = load_video(video_path)
true_cuts = remove_fault_detection(cuts, hists, fps)
visualization(video_path, true_cuts)
