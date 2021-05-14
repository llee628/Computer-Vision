import sys
import cv2
import numpy as np 
from random import randint

start=[]

def createSingleTracker():
    tracker = cv2.TrackerCSRT_create()
    return tracker


def deleteExitObject(multiTracker, frame, boxes, colors, IDs, ratio=(0.0,0.0)):
    list = []

    # Find boxes in valid range
    for i, box in enumerate(boxes):
        H, W = frame.shape[0], frame.shape[1]
        c = [box[0] + 0.5*box[2], box[1] + box[3]]
        # gets cutout from current image and gets the middle 3/4 of the image to avoid white noise in data
        f=frame[int(boxes[i][1]/8):int(7*(boxes[i][3]+boxes[i][1])/8),int(boxes[i][0]/8):int(7*(boxes[i][0]+boxes[i][2]/8))] 
        if (c[0]>ratio[0]*W) & (c[0]<(W-ratio[0]*W)) & (c[1]>ratio[1]*H) & (c[1]<(H-ratio[1]*H)):
            if (i>=len(start) or comp(f,start[i],.2)): # if cutout and original are similar enough then add
                list.append(int(i))
                if i>=len(start):
                    start.append(f)
        
    # Reinitailize the multi-tracker
    if len(list) != len(boxes):
        boxes = [tuple(boxes[k]) for k in list]
        colors = [colors[k] for k in list]
        IDs = [IDs[k] for k in list]
        multiTracker.clear()
        multiTracker = cv2.MultiTracker_create()
        for bbox in boxes:
            multiTracker.add(createSingleTracker(), frame, bbox)

    return multiTracker, boxes, colors, IDs


def IoU(box1, box2):
    # Intersection rectangle
    x1 = max(box1[0], box2[0])
    x2 = min(box1[0]+box1[2], box2[0]+box2[2])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[1]+box1[3], box2[1]+box2[3])
    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Union area
    unionArea = box1[2]*box1[3] + box2[2]*box2[3] - interArea

    # IoU
    iou = interArea / unionArea

    return iou


def addEnterObject(multiTracker, frame, boxes, colors, IDs, yoloboxes, objectNum):
    newBoxes = []
    restBoxes = yoloboxes.copy()
    newColors = []
    newIDs = []
    oldlist = []

    # Find old objects
    for i in range(len(boxes)):
        IoUs = []
        for j in range(len(yoloboxes)):
            IoUs.append(IoU(boxes[i], yoloboxes[j]))
        if IoUs == []:
            continue
        f=frame[int(boxes[i][1]/8):int(7*(boxes[i][3]+boxes[i][1])/8),int(boxes[i][0]/8):int(7*(boxes[i][0]+boxes[i][2]/8))]
        if i>=len(start) or (max(IoUs)>0.5 and (comp(f,start[i],.2))): # if cutout and original are similar enough then add
            oldlist.append([i, np.argmax(IoUs)])
            if i>=len(start):
                    start.append(f)
    # Assign existed ID
    deleteidx = []
    for idx in oldlist:
        newBoxes.append(tuple(yoloboxes[idx[1]]))
        newColors.append(colors[idx[0]])
        newIDs.append(IDs[idx[0]])
        deleteidx.append(idx[1])
    restBoxes = [box for i, box in enumerate(restBoxes) if i not in deleteidx]
    
    newBoxes.extend(restBoxes)
    newColors.extend([(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(len(restBoxes))])
    newIDs.extend(list(range(objectNum+1, objectNum+len(restBoxes)+1)))
    
    # Reinitailize the multi-tracker
    multiTracker.clear()
    multiTracker = cv2.MultiTracker_create()
    for bbox in newBoxes:
        multiTracker.add(createSingleTracker(), frame, bbox)

    return multiTracker, newBoxes, newColors, newIDs


def outputInfo(iter, boxes, IDs):
    ''' 
    The order of output information is 
    (frame, ID, X, Y, W, H, confidence, class, visibility).
    More details see the paper MOT16: A Benchmark for Multi-Object Tracking.
    '''
    info = ""
    for i, box in enumerate(boxes):
        info = info + "{},{},".format(iter, IDs[i]) + ",".join(map(str, box)) + ",1,-1,-1,-1\n"
    return info


def yoloDetection(iter, yoloData):
    idx = yoloData[:,0]==iter
    yoloboxes = yoloData[idx,2:6]
    yoloboxes = [tuple(box) for box in yoloboxes]
    return yoloboxes

def comp(im1,im2,c):
    # compares with correlation coefficient
    w=min(im2.shape[0],im1.shape[0])
    h=min(im2.shape[1],im1.shape[1])
    if h<=0 or w<=0:
        return False
    im1=cv2.resize(im1, (h,w), interpolation = cv2.INTER_AREA)
    im2=cv2.resize(im2, (h,w), interpolation = cv2.INTER_AREA)
    if (np.corrcoef(im1.flat, im2.flat)[0][1]<c): #if less than coefficient, not similar enough and so don't include
        return False
    else:
        return True

def main():
    # Setting =====================
    videofile = "TUD_Campus/TUD-Campus-raw.webm"
    # videofile = "TUD_Campus/TUD-Campus-raw.mp4"
    yoloTxt = "TUD_Campus/gtAsYolo.txt"
    saveOutput = True
    outputTxt = "TUD_Campus/test_result.txt"
    saveVideo = True
    outputVideo = "TUD_Campus/test_result.mp4"
    yoloCheckFreq = 20
    # =============================
    # videofile = "yolov4/match_V008.mp4"
    # yoloTxt = "yolov4/bounding_box_V008.txt"
    # saveOutput = True
    # outputTxt = "test_video_v008.txt"
    # saveVideo = True
    # outputVideo = "test_result_v008.mp4"
    # yoloCheckFreq = 20
    # =============================

    # Read video
    cap = cv2.VideoCapture(videofile)
    if not cap.isOpened():
        print("Could not open video " + videofile)
        sys.exit()
    success, frame = cap.read()
    if not success:
        print("Could not read video " + videofile)
        sys.exit()
    
    # Initailize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Wout, Hout = frame.shape[0], frame.shape[1]
    # writer = cv2.VideoWriter(outputVideo, fourcc, fps, (Wout, Hout))
    writer = cv2.VideoWriter(outputVideo, fourcc, fps, (640,480))
    if saveVideo:
        writer.write(frame)

    # Read yolo file
    yolofile = open(yoloTxt, "r")
    lines = yolofile.read().splitlines()
    yoloData = np.array([element.split(",") for element in lines]).astype(np.float)
    # yoloData = np.array([element.split(" ") for element in lines]).astype(np.float)
    bboxes = yoloDetection(yoloData[0,0], yoloData)
    colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(len(bboxes))]
    IDs = [int(i+1) for i in range(len(bboxes))]

    # Write initial info in txt file
    iter = yoloData[0,0]
    if saveOutput:
        info = outputInfo(yoloData[0,0], bboxes, IDs)
        txt = open(outputTxt, "w")
        txt.write(info)
        txt.close()
    
    # Create and Initialize MultiTracker object
    multiTracker = cv2.MultiTracker_create()
    for bbox in bboxes:
        multiTracker.add(createSingleTracker(), frame, bbox)

    #Original Images to compare to
    for i in range(len(bboxes)):
        start.append(frame[int(bboxes[i][1]/8):int(7*(bboxes[i][3]+bboxes[i][1])/8),int(bboxes[i][0]/8):int(7*(bboxes[i][0]+bboxes[i][2]/8))])

    # Process video and track objects
    objectNum = max(IDs)
    iter = 1
    while cap.isOpened():
        iter += 1 
        # read video util the end
        success, frame = cap.read()
        if not success:
            break

        if iter==yoloData[0,0]:
            # Create and Initialize MultiTracker object
            multiTracker = cv2.MultiTracker_create()
            for bbox in bboxes:
                multiTracker.add(createSingleTracker(), frame, bbox)

            # draw initial objects
            for i, newbox in enumerate(bboxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

        elif iter>yoloData[0,0]:
            # get updated location of objects in subsequent frames
            success, boxes = multiTracker.update(frame)
            multiTracker, boxes, colors, IDs = deleteExitObject(multiTracker, frame, boxes, colors, IDs)

            # run yolo again after a few frames
            if (iter % yoloCheckFreq) == 0:
                yoloboxes = yoloDetection(iter, yoloData)
                if yoloboxes!=[]:
                    multiTracker, boxes, colors, IDs = addEnterObject(multiTracker, frame, boxes, colors, IDs, yoloboxes, objectNum)
                    objectNum = max(IDs)

            # draw tracked objects
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            
            # write in txt file
            if saveOutput:
                info = outputInfo(iter, boxes, IDs)
                txt = open(outputTxt, "a")
                txt.write(info)
                txt.close()

        # show frame
        cv2.putText(frame, "Frame: "+str(int(iter)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
        cv2.imshow('MultiTracker', frame)

        # write frame in video
        if saveVideo:
            writer.write(frame)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Print notes and Release the video capture and video write objects
    if saveOutput:
        print("Save " + outputTxt)
    if saveVideo:
        print("Save " + outputVideo)
        writer.release()
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
