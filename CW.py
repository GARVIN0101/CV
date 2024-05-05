import cv2
import numpy as np
from tqdm import tqdm

def process_videos(video_paths, output_paths):
    weights_path = "yolov3.weights"
    cfg_path = "yolov3.cfg"
    labels_path = "obj.names"
    labels = open(labels_path).read().strip().split("\n")
    print(labels)

    total_videos = len(video_paths)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    for i, (video_path, output_path) in enumerate(zip(video_paths, output_paths), 1):
        print(f"Processing video {i}/{total_videos}: {video_path}")
        capture = cv2.VideoCapture(video_path)
        writer = None
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=frame_count)

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            if writer is None:
                writer = cv2.VideoWriter(output_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

            (H, W) = frame.shape[:2]
            resized_frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
                points = []
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > 0.2:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    points.append((x + w // 2, y + h // 2))
                    color = 120
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    if classIDs[i] < len(labels):
                        text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            writer.write(frame)
            progress_bar.update(1)

        capture.release()
        writer.release()
        progress_bar.close()

    cv2.destroyAllWindows()


video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
output_paths = ["output1.mp4", "output2.mp4", "output3.mp4"]
process_videos(video_paths, output_paths)