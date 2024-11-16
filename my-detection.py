import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

camera = jetson.utils.videoSource("/home/nvidia/Desktop/download.jpg")

display = jetson.utils.videoOutput("display://0")   # 'my_video.mp4' for file
while display.IsStreaming():
      img = camera.Capture()
      if img is None: # capture timeout
        continue
      detections = net.Detect(img)
      print(detections[0])

      display.Render(img)
      display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))


