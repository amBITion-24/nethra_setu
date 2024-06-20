from roboflow import Roboflow
rf = Roboflow(api_key="rNdVM1Z4lALeigBSPfZj")
project = rf.workspace("nethra-setu").project("nethrasetu")
version = project.version(1)
dataset = version.download("yolov8")