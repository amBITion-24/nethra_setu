
from roboflow import Roboflow
rf = Roboflow(api_key="LcP3OZofUZly5m44zT38")
project = rf.workspace("mahisha").project("bus-front-view")
version = project.version(3)
dataset = version.download("yolov8")