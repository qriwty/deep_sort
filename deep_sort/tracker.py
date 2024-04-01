import numpy as np

from .sort import nn_matching
from .sort.preprocessing import non_max_suppression
from .sort.tracker import Tracker as DeepSortTracker
from .sort.detection import Detection
from .deep.extractor import Extractor

from .deep.resnet import ResNet18Config


class Tracker:
    def __init__(self):
        self.tracks = None
        max_cosine_distance = 0.7
        nn_budget = 100

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

        self.tracker = DeepSortTracker(metric)

        self.extractor = Extractor(ResNet18Config())

    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()

            return

        bboxes = np.asarray([d[:-2] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-2] for d in detections]
        classes = [d[-1] for d in detections]

        indices = non_max_suppression(bboxes, 1.0, scores)

        features = self.extractor(frame, bboxes)

        dets = []
        for idx in indices:
            dets.append(Detection(bboxes[idx], scores[idx], features[idx], classes[idx]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            tracks.append(track)

        self.tracks = tracks
