from collections import OrderedDict
import numpy as np
from scipy.spatial import distance
from motrackers.utils.misc import get_centroid
from motrackers.track import Track
import sys

class Tracker:
    """
    Greedy Tracker with tracking based on ``centroid`` location of the bounding box of the object.
    This tracker is also referred as ``CentroidTracker`` in this repository.

    Parameters
    ----------
    max_lost : int
        Maximum number of consecutive frames object was not detected.
    tracker_output_format : str
        Output format of the tracker.
    """

    def __init__(self, max_lost=5, tracker_output_format='mot_challenge', exit_point=(320,420), exit_direction="Left", fps=1, min_serve_time=3):
        self.next_track_id = 0
        self.tracks = OrderedDict()
        self.max_lost = max_lost
        self.frame_count = 0
        self.fps = fps
        self.logs = {}
        self.exit_point = exit_point
        self.exit_direction = exit_direction
        self.tracker_output_format = tracker_output_format
        self.cummulative_serve_time = 0
        self.no_of_vehicles = 0
        self.min_serve_time = min_serve_time

    def _add_track(self, frame_id, bbox, detection_confidence, class_id, **kwargs):
        """
        Add a newly detected object to the queue.

        Parameters
        ----------
        frame_id : int
            Camera frame id.
        bbox : numpy.ndarray
            Bounding box pixel coordinates as (xmin, ymin, xmax, ymax) of the track.
        detection_confidence : float
            Detection confidence of the object (probability).
        class_id : Object
            Class label id.
        kwargs : dict
            Additional key word arguments.
        """

        self.tracks[self.next_track_id] = Track(
            self.next_track_id, frame_id, bbox, detection_confidence, class_id=class_id,
            data_output_format=self.tracker_output_format,
            **kwargs
        )
        self.next_track_id += 1

    def _remove_track(self, track_id):
        """
        Remove tracker data after object is lost.

        Parameters
        ----------
        track_id : int
                    track_id of the track lost while tracking
        """

        del self.tracks[track_id]

    def _update_track(self, track_id, frame_id, bbox, detection_confidence, class_id, lost=0, iou_score=0., **kwargs):
        """
        Update track state.

        Parameters
        ----------
        track_id : int
            ID of the track.
        frame_id : int
            Frame count.
        bbox : numpy.ndarray or list
            Bounding box coordinates as (xmin, ymin, width, height)
        detection_confidence : float
            Detection confidence (aka detection probability).
        class_id : int
            ID of the class (aka label) of the object being tracked.
        lost : int
            Number of frames the object was lost while tracking.
        iou_score : float
            Intersection over union.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------

        """

        self.tracks[track_id].update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs
        )

    @staticmethod
    def _get_tracks(tracks):
        """
        Output the information of tracks.

        Parameters
        ----------
        tracks : OrderedDict
            Tracks dictionary with (key, value) as (track_id, corresponding `Track` objects).

        Returns
        -------
        outputs : list
            List of tracks being currently tracked by the tracker.
        """

        outputs = []
        for trackid, track in tracks.items():
            if not track.lost:
                outputs.append(track.output())
        return outputs

    @staticmethod
    def preprocess_input(bboxes, class_ids, detection_scores):
        """
        Preprocess the input data.

        Parameters
        ----------
        bboxes : list or numpy.ndarray
            Array of bounding boxes with each bbox as a tuple containing `(xmin, ymin, width, height)`.
        class_ids : list or numpy.ndarray
            Array of Class ID or label ID.
        detection_scores : list or numpy.ndarray
            Array of detection scores (aka. detection probabilities).

        Returns
        -------
        detections : list[Tuple]
            Data for detections as list of tuples containing `(bbox, class_id, detection_score)`.
        """

        new_bboxes = np.array(bboxes, dtype='int')
        new_class_ids = np.array(class_ids, dtype='int')
        new_detection_scores = np.array(detection_scores)

        new_detections = list(zip(new_bboxes, new_class_ids, new_detection_scores))
        return new_detections

    
    def delete_track(self, old_id):
        """
        This function deletes the tracks of the vehicles which have surely crossed the exit point.
        This should enable us to Re-ID the car in few situations, where it gets the old ID, to a new ID.

        old_id : The track ID which is still in memory and should be deleted instead. 
        """    
        """
        self.tracks[self.next_track_id] = Track(
            self.next_track_id, frame_id, bbox, detection_confidence, class_id=class_id,
            data_output_format=self.tracker_output_format,
            **kwargs
        )
        self.next_track_id += 1

        self._add_track(self.frame_count, bbox, confidence, class_id=class_id)
        """
        if old_id in list(self.tracks.keys()):
            bbox, confidence, class_id = self.tracks[old_id].bbox, self.tracks[old_id].detection_confidence, self.tracks[old_id].class_id
            self._add_track(self.frame_count, bbox, confidence, class_id=class_id)
            self._remove_track(old_id)
        return




    def update_logs(self, outputs):
        """
        self.logs = {}
        self.exit_point = (320, 420)
        self.exit_direction = "Down"
        """
        if self.exit_direction == "Down":
            outputs = sorted(outputs, key = lambda x: x[3]+x[5], reverse=True)
        elif self.exit_direction == "Up":
            outputs = sorted(outputs, key = lambda x: x[3])
        elif self.exit_direction == "Left":
            outputs = sorted(outputs, key = lambda x: x[2])
        elif self.exit_direction == "Right":
            outputs = sorted(outputs, key = lambda x: x[2]+x[4], reverse=True)
        else:
            sys.exit("Exit Direction should be either 'Up','Down', 'Left' or 'Right'.")
        frame_no, id_, xmin, ymin, width, height, score = outputs[0][0:7]


        if (xmin <= self.exit_point[0] <= xmin+width) and (ymin <= self.exit_point[1] <= ymin+height):
            if id_ not in self.logs:
                if self.logs:
                    l = list(self.logs.values())[0]
                    in_lane_time = (l[1] - l[0] + 1)*self.fps
                    if in_lane_time > self.min_serve_time:
                        self.cummulative_serve_time = self.cummulative_serve_time + in_lane_time
                        self.no_of_vehicles += 1
                    
                    self.delete_track(list(self.logs.keys())[0])
                self.logs = {}   # Clearing older logs as new vehicle has been detected
                self.logs[id_]=[frame_no, frame_no]
                #print(self.logs)
            else:
                if frame_no - self.logs[id_][1] > 1:    print("Missed detection in previous frames!!")
                self.logs[id_][1] = frame_no
                #print(self.logs)

        return 


            




    def update(self, bboxes, detection_scores, class_ids):
        """
        Update the tracker based on the new bounding boxes.

        Parameters
        ----------
        bboxes : numpy.ndarray or list
            List of bounding boxes detected in the current frame. Each element of the list represent
            coordinates of bounding box as tuple `(top-left-x, top-left-y, width, height)`.
        detection_scores: numpy.ndarray or list
            List of detection scores (probability) of each detected object.
        class_ids : numpy.ndarray or list
            List of class_ids (int) corresponding to labels of the detected object. Default is `None`.

        Returns
        -------
        outputs : list
            List of tracks being currently tracked by the tracker. Each track is represented by the tuple with elements
            `(frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`.
        """

        self.frame_count += 1

        if len(bboxes) == 0:
            lost_ids = list(self.tracks.keys())

            for track_id in lost_ids:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

            outputs = self._get_tracks(self.tracks)
            return outputs, self.logs

        detections = Tracker.preprocess_input(bboxes, class_ids, detection_scores)

        track_ids = list(self.tracks.keys())

        updated_tracks, updated_detections = [], []

        if len(track_ids):
            track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])
            detection_centroids = get_centroid(bboxes)

            centroid_distances = distance.cdist(track_centroids, detection_centroids)
            #print(str(self.frame_count) + " : " + str(np.amin(centroid_distances, axis=1)))
            track_indices = np.amin(centroid_distances, axis=1).argsort()

            for idx in track_indices:
                track_id = track_ids[idx]

                remaining_detections = [
                    (i, d) for (i, d) in enumerate(centroid_distances[idx, :]) if i not in updated_detections]

                if len(remaining_detections):
                    detection_idx, detection_distance = min(remaining_detections, key=lambda x: x[1])
                    bbox, class_id, confidence = detections[detection_idx]
                    self._update_track(track_id, self.frame_count, bbox, confidence, class_id=class_id)
                    updated_detections.append(detection_idx)
                    updated_tracks.append(track_id)

                if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                    self.tracks[track_id].lost += 1
                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)

        for i, (bbox, class_id, confidence) in enumerate(detections):
            if i not in updated_detections:
                self._add_track(self.frame_count, bbox, confidence, class_id=class_id)

        outputs = self._get_tracks(self.tracks)

        self.update_logs(outputs)

        return outputs, self.logs
