from dataclasses import dataclass, field

import numpy as np

from fusion.tracklets import Group

GATE_METERS = 4.0
MAX_MISSED_FRAMES = 3


@dataclass
class Track:
    track_id: int
    group: Group
    position: np.ndarray
    missed: int = 0


@dataclass
class Tracker:
    tracks: list[Track] = field(default_factory=list)
    next_id: int = 0

    def update(self, positions: list[np.ndarray], groups: list[Group]) -> list[int]:
        assigned: list[int] = [-1] * len(positions)
        pairs = sorted(
            (float(np.linalg.norm(track.position[[0, 2]] - positions[i][[0, 2]])), t, i)
            for t, track in enumerate(self.tracks)
            for i in range(len(positions))
            if track.group is groups[i]
        )
        used_tracks: set[int] = set()
        for distance, t, i in pairs:
            if distance > GATE_METERS or t in used_tracks or assigned[i] != -1:
                continue
            used_tracks.add(t)
            assigned[i] = self.tracks[t].track_id
            self.tracks[t].position = positions[i]
            self.tracks[t].missed = 0
        for t, track in enumerate(self.tracks):
            if t not in used_tracks:
                track.missed += 1
        for i, position in enumerate(positions):
            if assigned[i] == -1:
                self.tracks.append(Track(self.next_id, groups[i], position))
                assigned[i] = self.next_id
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED_FRAMES]
        return assigned
