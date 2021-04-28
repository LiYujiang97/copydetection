class AugmentMatch:
    def __init__(self, match, scale_ratio, rotate_angle, key_point, key_point_target):
        self._scale_ratio = scale_ratio
        self._rotate_angle = rotate_angle
        self._match = match
        self._key_point = key_point
        self._key_point_target = key_point_target

    @property
    def scale_ratio(self):
        return self._scale_ratio

    @property
    def match(self):
        return self._match

    @property
    def rotate_angle(self):
        return self._rotate_angle

    @property
    def key_point(self):
        return self._key_point

    @property
    def key_point_target(self):
        return self._key_point_target

    @staticmethod
    def get_match_array_by_augment_array(augment_match_array):
        if len(augment_match_array) < 1:
            return []
        match_array = []
        for augment_match in augment_match_array:
            match_array.append(augment_match.match)
        return match_array

    @staticmethod
    def get_key_point_info(kp):
        angle = kp.angle
        point_x = kp.pt[0]
        point_y = kp.pt[1]
        size = kp.size
        return angle, point_x, point_y, size

    # 返回keyPoint到（0，0）的距离
    def get_key_point_distance(self):
        angle, point_x, point_y, size = self.get_key_point_info(self._key_point)
        return pow(pow(point_x, 2) + pow(point_y, 2), 0.5)
