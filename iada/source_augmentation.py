"""Suppress only vertical reflection while retaining its original random draw."""
import copy
import albumentations as A


class NoVerticalFlip(A.VerticalFlip):
    def apply(self, img, **params):
        return img


def without_vertical_application(reference):
    result = copy.deepcopy(reference)
    locations = [i for i, op in enumerate(result.transforms) if type(op) is A.VerticalFlip]
    assert len(locations) == 1
    index = locations[0]
    assert result.transforms[index].p == .2
    result.transforms[index] = NoVerticalFlip(p=result.transforms[index].p)
    return result
