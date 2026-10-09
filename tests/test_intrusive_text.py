import cv2
import numpy as np

from movie_broll.intrusive_text import evaluate_frames


def _text(y=45, scale=1.2):
    image=np.full((180,320,3), 80,dtype=np.uint8)
    cv2.putText(image,'EL JARDIN DEL', (15,y), cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),3,cv2.LINE_AA)
    return image


def test_large_persistent_overlay_is_rejected():
    assert evaluate_frames([_text() for _ in range(4)], minimum_persistent_samples=3)['decision'] == 'REJECT'


def test_persistent_lower_third_subtitle_is_rejected():
    assert evaluate_frames([_text(160,.85) for _ in range(4)], minimum_persistent_samples=3)['decision'] == 'REJECT'


def test_small_incidental_or_one_sample_text_is_not_a_hard_reject():
    small=np.full((180,320,3),80,dtype=np.uint8)
    cv2.putText(small,'Cafe',(12,30),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
    assert evaluate_frames([small for _ in range(4)], minimum_persistent_samples=3)['decision'] == 'PASS'
    assert evaluate_frames([_text(), small, small, small], minimum_persistent_samples=3)['decision'] == 'PASS'
