#!/usr/bin/python
import sys
import os
import dlib
import glob
import numpy as np

if len(sys.argv) != 4:
    print(
        "Call this program like this:\n"
        "   ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
        "You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    exit()

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
faces_folder_path = sys.argv[3]

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)



for f in glob.glob(os.path.join(faces_folder_path, "b1.jpg")):
   print("Processing file: {}".format(f))
   img = dlib.load_rgb_image(f)

   dets = detector(img, 1)
   print("Number of faces detected: {}".format(len(dets)))

   for k, d in enumerate(dets) :
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        shape = sp(img, d)

        face_descriptor = facerec.compute_face_descriptor(img, shape, 100)
        vector=np.array(face_descriptor)
        print(face_descriptor)
        




for f2 in glob.glob(os.path.join(faces_folder_path, "b11.jpg")):
   print("Processing file: {}".format(f2))
   img2 = dlib.load_rgb_image(f2)

   dets2 = detector(img2, 1)
   print("Number of faces detected: {}".format(len(dets2)))

   for k2, d2 in enumerate(dets2) :
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k2, d2.left(), d2.top(), d2.right(), d2.bottom()))
        shape2 = sp(img2, d2)

        face_descriptor2 = facerec.compute_face_descriptor(img2, shape2, 100)
        vector2=np.array(face_descriptor2)
        print(face_descriptor2)
        
dist=(np.sqrt((sum(vector)-sum(vector2))**2))
if(dist<=0.6):
   print("Image is same. ")
else:
   print("Not Same ")

dlib.hit_enter_to_continue()

