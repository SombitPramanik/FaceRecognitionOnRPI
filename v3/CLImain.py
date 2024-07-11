import face_recognition as fr
import cv2
import os
import numpy as np

def Recognizer(AllPersonName, AllEncodingData):
    vCap = cv2.VideoCapture(0)
    while True:
        _, Frame = vCap.read()
        RGBFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite("tst.jpg", Frame)
        CapturedFaceLocations = fr.face_locations(RGBFrame)
        # print("captured Face Location Data : ",CapturedFaceLocations)
        AllCapturedFaceEncodings = fr.face_encodings(RGBFrame, CapturedFaceLocations)
        # print("Captured Frame Data : ",AllCapturedFaceEncodings)

        for EachCapturedFaceEncodingData in AllCapturedFaceEncodings:
            matches = fr.compare_faces(AllEncodingData, EachCapturedFaceEncodingData)
            PersonName = ""
            FaceDistance = fr.face_distance(AllEncodingData, EachCapturedFaceEncodingData)
            BestMatchIndex = np.argmin(FaceDistance)
            if matches[BestMatchIndex]:
                PersonName = AllPersonName[BestMatchIndex]
                print(PersonName)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vCap.release()
    cv2.destroyAllWindows()

def ImageProcessor(ImagePath):
    ImageData = fr.load_image_file(ImagePath)
    ImageEncoding = fr.face_encodings(ImageData)[0]
    return ImageEncoding

def main():
    PhotoDirectory = "./photos"
    AllPhotoName = os.listdir(PhotoDirectory)
    AllPhotoPath = []
    AllPhotoEncodingData = []
    AllPersonName = []
    
    for EachName in AllPhotoName:
        AllPhotoPath.append(os.path.join(PhotoDirectory, EachName))
        AllPersonName.append(EachName.split(".")[0])
    
    for EachPath in AllPhotoPath:
        AllPhotoEncodingData.append(ImageProcessor(EachPath))

    DetectedPersonName = Recognizer(AllPersonName, AllPhotoEncodingData)

    print(AllPersonName)
    print(DetectedPersonName)


if __name__ == "__main__":
    main()
