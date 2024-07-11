import os
import face_recognition as fr
import cv2
import pickle

def EncodingsExtractor(ImagePath):
    # Create a Image object that will be the gray image of the person
    Image = cv2.imread(ImagePath)
    # we use RGB type of image, but the model is also be able to take input the Gray Images
    RGBImage = cv2.cvtColor(Image,cv2.COLOR_BGR2RGB)
    FaceLocation = fr.face_locations(RGBImage,model="hog")
    FaceEncoding = fr.face_encodings(RGBImage,FaceLocation)

    return FaceEncoding




def main():
    # let assume that all image files are saved like this :
    # {personName}.{jpeg,png,jpg}
    # so if we list the Directory then will able to have this relative path
    # for person 1 :
    # Database/personName.jpeg
    # now to have to send this path to the actual function that gives us the face encoding data

    AllImagePaths = []
    AllNames = []
    NameANDPath = {}

    ImagePaths = os.listdir("Database")
    # print(type(ImagePaths))
    for i in range (len(ImagePaths)):
        # print(ImagePaths[i])
        AllImagePaths.append(os.path.join("Database",ImagePaths[i]))
        AllNames.append(ImagePaths[i].split(".")[0])

    NameANDPath ={"Names":AllNames,"Paths":AllImagePaths}
    AllEncodings = []
    for EachImagePath in NameANDPath["Paths"]:
        AllEncodings.append(EncodingsExtractor(EachImagePath))

    TrainedData = {"Names":AllNames,"Encodings": AllEncodings}
    f = open("TrainedModel.pkl","wb")
    f.write(pickle.dumps(TrainedData))
    f.close()
    return None

if __name__ == "__main__":
    main()
    

