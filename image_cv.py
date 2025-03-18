
import cv2 as cv2



class CvImages(object):

    def __init__(self):
        self.printlog = util_log.WR_Logging()
   
    def flipBGRtoHSV(self, inputmat):
        self.printlog.log_info("_to_HSV")
        return cv2.cvtColor(inputmat, cv2.COLOR_BGR2HSV_FULL)

    def flipHSVtoBGR(self, inputmat):
        self.printlog.log_info("_to_BGR")
        return cv2.cvtColor(inputmat, cv2.COLOR_HSV2BGR_FULL)    

    def assertInputColourSpace(self, image, space):
        if space == "HSV":
            return self.flipBGRtoHSV(image)
        return image        

    def assertOutputColourSpace(self, image, space):
        if space == "HSV":
            return self.flipHSVtoBGR(image)
        return image
      