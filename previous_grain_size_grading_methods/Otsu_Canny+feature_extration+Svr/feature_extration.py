# gsize.py
# Measure grain size based on the Heyn intercept method
import sys
import os
import shutil
import numpy as np
from PIL import Image, ImageDraw
import random


## constant ##
DIRNAME = "output"
RESULT_FILE = DIRNAME + "/result.dat"
CUTOFF = 2 * np.sqrt(2) + 0.001     # cutting distance for grouping boundary
L_CONST = 1.13                      # D = L_CONST * l_ave 


## function
def usage():
    print("Usage: ")
    print(" python getgrainsize.py -f <filename>  -l <l:int> -c -s <x:int> <y:int> -n <n:int>")
    print("  -f <filename>: filename of image (mandatory)")
    print("  -l <l:int>: number of lines to measure grain size (mandatory)")
    print("  -c: (optional) Crop mode. Images are randomly cropped from the original one. -s and -n are mandatory")
    print("  -s <x:int> <y:int>: crop size of x and y axis (pixel)")
    print("  -n <n:int>: number of crop images")

def argumenterror():
    print("Arguments Error.")
    usage()
    sys.exit(1)


def output(text):
    print(text)
    with open(RESULT_FILE, "a") as rf:
        rf.write(text + "\n")


def cropimage(filename, xsize, ysize, cropnum):

    # read image file
    im_ori = Image.open(filename)
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)

    width, height = im.size

    range_x = width - xsize
    range_y = height - ysize

    # crop image randomly
    output("\n-- Crop images randomly --")
    croppedlist = []
    for i in range(cropnum):
        left = random.randrange(range_x)
        upper = random.randrange(range_y)
        right = left + xsize
        lower = upper + ysize

        im_crop = im_ori.crop((left, upper, right, lower))
        outfilename = DIRNAME + '/{:03d}.tif'.format(i)
        im_crop.save(outfilename)
        output('No.{0:03d} crop({1:4d}  {2:4d}  {3:4d}  {4:4d}) -> {5}'.format(i, left, upper, right, lower, outfilename))
        croppedlist.append(outfilename)

        draw.rectangle((left, upper, right, lower), outline=(255, 0, 0))
        draw.text((left, upper), "No.{0:03d}".format(i), fill=(255, 0, 0))

    im.save(filename)
    return croppedlist


# measure grains size
def grainsize(croppedlist, linenum):

    output("\n-- Measure Grain Size --")

    for f in croppedlist:
        im = Image.open(f)
        width, height = im.size

        output(f)

        # value of pixels
        im_pixels = np.array([[im.getpixel((i, j)) for j in range(height)] for i in range(width)])

        draw = ImageDraw.Draw(im)

        # draw line
        # upper left is (0, 0)
        sum_d = 0.0
        for l in range(linenum):

            start = np.zeros(2, dtype=int)
            end = np.zeros(2, dtype=int)
            linelength = 0.0

            while True:
                # gradient
                grad = np.tan(random.uniform(-np.pi/2, np.pi/2))

                # point
                x = random.randrange(width)
                y = random.randrange(height)

                # y = ax + b
                b = y -grad * x
                if b < 0:
                    start[0] = -b / grad
                    start[1] = 0
                elif 0 <= b <= height:
                    start[0] = 0
                    start[1] = b
                elif height < b:
                    start[0] = (height - b) / grad
                    start[1] = height

                y_width = grad * width + b
                if y_width < 0:
                    end[0] = - b / grad
                    end[1] = 0
                elif 0 <= y_width <= height:
                    end[0] = width
                    end[1] = y_width
                elif height < y_width:
                    end[0] = (height - b) / grad
                    end[1] = height

                # check length (short line is not good)
                length2 = (start[0] - end[0])**2 + (start[1] - end[1])**2
                if length2 > width**2 and length2 > height**2:
                    #print(x, y, grad, start, end)
                    linelength = np.sqrt(length2)
                    break

            # find cross points
            # draw line
            im2 = Image.new('RGB', (width, height))
            draw2 = ImageDraw.Draw(im2)
            draw2.line(((start[0], start[1]), (end[0], end[1])), fill=(255, 0, 0))
            im2_pixels = np.array([[im2.getpixel((i, j)) for j in range(height)] for i in range(width)])
            for j in range(height):
                for i in range(width):
                    r, g, b = im2_pixels[i][j] + im_pixels[i][j]
                    im2.putpixel((i, j), (r, g, b))
            #im2.save(f+str(l)+".tif")

            # pick up pixels on grain boundary
            pixelsonGB = []
            im2_pixels = np.array([[im2.getpixel((i, j)) for j in range(height)] for i in range(width)])
            for j in range(height):
                for i in range(width):
                    r, g, b = im2_pixels[i][j]
                    if r == 255 and g == 0 and b == 0:
                        pixelsonGB.append((i, j))
            # print(pixelsonGB)

            # grouping neighbor pixels (distance <= CUTOFF)
            gbs = []
            p_coord = []
            i = 0
            for coord in pixelsonGB:
                if i == 0:
                    gbs.append([coord])

                else:
                    distance = np.sqrt((coord[0] - p_coord[0])**2 + (coord[1] - p_coord[1])**2)
                    if distance <= CUTOFF:
                        gbs[-1].append(coord)
                    else:
                        gbs.append([coord])

                p_coord = [coord[0], coord[1]]
                i += 1
            #print(gbs)

            # calc grain size
            d_grain = L_CONST * linelength / len(gbs)
            sum_d += d_grain
            output(" D_" + str(l) + " = " + str(d_grain) + " [px]")

            # draw line and G.B.
            draw.line(((start[0], start[1]), (end[0], end[1])), fill=(255, 0, 0))
            draw.text((start[0] + 0.3 * (end[0] - start[0]), start[1] + 0.3 * (end[1] - start[1])), str(l), fill=(255, 0, 0))
            for gb in gbs:
                upperleft = [width, height]
                lowerright = [0, 0]
                for coord in gb:
                    if coord[0] < upperleft[0]:
                        upperleft[0] = coord[0]
                    if coord[1] < upperleft[1]:
                        upperleft[1] = coord[1]
                    if lowerright[0] < coord[0]:
                        lowerright[0] = coord[0]
                    if lowerright[1] < coord[1]:
                        lowerright[1] = coord[1]

                draw.ellipse((upperleft[0]-1, upperleft[1]-1, lowerright[0]+1, lowerright[1]+1), outline=(0, 0, 255))

        im.save(f)
        ave_d = sum_d / linenum
        output(" D_ave = " + str(ave_d) + " [px]\n")


##
## main
##
if __name__ == "__main__":

    # check commandline arguments
    args = sys.argv

    # read arguments
    filename = '51276.jpg'
    xsize = 300
    ysize = 300
    cropnum = 3 
    isCrop = True
    linenum = 8
    try:
        for i in range(len(args)):
            if args[i] == "-f":
                filename = args[i+1]
            elif args[i] == "-c":
                isCrop = True
            elif args[i] == "-s":
                xsize = int(args[i+1])
                ysize = int(args[i+2])
            elif args[i] == "-n":
                cropnum = int(args[i+1])
            elif args[i] == "-l":
                linenum = int(args[i+1])
            elif args[i] == "-h":
                usage()
                sys.exit(0)
    except Exception as e:
        print(e)
        argumenterror()

    # check file existence
    if not filename or not linenum:
        argumenterror()
    else:
        if not os.path.isfile(filename):
            print(filename + " is not exist or not file.")
            sys.exit(1)

    # check crop parameter
    if isCrop:
        if not isinstance(xsize, int) or not isinstance(ysize, int) or not isinstance(cropnum, int):
            argumenterror()

    # mkdir
    os.makedirs(DIRNAME, exist_ok=True)
    rf = open(RESULT_FILE, "w")
    rf.close()

    # copy original file
    filename2 = DIRNAME + "/" + filename
    shutil.copyfile(filename, filename2)

    # file size
    im = Image.open(filename2)
    width, height = im.size

    # print
    output("-- INPUT DATA --")
    output("Filename: " + filename + " (" + str(width) + ", " + str(height) + ")")
    if isCrop:
        output("Crop Size: (" + str(xsize) + ", " + str(ysize) + ")")
        output("Number of crop images: " + str(cropnum))

    # crop images
    imglist = [filename2]
    if (isCrop):
        imglist = cropimage(filename2, xsize, ysize, cropnum)

    # analize grain size
    grainsize(imglist, linenum)


