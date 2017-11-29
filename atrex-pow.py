import Tkinter, tkFileDialog
import os, fnmatch
from tkSimpleDialog import *
from PIL import Image
import numpy
import fabio
import math 
import time
from pyFAI.geometry import Geometry
import sys
import Tkinter, tkFileDialog
import os, fnmatch
from numpy import sin
from numpy import cos
from numpy import arctan
from math import  pi
from math import sqrt
#numpy.set_printoptions(threshold='nan')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import style
from matplotlib.pyplot import grid

global directoryname, diffimg0, file, data_images_summed, originalpic
global thetamap,  twothetaindeg

def directorylocator():
    global diffimg0, originalpic
    diffimg0 = tkFileDialog.askopenfilename()
    originalpic=diffimg0
    global directoryname
    directoryname=os.path.dirname(unicode(diffimg0))

def addimages():#adds or averages .tif's and saves them as a .tif, .edf or .txt files
    global directoryname, diffimg0, data_images_summed
    start_time = time.time()
    xdimension_pixelframe = Entry.get(xdimension)
    ydimension_pixelframe = Entry.get(ydimension)
    pixelframe_area = xdimension_pixelframe * ydimension_pixelframe
    blank = numpy.zeros(pixelframe_area, dtype=numpy.int32)
    blank.shape = (xdimension_pixelframe, ydimension_pixelframe)
    data_images_summed= numpy.array(blank,numpy.int32)
    counter=1

    for file in os.listdir(directoryname):
        if fnmatch.fnmatch(file, '*.tif'):
        #if fnmatch.fnmatch(file, '*.edf'):
            arrays=fabio.open(os.path.join(directoryname, file)).data
            data_images_summed += arrays
            counter += 1
        print counter - 1
        print 'Program took', time.time() - start_time, 'seconds to run.'
    denominator = counter - 1
    meanI = data_images_summed/denominator
    Avgtif = fabio.tifimage.tifimage(data=meanI.astype(numpy.int32))
    Avgtif.save('%s' %Entry.get(nameofsummedfile))
    #Avgtif = fabio.edfimage.edfimage(data=data_images_summed.astype(numpy.int32))
    #Avgtif.save('%s' %Entry.get(nameofsummedfile)) #if used put .edf instead of .tif when saving    
    #Summedtif = fabio.tifimage.tifimage(data=data_images_summed.astype(numpy.int32))
    #Summedtif.save('%s' %Entry.get(nameofsummedfile))
    #Summedtif = fabio.edfimage.edfimage(data=data_images_summed.astype(numpy.int32))
    #Summedtif.save('%s' %Entry.get(nameofsummedfile)) #if used put .edf instead of .tif when saving
    #numpy.savetxt('%s' %Entry.get(nameofsummedfile), data_images_summed) #if used put .txt when saving
    #numpy.savetxt('%s' %Entry.get(nameofsummedfile), meanI) #if used put .txt when saving
    print 'Done!'

def MinusDark():
    start_time = time.time()
    data = fabio.open('%s' %Entry.get(tifdata)).data #Puts name of tifdark file in %S
    dark = fabio.open('%s' %Entry.get(tifdark)).data #Puts name of tifdark file in %S
    #data = fabio.open('BiFeO3_0GPa_0_mean_150_Even.edf').data #Testing Reading in Data
    #dark = fabio.open('BiFeO3_0GPa_0_mean_150_Odd.edf').data #Testing Reading in Data
    dark_corrected_data = data - dark
    tif = fabio.tifimage.tifimage(data=dark_corrected_data.astype(numpy.int32))
    tif.save('%s' %Entry.get(savefilename))
    #tif = fabio.edfimage.edfimage(data=dark_corrected_data.astype(numpy.int32))
    #tif.save('%s' %Entry.get(savefilename))  #if used put .edf instead of .tif when saving    
    #numpy.savetxt('%s' %Entry.get(savefilename), dark_corrected_data) #if used put .txt when saving
    print 'Subtraction Completed'
    print 'Saved file(s) are located in C:\Users\MainDirectoryName\workspace\PythonProjectFolder'
    print "Program took", time.time() - start_time, "seconds to run"

def FIT2Dgeometry_pyFAI():
    start_time = time.time()
    g = pyFAI.geometry.Geometry()
    directDist = float(Entry.get(Samp2Det_Dis))
    centerX = float(Entry.get(XPCoor))
    centerY = float(Entry.get(YPCoor))
    tiltofDet = float(Entry.get(DetTilt))
    RotAngofTiltPlan = float(Entry.get(RotAng))
    pixelX = float(Entry.get(hpixel))
    pixelY = float(Entry.get(vpixel))

    tth = g.setFit2D(directDist, centerX, centerY,
                 tiltofDet, RotAngofTiltPlan,
                 pixelX, pixelY, splineFile=None)    
    
    #tth = g.setFit2D(directDist=276.6556, centerX=1050.182, centerY=1035.369,
    #             tilt=0.206068, tiltPlanRotation=-84.71832,
    #             pixelX=200, pixelY=200, splineFile=None)
    tth = g.getFit2D()
    #mapping = g.twoThetaArray((2048, 2048))
    xdimension_pixelframe = Entry.get(xdimension)
    ydimension_pixelframe = Entry.get(ydimension)
    mapping = numpy.rad2deg(g.twoThetaArray((xdimension_pixelframe, ydimension_pixelframe)))
    sortedindex = mapping.argsort(axis=None)
    xaxisval = mapping.flatten()[sortedindex]
    #mapping = g.twoThetaArray((numpy.array([2048]), numpy.array([2048])))
    #thetatif = fabio.tifimage.tifimage(data=mapping.astype(numpy.float32))
    #thetatif.save('%s' %Entry.get(nameofthetamapfile)) #Puts name of file in %s
    #thetatif.save('thetamap.tif') #How saving looks in the code without Entry.get method
    #thetatif = fabio.edfimage.edfimage(data=mapping.astype(numpy.float32))
    #thetatif.save('%s' %Entry.get(nameofthetamapfile)) #if used put .edf instead of .tif when saving    
    numpy.savetxt('%s' %Entry.get(nameofthetamapfile), mapping) #if used put .txt when saving
    print mapping
    print sortedindex
    print xaxisval
    print "Program took", time.time() - start_time, "seconds to run"
    print 'Theta mapping complete'
    #print tth
    #print mapping
    #print numpy.degrees(0.81546705)

def plot1D(): 
    start_time = time.time()
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator()
    ai.load('%s' %Entry.get(geometryponifile))  
    Intensitydata = fabio.open('%s' %Entry.get(intensityfile)).data
    Intensitydata = numpy.flipud(Intensitydata)
    #ai.load('BiFeO3FIT2DGeometry_Dioptas.poni')  
    #Intensitydata = fabio.open('data_minus_dark.tif').data
    x=[] 
    y=[] 
    xdimension_pixelframe = Entry.get(xdimension)
    ydimension_pixelframe = Entry.get(ydimension)
    x, y = ai.integrate1d(Intensitydata, npt=xdimension_pixelframe*ydimension_pixelframe, unit='2th_deg', polarization_factor=0.999, method = 'BBox')
    plt.plot(x, y)
    plt.xlabel('2-Theta')
    plt.ylabel('Intensity')
    print "Program took", time.time() - start_time, "seconds to run"
    plt.show() 

def plot2D():
    style.use('fivethirtyeight') 
    start_time = time.time()
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator()
    ai.load('%s' %Entry.get(geometryponifile))  
    Intensitydata = fabio.open('%s' %Entry.get(intensityfile)).data
    Intensitydata = numpy.flipud(Intensitydata)
    #ai.load('BiFeO3FIT2DGeometry_Dioptas.poni')  
    #Intensitydata = fabio.open('data_minus_dark.tif').data    
    x=[] 
    y=[]
    xdimension_pixelframe = Entry.get(xdimension) 
    #x, y = ai.integrate1d(Intensitydata, npt=2048*2048, unit='2th_deg', polarization_factor=0.999, method = 'BBox')
    #TwoDplot = ai.calcfrom1d(x, y, shape=(2048,2048), mask=None, dim1_unit='2th_deg', correctSolidAngle=None)
    #plt.imshow(TwoDplot)
    TwoDimage = ai.integrate2d(Intensitydata,npt_rad=xdimension_pixelframe, npt_azim=360, polarization_factor= 0.999, method ='BBox', unit='2th_deg')
    image_data=numpy.resize(TwoDimage,(xdimension_pixelframe,ydimension_pixelframe))
    #plt.plot(x, y)
    #TwoDimage = ai.integrate2d(Intensitydata,npt_rad=1024, npt_azim=360, polarization_factor= 0.999, method ='BBox', unit='2th_deg')
    plt.imshow(image_data)
    print "Program took", time.time() - start_time, "seconds to run"
    cbar = plt.colorbar()
    cbar.set_label('Intensity')
    plt.grid(False) 
    plt.show()
 
def BlackWhite2Dplot():
    #style.use('fivethirtyeight')
    style.use('grayscale')
    start_time = time.time()
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator()
    ai.load('%s' %Entry.get(geometryponifile))  
    Intensitydata = fabio.open('%s' %Entry.get(intensityfile)).data
    Intensitydata = numpy.flipud(Intensitydata)
    #ai.load('BiFeO3FIT2DGeometry_Dioptas.poni')  
    #Intensitydata = fabio.open('data_minus_dark.tif').data    
    x=[] 
    y=[]
    xdimension_pixelframe = Entry.get(xdimension)
    ydimension_pixelframe = Entry.get(ydimension) 
    x, y = ai.integrate1d(Intensitydata, npt=xdimension_pixelframe*ydimension_pixelframe, unit='2th_deg', polarization_factor=0.999, method = 'BBox')
    TwoDplot = ai.calcfrom1d(x, y, shape=(xdimension_pixelframe,ydimension_pixelframe), mask=None, dim1_unit='2th_deg', correctSolidAngle=None)
    plt.imshow(TwoDplot, cmap=cm.gray)
    print "Program took", time.time() - start_time, "seconds to run"
    cbar = plt.colorbar()
    cbar.set_label('Intensity')
    plt.grid(False) 
    plt.show()

global mask_data, thetamap, sortedpic_index, keepit_pix, keepit_inten, sortedtheta, sortedinten 
def StartUp(): 
    global mask_data, thetamap, sortedpic_index, keepit_pix, keepit_inten, sortedtheta, sortedinten 
    start_time = time.time()
    Intensitydata = fabio.open('%s' %Entry.get(Intensitydata_tif)).data 
    #Intensitydata = fabio.open('AvgDataMinusDark.tif').data
    Intensitydata = numpy.flipud(Intensitydata) 
    thetamap = fabio.open('%s' %Entry.get(TwoTheta_tif)).data
    #thetamap = fabio.open('Correct2Theta.tif').data
    xdimension_pixelframe = Entry.get(xdimension)
    ydimension_pixelframe = Entry.get(ydimension)     
    sorted_index = thetamap.argsort(axis=None)
    pic_index=numpy.arange(xdimension_pixelframe*ydimension_pixelframe) 
    sortedpic_index=pic_index[sorted_index]
    sortedtheta=numpy.rad2deg(thetamap.flatten()[sorted_index]) #changes outputs from radians to degrees     
    sortedinten=Intensitydata.flatten()[sorted_index] 
    print "Program took", time.time() - start_time, "seconds to run"
     
def eliminate(bin_intensity_array, sigma, mapped_pic_index1):
    start_time = time.time()
    global sigmanum, keepit_pix, keepit_inten 
    midpoint1=numpy.median(bin_intensity_array)
    lower_bound= float(midpoint1 - 0.5*sigmanum*sigma)
    upper_bound= float(midpoint1 + 0.5*sigmanum*sigma)
    index_location=numpy.where((bin_intensity_array>=lower_bound) & (bin_intensity_array<=upper_bound)) # produces good intensity LOCATIONS
    keepit_pix.extend(mapped_pic_index1[index_location])
    keepit_inten.extend(bin_intensity_array[index_location])
    print "Program took", time.time() - start_time, "seconds to run"

def freq_occurrence(Is,sigma1, midpoint1, mapped_pic_index1):     
    start_time = time.time()
    frequency=[]
    intensities=Is 
    last=int(numpy.size(Is)-1) #upper bound index
    sorted_bin=sorted(Is)
    lowerbound=sorted_bin[0]
    upperbound=sorted_bin[last] 
    #print 'upper and lower intensity bounds:', lowerbound, upperbound
    bin_intensity_array = []
    total = int(round(upperbound-lowerbound))
    counter2=1 # TO COUNT THRU THE WHOLE INTERVAL; 
    while counter2 < total: # counts number of times a particular intensity shows up
        intensity_value = float ((round(lowerbound)-2)+counter2) 
        lower_bound= float(intensity_value - .5) 
        upper_bound= float(intensity_value + .5) 
        index_location=numpy.where(( intensities>=lower_bound) & (intensities<=upper_bound)) #gives good intensity locations
        bin_size = numpy.size(index_location)
        frequency.append(bin_size) # adds to array containing the number of occurrences 
        bin_intensity_array.append(intensity_value) 
        counter2 = counter2 + 1
    frequency_statistics=eliminate(Is, sigma1, mapped_pic_index1)
    #plt.plot(bin_intensity_array, frequency)
    #plt.show()  
    print "Program took", time.time() - start_time, "seconds to run"

def bin_averaging():
    start_time = time.time() 
    global sortedtheta, keepit_pix, keepit_inten, sortedinten, lower_bound, sortedpic_index, upper_bound, sigmanum, trash, keepit
    keepit_pix=[]
    keepit_inten=[]
    trash=[]
    sigmanum1 = (Entry.get(sigmanum))
    sigmanum=float(sigmanum1)
    delta = float(Entry.get(delta_input))
    delta2theta= float(delta) # should be an angle 
    halfdelta=float(delta2theta/2)
    counter=1
    counter_total= int(numpy.amax(sortedtheta)/delta)+1
    index_location=[]
    bin_avg=[] # array containing bin average values (same order as theta)
    arrayindex=[] # array containing midpoint values
    while counter < counter_total:
        print counter
        midpoint = float (delta2theta*counter) 
        lower_bound= float(midpoint - halfdelta)
        upper_bound= float(midpoint + halfdelta)
        index_location=numpy.where((sortedtheta>=lower_bound) & (sortedtheta<=upper_bound)) #keeps theta values between bounds
        kept_pic_index=sortedpic_index[index_location] #pixel locations for theta values within bounds
        kept_bin=sortedinten[index_location]
        kept_theta=sortedtheta[index_location]         
        bin_sum=float(numpy.sum(kept_bin)) #sums values in bin
        bin_size = float (numpy.size(index_location))
        #print 'midpoint', midpoint
        if bin_size ==0:
            counter = counter+1
            bin_average= float(bin_sum/1)
            sigma=sqrt(bin_average) #reduces to n-sigma
            
        else:
            bin_average= float(bin_sum/bin_size)
            sigma=sqrt(bin_average) #reduces to n-sigma 
            freq_occurrence(kept_bin, sigma, midpoint, kept_pic_index) # at particular theta
            arrayindex.append(midpoint) # adds value to array midpoint values
            bin_avg.append(bin_average) # adds value to array bin average values
            counter = counter + 1
        #print 'counter1', counter 
#    print 'pixel locations:', keepit_pix
#    print 'pixel intensities:', keepit_inten
    print "Program took", time.time() - start_time, "seconds to run"

def maskmaker(): # zero is black white is one 
    start_time = time.time()
    global keepit_pix, keepit_inten
    xdimension_pixelframe = Entry.get(xdimension)
    ydimension_pixelframe = Entry.get(ydimension)      
    mask=numpy.zeros((xdimension_pixelframe*xdimension_pixelframe), dtype=int) 
    mask[keepit_pix] = keepit_inten #keepit_pix#=255 #255 makes the good intensity areas white
    #print 'mask', mask  
    mask_data=numpy.resize(mask,(xdimension_pixelframe,ydimension_pixelframe))
    masked_img=Image.fromarray(mask_data) 
    MaskedData = fabio.tifimage.tifimage(data = mask_data.astype(numpy.int32))   
    MaskedData.save('%s' %Entry.get(savemaskedData))   
    plt.imshow(masked_img)#,cmap='Greys')
    plt.show() 
    print "Program took", time.time() - start_time, "seconds to run"

def Plot1DMask():
    start_time = time.time() 
    openmaskedData = fabio.open('%s' %Entry.get(maskedData_tif)).data 
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator()
    ai.load('%s' %Entry.get(geometryponifile))  
    x=[] 
    y=[] 
    xdimension_pixelframe = Entry.get(xdimension)
    ydimension_pixelframe = Entry.get(ydimension)     
    x, y = ai.integrate1d(openmaskedData, npt=xdimension_pixelframe*ydimension_pixelframe, unit='2th_deg', polarization_factor=0.999, method = 'BBox')
    plt.xlabel('2-Theta')
    plt.ylabel('Intensity')
    print "Program took", time.time() - start_time, "seconds to run"
    plt.show()     


root = Tkinter.Tk()
root.wm_title("py321DIAS")

Button(root, text='Select Any File in the Directory', 
       command=directorylocator).grid(row=2, column=0, columnspan=1) 

l1 = Tkinter.Label(root, text="Input x-dimension of arrays")
xdimension = Tkinter.Entry(root)
l1.grid(row=4, column=0)
xdimension.grid(row=6, column=0)       

l1 = Tkinter.Label(root, text="Input y-dimension of arrays")
ydimension = Tkinter.Entry(root)
l1.grid(row=8, column=0)
ydimension.grid(row=10, column=0)    

l1 = Tkinter.Label(root, text="Input nameofsavefile.tif before adding images")
nameofsummedfile = Tkinter.Entry(root)
l1.grid(row=12, column=0)
nameofsummedfile.grid(row=14, column=0) 
   
Button(root, text='Add or Average Data or Dark Images', 
       command=addimages).grid(row=16, column=0, columnspan=1)  

l1 = Tkinter.Label(root, text="Input nameofdatafile.tif (or any file type that FabIO reads) before subtracting images.")
tifdata = Tkinter.Entry(root)
l1.grid(row=18, column=0)
tifdata.grid(row=20, column=0)

l2 = Tkinter.Label(root, text="Input nameofdarkfile.tif (or any file type that FabIO reads) before subtracting images.")
tifdark = Tkinter.Entry(root)
l2.grid(row=22, column=0)
tifdark.grid(row=24, column=0)

l3 = Tkinter.Label(root, text="Input 'nameofsavefile.tif' before subtracting images. Place '' around the name if you get an error.")
savefilename = Tkinter.Entry(root)
l3.grid(row=26, column=0)
savefilename.grid(row=28, column=0)

Button(root, text='Minus Dark Images from Data Images', 
       command=MinusDark).grid(row=30, column=0, columnspan=1) 

lab = Tkinter.Label(root, text="FIT2D Geometry")
lab.grid(row=2, column=1)

l1 = Tkinter.Label(root, text="Hieght of Pixels (microns)")
vpixel = Tkinter.Entry(root)
l1.grid(row=4, column=1)
vpixel.grid(row=6, column=1)

l2 = Tkinter.Label(root, text="Width of Pixels (microns)")
hpixel = Tkinter.Entry(root)
l2.grid(row=8, column=1)
hpixel.grid(row=10, column=1)

l3 = Tkinter.Label(root, text="Sample to Detector Distance (mm)")
Samp2Det_Dis = Tkinter.Entry(root)
l3.grid(row=12, column=1)
Samp2Det_Dis .grid(row=14, column=1)

l4 = Tkinter.Label(root, text="Wavelength (Angstroms)")
wavelength = Tkinter.Entry(root)
l4.grid(row=16, column=1)
wavelength.grid(row=18, column=1)

l6 = Tkinter.Label(root, text="X-Pixel Beam Center")
XPCoor = Tkinter.Entry(root)
l6.grid(row=20, column=1)
XPCoor.grid(row=22, column=1)

l7 = Tkinter.Label(root, text="Y-Pixel Beam Center")
YPCoor = Tkinter.Entry(root)
l7.grid(row=24, column=1)
YPCoor.grid(row=26, column=1)
l8 = Tkinter.Label(root, text="Rotation Angle of Tilting Plane (Degrees)")
RotAng = Tkinter.Entry(root)

l8.grid(row=28, column=1)
RotAng.grid(row=30, column=1)

l9 = Tkinter.Label(root, text="Detector Tilt (Degrees)")
DetTilt = Tkinter.Entry(root)
l9.grid(row=32, column=1)
DetTilt.grid(row=34, column=1)

l3 = Tkinter.Label(root, text="Input nameofsavedthetafile.tif before creating 2-theta plot.")
nameofthetamapfile = Tkinter.Entry(root)
l3.grid(row=36, column=1)
nameofthetamapfile.grid(row=38, column=1)

Button(root, text='Create 2-theta Plot', command=FIT2Dgeometry_pyFAI).grid(row=40, column=1, columnspan=1)

l1 = Tkinter.Label(root, text="Input nameofPONIfile.poni before creating plots")
geometryponifile = Tkinter.Entry(root)
l1.grid(row=42, column=1)
geometryponifile.grid(row=44, column=1)

l2 = Tkinter.Label(root, text="Input nameofintensitydata.tif (or any file type that Fabio reads) before creating plots")
intensityfile = Tkinter.Entry(root)
l2.grid(row=46, column=1)
intensityfile.grid(row=48, column=1)

Button(root, text='Create 1D plot', 
       command=plot1D).grid(row=50, column=1, columnspan=1)

Button(root, text='Create Colored 2D plot', 
       command=plot2D).grid(row=52, column=1, columnspan=1)    

Button(root, text='Create Black and White 2D plot', 
       command=BlackWhite2Dplot).grid(row=54, column=1, columnspan=1)

l11 = Tkinter.Label(root, text="Input nameofintensitydata.tif")
Intensitydata_tif = Tkinter.Entry(root)
l11.grid(row=2, column=3)
Intensitydata_tif.grid(row=4, column=3)

l12 = Tkinter.Label(root, text="Input nameofTwoThetaGeometry.tif")
TwoTheta_tif = Tkinter.Entry(root) 
l12.grid(row=6, column=3)
TwoTheta_tif.grid(row=8, column=3)

Button(root, text='Start Up', command=StartUp).grid(row=10, column=3, columnspan=1)

l13 = Tkinter.Label(root, text="Give Bin width in degrees")
delta_input = Tkinter.Entry(root)
l13.grid(row=12, column=3)
delta_input.grid(row=14, column=3)

l14 = Tkinter.Label(root, text="Number of Sigmas")
sigmanum = Tkinter.Entry(root)
l14.grid(row=16, column=3)
sigmanum.grid(row=18, column=3)

l15 = Tkinter.Label(root, text="Input savenameofmaskedData.tif before creating mask")
savemaskedData = Tkinter.Entry(root)
l15.grid(row=20, column=3)
savemaskedData.grid(row=22, column=3)

Button(root, text='Find Binned Averages and Do Statistics', command=bin_averaging).grid(row=24, column=3, columnspan=1)

Button(root, text='Make Mask', command=maskmaker).grid(row=26, column=3, columnspan=1)

l16 = Tkinter.Label(root, text="Input nameofmaskedData.tif before plotting mask")
maskedData_tif = Tkinter.Entry(root)
l16.grid(row=28, column=3)
maskedData_tif.grid(row=30, column=3)
 
l17 = Tkinter.Label(root, text="Input nameofPONIfile.poni before plotting mask")
geometryponifile = Tkinter.Entry(root)
l17.grid(row=32, column=3)
geometryponifile.grid(row=34, column=3)

Button(root, text='Create 1D Plot of Masked Data', command=Plot1DMask).grid(row=36, column=3, columnspan=1)

root.mainloop()
