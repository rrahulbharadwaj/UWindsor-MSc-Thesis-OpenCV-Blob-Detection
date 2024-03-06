import cv2 as cv
import math
import numpy as np 
import os
import sys
import re
import json
from matplotlib import pyplot as plt

#-----------------------------------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------------------------------
def CalcBlockMeanVariance(Img,blockSide=11): # blockSide - the parameter (set greater for larger font on image)            
    I=np.float32(Img)/255.0
    Res=np.zeros( shape=(int(Img.shape[0]/blockSide),int(Img.shape[1]/blockSide)),dtype=np.float)
    
    for i in range(0,Img.shape[0]-blockSide,blockSide):           
        for j in range(0,Img.shape[1]-blockSide,blockSide):        
            patch=I[i:i+blockSide+1,j:j+blockSide+1]
            m,s=cv.meanStdDev(patch)
            if(s[0]>0.01): # Thresholding parameter (set smaller for lower contrast image)
                Res[int(i/blockSide),int(j/blockSide)]=m[0]
            else:            
                Res[int(i/blockSide),int(j/blockSide)]=0
                      
    smallImg=cv.resize(I,(Res.shape[1],Res.shape[0] ) )    
    _,inpaintmask=cv.threshold(Res,0.02,1.0,cv.THRESH_BINARY);    
    smallImg=np.uint8(smallImg*255)    
    cv.imshow("contrast matrix",smallImg)
    #cv.waitKey(0)
    inpaintmask=np.uint8(inpaintmask)
    cv.imshow("inpaint mask",inpaintmask)
    #cv.waitKey(0)
    inpainted=cv.inpaint(smallImg, inpaintmask, 5, cv.INPAINT_TELEA)    
    Res=cv.resize(inpainted,(Img.shape[1],Img.shape[0]))
    Res=np.float32(Res)/255
    cv.imshow("inpainted image",Res)
    #cv.waitKey(0)    
    return Res

#-----------------------------------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------------------------------
def binarize(gray):
    ## convert image to grayscale/gray
    Img=gray.copy() #cv.imread("F:\\ImagesForTest\\BookPage.JPG",0)
    res=CalcBlockMeanVariance(Img)
    res=1.0-res
    Img=np.float32(Img)/255
    res=Img+res
    cv.imshow("Img",Img);
    ## Binarize image
    _,res=cv.threshold(res,0.85,1,cv.THRESH_BINARY);
    #res=cv.resize(res,( int(res.shape[1]/2),int(res.shape[0]/2) ))
    cv.imwrite("result.jpg",res*255);
    cv.imshow("Edges",res)
    cv.waitKey(0)
    #sys.exit()
    
    return res


# ----------------------------------------
# Find contours and their centers
# Filter contours by area and squareness
# ----------------------------------------

def findContours(gray):
    gray=binarize(gray)
    gray=np.uint8(gray*255)
    edges = cv.Canny(gray, 150,200)     
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    filtered_contours=[]
    areas=[]
    for cnt in contours:
        area = cv.contourArea(cnt)
        arch = cv.arcLength(cnt, True)    
        
        if area==0 or arch==0:
            continue 

        ## Find the Circularity of the object detected
        squareness = 4 * 3.14 * area / pow(arch,2);        
        
        if (area>40) and (area < 1000) and (squareness>0.8):        
            areas.append(area)
            filtered_contours.append(cnt)    
    
    centers=[]
    for c in filtered_contours:
        # compute the center of the contour
        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append((cX,cY))
        # draw the contour and center of the shape on the image
        #cv.imshow=cv.circle(img, (cX, cY),2, (255, 255, 255), -1)
        cv.waitKey(0)
    centers=np.array(centers)
    return filtered_contours, centers
    
    
# ----------------------------------------
# Filter centers by distance to 
# of distances between centers, as we have
# a grid there should be maximums on histogram,
# showing internodes distances. We extract
# these distances and use to filter grid nodes.
# ----------------------------------------
def filterCenters(centers):
    n_centers=len(centers)
    # build distance matrix
    dist=np.zeros((n_centers,n_centers),np.int64)
    for i in range(n_centers):
        for j in range(n_centers):
            if i!=j:            
                diff=(centers[i]-centers[j])        
                dist[i,j]=math.sqrt(diff[0]**2+diff[1]**2)
            else:
                # diagonals are always zero
                dist[i,j]=1e7
                
    # we need nearest neighbour
    m_dist=np.min(dist,axis=1)

    # find the range of minimal distabces
    dmin=int(min(m_dist))
    dmax=int(max(m_dist))

    dists=np.array([m_dist],dtype=np.float32)
    dists=np.transpose(dists,(1,0))
    print(dmin)
    print(dmax)
    # compure histogram
    n_bins=20
    hist=cv.calcHist([dists], [0],None,[n_bins],[dmin,dmax])
    plt.plot(hist)
    plt.show()
    # find maximal bin
    max_pos = np.argmax(hist)
    # compute values of distances from hist bin index
    max_range_start=(dmax-dmin)/n_bins*(max_pos)+dmin
    max_range_end=(dmax-dmin)/n_bins*(max_pos+1)+dmin
    
    #plt.hist(img.ravel(),n_bins,[dmin,dmax]); plt.show()

    print(max_range_start)
    print(max_range_end)
    # filter centers by distabce
    filtered_centers=[]
    i=0
    for c in centers:
        if m_dist[i]>max_range_start*0.8 and m_dist[i]<max_range_end*2:
            filtered_centers.append(c)            
            #cv.circle(img, (int(c[0]),int(c[1])) ,4, (255, 0, 255), -1)
        i=i+1   
    
    filtered_centers=np.array(filtered_centers)
    return filtered_centers
# ----------------------------------------
# line intersection
# ----------------------------------------
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
# ----------------------------------------
# point to line distance
# ----------------------------------------
def linePointDistance(p1,p2,p3):
    p1=np.array(p1)
    p2=np.array(p2)
    p3=np.array(p3)
    d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)    
    return d
# ----------------------------------------
# Compute centers of masses for contours
# ----------------------------------------
def get_corners_from_contours(contours, corner_amount=8):
    """
    Finds four corners from a list of points on the goal
    epsilon - the minimum side length of the polygon generated by the corners

    Parameters:
        :param: `contours` - a numpy array of points (opencv contour) of the
                             points to get corners from
        :param: `corner_amount` - the number of corners to find
    """
    coefficient = .005
    iters=0
    
    while True:
        print(contours)
        epsilon = coefficient * cv.arcLength(contours, True)
        # epsilon =
        # print("epsilon:", epsilon)
        
        if(coefficient > 2 or coefficient < 0 or iters>1000):
            return None
        
        poly_approx = cv.approxPolyDP(contours, epsilon, True,8)
        
        hull = cv.convexHull(poly_approx)
        if len(hull) == corner_amount:
            return hull
        else:
            if len(hull) > corner_amount:
                coefficient += .001
            else:
                coefficient -= .001
        iters=iters+1
    cv.imshow(hull)
            
# ----------------------------------------
# Sort points in the list according to 
# distance to given point
# ----------------------------------------
def get_ordered_list(points, x, y):
    points.sort(key = lambda p: math.sqrt((p[0] - x)**2 + (p[1] - y)**2))
    return points

def processImage(img):
    # Add border for getting closed contours
    img = cv.copyMakeBorder(img,20,20,20,20, 0, (255,255,255))
    # we will work witg gray image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
    # find blobs and their centers of masses
    filtered_contours, centers = findContours(gray)
     
    
    if len(filtered_contours)<28:
        print('Not enough contours found')
        # cv.imwrite('out/'+f,img)
        return -1   
    # filter centers by distance
    filtered_centers=filterCenters(centers)
    if len(filtered_centers)<28: # 4+4+10+10
        print('Not enough centers found')
        # cv.imwrite('out/'+f,img)
        return -1

    # find border points
    hull = cv.convexHull(filtered_centers)
    if hull is None or len(hull)<3:
        print('Hull is empty')
        # cv.imwrite('out/'+f,img)
        return -1
    
    # find 8 cornter points
    hull=get_corners_from_contours(hull)
    if hull is None or len(hull)<3:
        print('Hull is empty')
        # cv.imwrite('out/'+f,img)
        return -1

    # draw hull
    imCopy = img.copy()
    conCopy = cv.drawContours(imCopy, [hull], 0, (255,0,0), 2)
    cv.imshow = ('contours',conCopy)
    
    
    # draw corners
    for c in hull:
     img = cv.circle(img, (int(c[0][0]),int(c[0][1])) ,10, (0, 0, 255), -1)
     cv.imshow = ('contours',img)
     cv.waitKey(0)

    # Search centers, belonging to borders.
    # Remove short sides (2 centers).
    # Remaine only sides with 4 and 10 centers.
    b=[] # list with lists of centers belonging each edge of hull
    # colors to distinct sides
    colors=[(255,0,0),(255,255,0),(255,0,255), (0,255,0), (0,0,255),(0,255,255),(255,255,255),(255,127,255)]
    # Hull with removed second points for each side.
    # Need to sort centers according distance to edge start points
    new_hull=[]
    for i in range(len(hull)-1):    
        p1=hull[i]
        p2=hull[i+1]
        bb=[]
        for c in filtered_centers:
            d=linePointDistance(p1,p2,c)
            # if center is closer than 5 pixels - it belongs the edge
            if d<5 :            
                bb.append(c)
        # we need only edges with 4 and 10 centers on it
        if len(bb)>2:        
            b.append(bb)
            new_hull.append(p1)
    # And close the loop
    p1=hull[len(hull)-1]
    p2=hull[0]
    bb=[]
    
    for c in filtered_centers:
        d=linePointDistance(p1,p2,c)
        if d<5 :        
            bb.append(c)
    if len(bb)>2:        
        b.append(bb)
        new_hull.append(p1)

    if len(b)!=4:
        print('b!=4')
        # cv.imwrite('out/'+f,img)
        return -1
    
    if len(new_hull)!=4:
        print('new_hull!=4')
        # cv.imwrite('out/'+f,img)
        return -1
        
    # Now we need to make the order in edge sequence.
    
    # Search for edges with 10 centers
    first_long=-1
    l=0
    for bb in b:
        if len(bb)==10:
            first_long=l
            break
        l=l+1
    
    if first_long==-1:
        print('first long not detected')
        # cv.imwrite('out/'+f,img)
        return -1
     
    
    second_long=(first_long+2)%len(b)
    if len(b[second_long])!=10:
        print('b[second_long] not contains 10 centers')
        # cv.imwrite('out/'+f,img)
        return -1
    # We just need any 2 points on each long side
    bb_s=get_ordered_list(b[first_long], 0, 0)
    p1l=bb_s[0]
    p2l=bb_s[9]
    bb_s=get_ordered_list(b[second_long], 0, 0)
    p3l=bb_s[0]
    p4l=bb_s[9]       
    
    # do the same for short sides
    first_short=(first_long+1)%len(b)
    second_short=(first_short+2)%len(b)
    
    if(len(b[first_short])!=4):
        print('first short not correct')
        # cv.imwrite('out/'+f,img)
        return -1
    
    if(len(b[second_short])!=4):
        print('second short not correct')
        # cv.imwrite('out/'+f,img)
        return -1
    
    bb_s=get_ordered_list(b[first_short], 0, 0)    
    p1s=bb_s[0]
    p2s=bb_s[3]
    
    bb_s=get_ordered_list(b[second_short], 0, 0)
    p3s=bb_s[0]
    p4s=bb_s[3]
    
    # Find center
    c2=line_intersection((p1s,p4s), (p2s,p3s))
    cv.circle(img, (int(c2[0]),int(c2[1])) ,10, colors[1], -1)
    
    # Find sheet left side    
    d1=linePointDistance(p1l,p2l,c2)
    d2=linePointDistance(p3l,p4l,c2)
    
    cv.circle(img, (int(c2[0]),int(c2[1])) ,10, colors[l], -1)
    # make left edge first in list
    shift=0
    if(d1>d2):    
        shift=-first_long
        print (first_long)
    else:
        shift=-second_long
        print (second_long)
    
    # make shift
    b = [b[(i - shift) % len(b)] for i, x in enumerate(b)] 
    new_hull = [new_hull[(i - shift) % len(new_hull)] for i, x in enumerate(new_hull)] 
    
    # reorder centers along edges from first point to laset
    b_sorted=[]
    i=0
    
    for bb in b:
        p1=new_hull[(i)%len(new_hull)]
        bb_s=get_ordered_list(bb, p1[0][0], p1[0][1])
        b_sorted.append(bb_s)
        i=i+1

    b=b_sorted
    
    # font 
    font = cv.FONT_HERSHEY_SIMPLEX 
    # fontScale 
    fontScale = 0.5
    # Blue color in BGR 
    color = (255, 0, 0) 
    # Line thickness of 2 px 
    thickness = 1
    
    # Plot edges centers to check correctness
    l=0
    for bb in b:
        i=0
        for c in bb:
            cv.circle(img, (int(c[0]),int(c[1])) ,10, colors[l], -1)
            cv.putText(img, str(i), (int(c[0]),int(c[1])), font, fontScale, colors[6], thickness, cv.LINE_AA)    
            i=i+1
        l=l+1
    
    # compute grid lines
    lin_h=[]
    lin_v=[]
    for i in range(10):
        #cv.line(img,(int(b[0][i][0]),int(b[0][i][1])),(int(b[2][i][0]),int(b[2][i][1])),(255,255,255),1)
        lin_h.append( ( (int(b[0][i][0]),int(b[0][i][1])),(int(b[2][9-i][0]),int(b[2][9-i][1])) ) )
    for i in range(4):
        #cv.line(img,(int(b[1][i][0]),int(b[1][i][1])),(int(b[3][i][0]),int(b[3][i][1])),(255,255,255),1)
        lin_v.append( ( (int(b[1][i][0]),int(b[1][i][1])),(int(b[3][3-i][0]),int(b[3][3-i][1])) ) )
    
    
    # compute cells centers and fill the names
    letters=['A','B','C','D']
    cell_centers=[]
    cell_names=[]
    letter=0
    for lv in lin_v:
        ind = 10
        for lh in lin_h:        
            c=line_intersection(lh,lv)
            cell_name=letters[letter]+str(ind)
            cv.putText(img, cell_name, (int(c[0]),int(c[1])), font,  
                                fontScale, color, thickness, cv.LINE_AA)         
            cv.circle(img, (int(c[0]),int(c[1])) ,10, colors[3], 1)
            cell_centers.append(c)        
            cell_names.append(cell_name)
            ind=ind-1
        letter=letter+1
    
    # find centers close to cell centers
    # and print their names
    print("----------------------")
    print("checked")
    print("----------------------")
    i=0
    for c in centers:
        i=0
        for cell in cell_centers:
            d=np.linalg.norm(c-cell)
            if d<5:
                # print(cell_names[i])
                temp = re.compile("([a-zA-Z]+)([0-9]+)") 
                res = temp.match(str(cell_names[i])).groups()

                dict_of_answers[int(res[1])] = res[0]

            i=i+1    
        
    return 0

# ----------------------------------------
# MAIN
# ----------------------------------------
if __name__ == '__main__':

    if not os.path.exists('./output_file'):
        os.makedirs('./output_file')

    dict_of_answers = {}
    default_sol = 'Not Marked'
    for i in range(1,11,1):
        dict_of_answers[i]=default_sol

    total=0
    sum_err=0
    # for f in files:        
    img = cv.imread("D:/university of Windsor/THESIS/For Rahul/scanner/WIN_20200512_14_29_03_Pro.jpg ")  #D:/university of Windsor/THESIS/For Rahul/scanner/WIN_20200420_08_22_38_Pro.jpg
    if img is None:
        print('Image is empty')    

    res=processImage(img)
    ## Pyramid of Images to rescale & upscaling 
    if res==-1:
        res=0
        #img=src_img.copy()
        height, width = img.shape[:2]
        img_tmp = cv.resize(img, (int(0.8*width),int(0.8*height)), interpolation = cv.INTER_CUBIC)            
        res=processImage(img_tmp)

    if res==-1:
        res=0
        #img=src_img.copy()
        height, width = img.shape[:2]
        img_tmp = cv.resize(img, (int(1.5*width),int(1.5*height)), interpolation = cv.INTER_CUBIC)            
        res=processImage(img_tmp)
    
    if res==-1:
        res=0
        #img=src_img.copy()
        height, width = img.shape[:2]
        img_tmp = cv.resize(img, (int(2*width),int(2*height)), interpolation = cv.INTER_CUBIC)            
        res=processImage(img_tmp)
    
    if res==-1:    
        print('Bad Input Image!')
    else:    
        print(dict_of_answers)
        with open('./output_file/result.json', 'w') as fp:
            json.dump(dict_of_answers, fp) 
