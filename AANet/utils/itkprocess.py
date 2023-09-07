import SimpleITK as sitk

import numpy as np


def dicomseriesReader(pathDicom):
    """
    dicom series reader
    :param pathDicom:input dicom path
    :return:dicom image
    """
    reader = sitk.ImageSeriesReader()
    filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom)
    reader.SetFileNames(filenamesDICOM)
    imgOriginal = reader.Execute()
    return imgOriginal


def GetMaskImage(sitk_src, sitk_mask, replacevalue=0):
    """
    get mask image
    :param sitk_src:input image
    :param sitk_mask:input mask
    :param replacevalue:replacevalue of maks value equal 0
    :return:mask image
    """
    array_src = sitk.GetArrayFromImage(sitk_src)
    array_mask = sitk.GetArrayFromImage(sitk_mask)
    array_out = array_src.copy()
    array_out[array_mask == 0] = replacevalue
    outmask_sitk = sitk.GetImageFromArray(array_out)
    outmask_sitk.SetDirection(sitk_src.GetDirection())
    outmask_sitk.SetSpacing(sitk_src.GetSpacing())
    outmask_sitk.SetOrigin(sitk_src.GetOrigin())
    return outmask_sitk


def BinaryThreshold(sitk_src, lowervalue, uppervalue):
    """
    image binary threshold
    :param sitk_src:input image
    :param lowervalue:lower threshold value
    :param uppervalue:upper threshold value
    :return:threshold image
    """
    seg = sitk.BinaryThreshold(sitk_src, lowerThreshold=lowervalue, upperThreshold=uppervalue, insideValue=255,
                               outsideValue=0)
    return seg


def OtsuThreshold(sitk_src):
    """
    otsu threshold
    :param sitk_src:input image
    :return:threshold image
    """
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(255)
    seg = otsu_filter.Execute(sitk_src)
    return seg


def RegionGrowThreshold(input_sitk, seedList, lower=0., upper=255.):
    """
    region grow threshold
    :param input_sitk:input image
    :param seedList:seed points
    :param lower:lower value
    :param upper:upper value
    :return:threshold image
    """
    seg_sitk = sitk.ConnectedThreshold(image1=input_sitk,
                                       seedList=seedList,
                                       lower=float(lower),
                                       upper=float(upper), replaceValue=255)
    return seg_sitk


def GetLargestConnectedCompont(binarysitk_image):
    """
    save largest object
    :param sitk_maskimg:binary itk image
    :return: largest region binary image
    """
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 255
    outmask[labelmaskimage != maxlabel] = 0
    outmask_sitk = sitk.GetImageFromArray(outmask)
    outmask_sitk.SetDirection(binarysitk_image.GetDirection())
    outmask_sitk.SetSpacing(binarysitk_image.GetSpacing())
    outmask_sitk.SetOrigin(binarysitk_image.GetOrigin())
    return outmask_sitk


def RemoveSmallConnectedCompont(sitk_maskimg, rate=0.5):
    """
    remove small object
    :param sitk_maskimg:input binary image
    :param rate:size rate
    :return:binary image
    """
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    not_remove = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size > maxsize * rate:
            not_remove.append(l)
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage != maxlabel] = 0
    for i in range(len(not_remove)):
        outmask[labelmaskimage == not_remove[i]] = 255
    outmask_sitk = sitk.GetImageFromArray(outmask)
    outmask_sitk.SetDirection(sitk_maskimg.GetDirection())
    outmask_sitk.SetSpacing(sitk_maskimg.GetSpacing())
    outmask_sitk.SetOrigin(sitk_maskimg.GetOrigin())
    return outmask_sitk


def FloodFilled(sitk_image):
    """
    floodfilled operation
    :param sitk_image:input binary image
    :return:binary image
    """
    NbhC_filter = sitk.NeighborhoodConnectedImageFilter()
    NbhC_filter.SetLower(0)
    NbhC_filter.SetUpper(1)
    NbhC_filter.SetReplaceValue(255)
    NbhC_filter.SetRadius(0)
    x = sitk_image.GetSize()[0]
    y = sitk_image.GetSize()[1]
    z = sitk_image.GetSize()[2]
    seed0 = [0, 0, 0]
    NbhC_filter.AddSeed(seed0)
    seed1 = [x - 1, y - 1, z - 1]
    NbhC_filter.AddSeed(seed1)
    seed2 = [x - 1, 0, 0]
    NbhC_filter.AddSeed(seed2)
    seed3 = [x - 1, y - 1, 0]
    NbhC_filter.AddSeed(seed3)
    seed4 = [0, y - 1, 0]
    NbhC_filter.AddSeed(seed4)
    seed5 = [0, 0, z - 1]
    NbhC_filter.AddSeed(seed5)
    seed6 = [x - 1, 0, z - 1]
    NbhC_filter.AddSeed(seed6)
    seed7 = [0, y - 1, z - 1]
    NbhC_filter.AddSeed(seed7)
    image = NbhC_filter.Execute(sitk_image)
    return image


def FillHole(sitk_src):
    """
    fill binary region inside small holes
    :param sitk_src: input binary image
    :return: binary image
    """
    sitk_fillhole = sitk.BinaryFillhole(sitk_src)
    return sitk_fillhole


def MorphologicalOperation(sitk_maskimg, kernelsize, name='open'):
    """
    morphological operation
    :param sitk_maskimg:input binary image
    :param kernelsize:kernel zie
    :param name:operation name
    :return:binary image
    """
    if name == 'open':
        morphoimage = sitk.BinaryMorphologicalOpening(sitk_maskimg != 0, kernelsize)
        return morphoimage
    if name == 'close':
        morphoimage = sitk.BinaryMorphologicalClosing(sitk_maskimg != 0, kernelsize)
        return morphoimage
    if name == 'dilate':
        morphoimage = sitk.BinaryDilate(sitk_maskimg != 0, kernelsize)
        return morphoimage
    if name == 'erode':
        morphoimage = sitk.BinaryErode(sitk_maskimg != 0, kernelsize)
        return morphoimage


def lungSegment(sitk_src):
    # 1
    sitk_seg = BinaryThreshold(sitk_src, lowervalue=-300, uppervalue=2000)
    # sitk.WriteImage(sitk_seg, 'step1.mha')
    # 2
    sitk_floodfilled = FloodFilled(sitk_seg)
    # sitk.WriteImage(sitk_floodfilled, 'step2.mha')
    # 3
    sitk_xorop = sitk.XorImageFilter()
    sitk_mask1 = sitk_xorop.Execute(sitk_seg, sitk_floodfilled)
    sitk_notop = sitk.NotImageFilter()
    sitk_mask2 = sitk_notop.Execute(sitk_mask1)
    # sitk.WriteImage(sitk_mask2, 'step3.mha')
    # 4
    sitk_mask3 = FillHole(sitk_mask2)
    # sitk.WriteImage(sitk_mask3, 'step4.mha')
    # 5
    sitk_mask4 = RemoveSmallConnectedCompont(sitk_mask3, 0.2)
    # sitk.WriteImage(sitk_mask4, 'step5.mha')
    # 6 segtrachea
    lstSeeds = []
    seed1 = [259, 293, 98]
    seed2 = [222, 314, 75]
    seed3 = [282, 304, 75]
    lstSeeds.append(seed1)
    lstSeeds.append(seed2)
    lstSeeds.append(seed3)
    sitk_tracheamask = RegionGrowThreshold(sitk_src, lstSeeds, -1024, -880)
    # sitk.WriteImage(sitk_tracheamask, 'step6.mha')
    # 7 lung reduce trachea
    array_tracheamask = sitk.GetArrayFromImage(sitk_tracheamask)
    array_mask4 = sitk.GetArrayFromImage(sitk_mask4)
    array_mask4 = array_mask4 - array_tracheamask
    sitk_mask4 = sitk.GetImageFromArray(array_mask4)
    sitk_mask4.SetDirection(sitk_tracheamask.GetDirection())
    sitk_mask4.SetSpacing(sitk_tracheamask.GetSpacing())
    sitk_mask4.SetOrigin(sitk_tracheamask.GetOrigin())
    # sitk.WriteImage(sitk_mask4, 'step7.mha')
    # 8
    sitk_mask4 = MorphologicalOperation(sitk_mask4, kernelsize=3, name='open')
    sitk_mask5 = MorphologicalOperation(sitk_mask4, kernelsize=9, name='close')
    # sitk.WriteImage(sitk_mask5, 'step8.mha')
    # 9
    # sitk_lung = GetMaskImage(sitk_src, sitk_mask5, replacevalue=-1500)
    # sitk.WriteImage(sitk_lung, 'step9.mha')
    return sitk_mask5


def tracheaSegment(pathDicom):
    sitk_src = dicomseriesReader(pathDicom)
    lstSeeds = []
    seed1 = [259, 293, 98]
    seed2 = [222, 314, 75]
    seed3 = [282, 304, 75]
    lstSeeds.append(seed1)
    lstSeeds.append(seed2)
    lstSeeds.append(seed3)
    sitk_mask = RegionGrowThreshold(sitk_src, lstSeeds, -1024, -880)
    sitk.WriteImage(sitk_mask, 'tracheamask.mha')
    sitk_trachea = GetMaskImage(sitk_src, sitk_mask, replacevalue=-1500)
    return sitk_trachea


def skeletonSegment(sitk_src):
    # sitk_src = dicomseriesReader(pathDicom)
    # 1
    sitk_seg = BinaryThreshold(sitk_src, lowervalue=100, uppervalue=3000)
    # sitk.WriteImage(sitk_seg, 'step1.mha')
    # 2
    sitk_open = MorphologicalOperation(sitk_seg, kernelsize=2, name='open')
    sitk_open = GetLargestConnectedCompont(sitk_open)
    # sitk.WriteImage(sitk_open, 'step2.mha')
    # 3
    array_open = sitk.GetArrayFromImage(sitk_open)
    array_seg = sitk.GetArrayFromImage(sitk_seg)
    array_mask = array_seg - array_open
    sitk_mask = sitk.GetImageFromArray(array_mask)
    sitk_mask.SetDirection(sitk_seg.GetDirection())
    sitk_mask.SetSpacing(sitk_seg.GetSpacing())
    sitk_mask.SetOrigin(sitk_seg.GetOrigin())
    # sitk.WriteImage(sitk_mask, 'step3.mha')
    # 4
    skeleton_mask = GetLargestConnectedCompont(sitk_mask)
    # sitk.WriteImage(skeleton_mask, 'step4.mha')
    # 5
    # sitk_skeleton = GetMaskImage(sitk_src, skeleton_mask, replacevalue=-1500)
    # sitk.WriteImage(sitk_skeleton, 'step5.mha')
    return skeleton_mask


def vessleSegment(pathDicom):
    sigma_minimum = 0.2
    sigma_maximum = 3.
    number_of_sigma_steps = 8
    lowerThreshold = 40
    output_image = 'vessel.mha'
    input_image = itk.imread(pathDicom, itk.F)
    # 1
    ImageType = type(input_image)
    Dimension = input_image.GetImageDimension()
    HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
    HessianImageType = itk.Image[HessianPixelType, Dimension]
    objectness_filter = itk.HessianToObjectnessMeasureImageFilter[HessianImageType, ImageType].New()
    objectness_filter.SetBrightObject(True)
    objectness_filter.SetScaleObjectnessMeasure(True)
    objectness_filter.SetAlpha(0.5)
    objectness_filter.SetBeta(1.0)
    objectness_filter.SetGamma(5.0)
    multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[ImageType, HessianImageType, ImageType].New()
    multi_scale_filter.SetInput(input_image)
    multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
    multi_scale_filter.SetSigmaStepMethodToLogarithmic()
    multi_scale_filter.SetSigmaMinimum(sigma_minimum)
    multi_scale_filter.SetSigmaMaximum(sigma_maximum)
    multi_scale_filter.SetNumberOfSigmaSteps(number_of_sigma_steps)
    itk.imwrite(multi_scale_filter.GetOutput(), "step1.mha")
    # 2
    OutputPixelType = itk.UC
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
    rescale_filter.SetInput(multi_scale_filter)
    itk.imwrite(rescale_filter.GetOutput(), "step2.mha")
    # 3
    thresholdFilter = itk.BinaryThresholdImageFilter[OutputImageType, OutputImageType].New()
    thresholdFilter.SetInput(rescale_filter.GetOutput())
    thresholdFilter.SetLowerThreshold(lowerThreshold)
    thresholdFilter.SetUpperThreshold(255)
    thresholdFilter.SetOutsideValue(0)
    thresholdFilter.SetInsideValue(255)
    itk.imwrite(thresholdFilter.GetOutput(), "step3.mha")
    # 4
    itk.imwrite(thresholdFilter.GetOutput(), output_image)


if __name__ == '__main__':
    pathDicom = 'E:\BG0001.nii.gz'
    vessleSegment(pathDicom)