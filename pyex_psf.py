# Adapted from the aperpy code available at https://github.com/astrowhit/aperpy

import yaml
import os
import subprocess
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, hstack
from astropy.nddata import block_reduce, Cutout2D
from astropy.stats import mad_std, sigma_clip
from astropy.convolution import convolve, convolve_fft
from astropy.modeling.fitting import LinearLSQFitter, FittingWithOutlierRemoval
from astropy.modeling.models import Linear1D
from astropy.visualization import ImageNormalize, LinearStretch

from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_com
from photutils.detection import find_peaks

from scipy.ndimage import zoom, binary_dilation
from skimage.morphology import disk

import warnings
warnings.resetwarnings()
warnings.filterwarnings('ignore', category=UserWarning, append=True)
np.errstate(invalid='ignore')

class SquareRootScale(mscale.ScaleBase):

    """
    ScaleBase class for generating square root scale.
    """

    name = 'squareroot'

    def __init__(self, axis, **kwargs):

        mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a)**0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a)**2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()

class PSF():

    def __init__(self, image_filename, band, psf_configfile):

        """__init__ method for PSF

        Args:
            image_filename (str):
                Path to image on which to base the PSF.
            band (str):
                Filter code corresponding to this PSF (eg. 'F444W').
            psf_configfile (str):
                Path to .yml file containing the configuration parameters for 
                PSF measuring, kernel genration and image convolution.
        """

        # Load the config file.
        self.psf_configfile = psf_configfile
        with open(self.psf_configfile, 'r') as file:
            yml = yaml.safe_load_all(file)
            content = []
            for entry in yml:
                content.append(entry)
            self.config = content[0]

        # Image to measure PSF from.
        self.image_filename = image_filename
        self.img, self.hdr = fits.getdata(image_filename, header=True) # Store image and header.
        self.wcs = WCS(self.hdr) # and WCS information.

        # The measurement band corresponding to the image.
        self.band = band.upper()

        # Some issue with pypher and some pixel scale headers.
        # So add 'PIXSCALE' from config file if not already in keys.
        if 'PIXSCALE' not in list(self.hdr.keys()):
            self.hdr['PIXSCALE'] = (self.config["PIXEL_SCALE"], 'Pixel scale given in config file')

        # Base output name.
        self.outname = f'{self.config["PSF_DIR"]}/{os.path.basename(image_filename).removesuffix(".fits")}'
    
    def powspace(self, start, stop, power=0.5, num = 30, **kwargs):
        """Generate a square-root spaced array with a specified number of points between two endpoints.

        Parameters
        ----------
        start (float):
            The starting value of the range.
        stop (float):
            The ending value of the range.
        power (float): 
            Power of distribution, defaults to sqrt.
        num (int):
            The number of points to generate in the array. Default is 30.

        Returns
        -------
        (numpy.ndarray):
            A 1-D array of 'num' values spaced equally in square-root space
            between 'start' and 'stop'.
        """

        return np.linspace(start**power, stop**power, num=num, **kwargs)**(1/power)
    
    def measure_curve_of_growth(self, image, position=None, radii=None, rnorm='auto', nradii=30, rbackg=True, showme=False, verbose=False):
        """Measure a curve of growth from cumulative circular aperture photometry on a list of radii 
            centered on the center of mass of a source in a 2D image.

        Parameters
        ----------
        image (numpy.ndarray):
            2D image array.
        position (astropy.coordinates.SkyCoord/None):
            Position of the centre of the source. If 'None', it will be measured.
        radii (numpy.ndarray/None):
            Array of aperture radii. If None, use 'nradii'.
        rnorm (float):
            The radius to use for normalisation. Must be in 'radii'.
        nradii (int):
            Number of aperture radii to get from self.powspace. Only used if radii==None.
        rbackg (bool):
            Whether to perform backgound subtraction. 
        showme (bool):
            Whether to save COG and profile figure. 
        verbose (bool):
            Whether to print progress information.
        
        Returns
        -------
        radii (numpy.ndarray):
            The aperture radii used.
        cog (numpy.ndarray):
            The measured curve of growth.
        profile (numpy.ndarray):
            The measured profile.
        """

        # Default to a sqaure root spaced array.
        if type(radii) is type(None):
            radii = self.powspace(0.5,image.shape[1]/2,num=nradii)

        # Calculate the centroid of the source in the image if not given.
        if type(position) is type(None):
            position = centroid_com(image)

        # Create an aperture for each radius in radii.
        apertures = [CircularAperture(position, r=r) for r in radii]

        # Remove background if requested.
        if rbackg == True:
            bg_mask = apertures[-1].to_mask().to_image(image.shape) == 0
            # Background is median of image unmasked by apertures.
            bg = np.nanmedian(image[bg_mask])
            if verbose: print('background',bg)
        else:
            bg = 0.

        # Perform aperture photometry for each aperture
        phot_table = aperture_photometry(image-bg, apertures)
        # Calculate cumulative aperture fluxes
        cog = np.array([phot_table['aperture_sum_'+str(i)][0] for i in range(len(radii))])

        # Normalise at some radius.
        if rnorm == 'auto': rnorm = image.shape[1]/2.0
        if rnorm:
            rnorm_indx = np.searchsorted(radii, rnorm)
            cog /= cog[rnorm_indx]

        # Get the profile.
        area = np.pi*radii**2 # Area enclosed by each apperture.
        area_cog = np.insert(np.diff(area),0,area[0]) # Difference between areas.
        profile = np.insert(np.diff(cog),0,cog[0])/area_cog # Difference between COG elements.
        profile /= profile.max() # Normalise profile.

        # Show the COG and profile if requested.
        if showme:
            plt.scatter(radii, cog, s = 25, alpha = 0.7)
            plt.plot(radii,profile/profile.max())
            plt.xlabel('Radius [pix]')
            plt.ylabel('Curve of Growth')

        # Return the aperture radii, COG and profile.
        return radii, cog, profile

    def imshow(self, args, crosshairs=False, **kwargs):
        """Plot a PSF image.

        Parameters
        ----------
        args (array_like):
            The images to be plotted.
        crosshairs (bool):
            Whether to plot crosshairs on the image.

        Returns
        -------
        fig (pyplot.figure):
            Pyplot figure object.
        ax (pyplot.axes):
            Pyplot axes object.
        """

        # Base image width.
        width = 20

        # Return if no images given to plot.
        nargs = len(args)
        if nargs == 0: 
            return

        # Set some plotting keywords if not given.
        if not (ncol := kwargs.get('ncol')): ncol = int(np.ceil(np.sqrt(nargs)))+1 # Number of cloumns.
        if not (nsig := kwargs.get('nsig')): nsig = 5 # Number of sigma to use in normalisation.
        if not (stretch := kwargs.get('stretch')): stretch = LinearStretch() # Stretching to use.

        # Set up the figure.
        nrow = int(np.ceil(nargs/ncol)) # Number of rows to plot.
        panel_width = width/ncol # Width of each panel.
        fig, ax = plt.subplots(nrows=nrow, ncols=ncol,figsize=(ncol*panel_width,nrow*panel_width))

        if type(ax) is not np.ndarray: ax = np.array(ax)
        usedaxes = []
        for arg, axi in zip(args, ax.flat):
            usedaxes.append(axi)
            # Calculate MAD and use to normalise.
            sig = mad_std(arg[(arg != 0) & np.isfinite(arg)])
            if sig == 0: sig=1
            norm = ImageNormalize(np.float32(arg), vmin=-nsig*sig, vmax=nsig*sig, stretch=stretch)

            # Plot the image.
            axi.imshow(arg, norm=norm, origin='lower', interpolation='nearest')
            axi.set_axis_off()
            # Add crosshairs if requested.
            if crosshairs:
                axi.plot(50,50, color='red', marker='+', ms=10, mew=1)

        # Remove any unused axes.
        for axi in ax.flat:
            if axi not in usedaxes:
                fig.delaxes(axi)

        # If no title given, use the object index.
        if type(title := kwargs.get('title')) is not type(None):
            for fi,axi in zip(title,ax.flat): axi.set_title(fi)

        return fig, ax
    
    def get_acceptable_cutouts(self, peaks):
        """Update the list of acceptable peaks.

        Parameters
        ----------
        peaks (astropy.table.Table):
            Table of peaks information produced by self.find_stars.
        """

        # The RA, DEC and ID of each object.
        ra_stars = peaks['ra']
        dec_stars = peaks['dec']
        id_stars = peaks['id']

        # Convert these to pixel values based on WCS values.
        x_stars,y_stars = self.wcs.all_world2pix(ra_stars, dec_stars, 0)

        # The centre of the square cutout.
        self.cutout_centre = self.config["PSF_SIZE"]//2

        # Create a table containing star information.
        self.star_cat = Table([id_stars,x_stars,y_stars,ra_stars,dec_stars],names=['id','x','y','ra','dec'])

        # Take cutouts from base image for each object using pixel position calculated above.
        data = np.array([Cutout2D(self.img, (x_stars[i],y_stars[i]), (self.config["PSF_SIZE"], self.config["PSF_SIZE"]),mode='partial').data for i in np.arange(len(ra_stars))])
        
        # Mask out infinite values and zeros.
        self.psf_data = np.ma.array(data,mask = ~np.isfinite(data) | (data == 0) )
        self.psf_data_orig = self.psf_data.copy() # Keep a copy for later.

        # Reset the selection array.
        self.ok = np.ones(len(self.star_cat))

        return

    def find_stars(self):
        """Find stars in the image used to initalise the PSF object.

        Returns
        -------
        peaks[ok] (pyplot.figure):
            Information associated with each of the acceptable measured peaks.
        stars[ok] (list):
            Contains all the measured COGs for the acceptable peaks.
        """

        print(f'Identifying bright sources in {self.image_filename}...')

        # Get image and corresponding header.
        img, hdr = fits.getdata(self.image_filename, header=True)
        wcs = WCS(hdr)

        # Calaculate MAD of image.
        imgb = block_reduce(img, self.config["BLOCK_SIZE"], func=np.sum)
        sig = mad_std(imgb[imgb>0], ignore_nan=True)/self.config["BLOCK_SIZE"]

        print(f' Finding peaks {self.config["NSIG_THRESHOLD"]}x above the MAD')
        # Find at maximum 'npeaks' above the threshold.
        peaks = find_peaks(img, threshold=self.config["NSIG_THRESHOLD"]*sig, npeaks=self.config["N_PEAKS"])
        peaks.rename_column('x_peak','x')
        peaks.rename_column('y_peak','y')

        # Convert pixel locations to world.
        ra,dec = wcs.all_pix2world(peaks['x'], peaks['y'], 0)
        peaks['ra'] = ra
        peaks['dec'] = dec
        peaks['x0'] = 0.0
        peaks['y0'] = 0.0
        peaks['minv'] = 0.0

        # Add columns for the COG and profile within each radius.
        for ir in np.arange(len(self.config["RADII"])): peaks['r'+str(ir)] = 0.
        for ir in np.arange(len(self.config["RADII"])): peaks['p'+str(ir)] = 0.

        print(f' Measuring properites of each object')
        stars = []
        # Iterate over the measured peaks.
        for ip,p in enumerate(peaks):

            # Create cutout.
            co = Cutout2D(img, (p['x'], p['y']), self.config["STAR_SIZE"], mode='partial')

            position = centroid_com(co.data) # Find centre of mass of the image.

            peaks['x0'][ip] = position[0] - self.config["STAR_SIZE"]//2 # Used for shift.
            peaks['y0'][ip] = position[1] - self.config["STAR_SIZE"]//2

            peaks['minv'][ip] = np.nanmin(co.data) # Image minimum value.

            # Measure the the COG and profile.
            _ , cog, profile = self.measure_curve_of_growth(co.data, radii=np.array(self.config["RADII"]), position=position, rnorm=None, rbackg=False)
            for ir in np.arange(len(self.config["RADII"])): peaks['r'+str(ir)][ip] = cog[ir]
            for ir in np.arange(len(self.config["RADII"])): peaks['p'+str(ir)][ip] = profile[ir]

            # Store radii, COG and profile as part of cutout and add to stars.
            co.radii = np.array(self.config["RADII"])
            co.cog = cog
            co.profile = profile
            stars.append(co)

        # Array containing all measured stars.
        stars = np.array(stars)

        # Magnitude of star based on maximum COG at maximum radius.
        peaks['mag'] = self.config["MAG_ZP"]-2.5*np.log10(peaks['r4'])

        # Ratio of COG at maxmium and middle value.
        r = peaks['r4']/peaks['r2']
        shift_lim_root = np.sqrt(self.config["SHIFT_LIM"])

        # Check conditions:
        ok_mag =  peaks['mag'] < self.config["MAG_MIN"] # Above magnitude limit?
        ok_min =  peaks['minv'] > self.config["THRESHOLD_MIN"] # Minimum value above threshold?
        ok_phot = np.isfinite(peaks['r'+str(len(self.config["RADII"])-1)]) &  np.isfinite(peaks['r2']) & np.isfinite(peaks['p1']) # COG well defined?
        ok_shift = (np.sqrt(peaks['x0']**2 + peaks['y0']**2) < self.config["SHIFT_LIM"]) & \
                (np.abs(peaks['x0']) < shift_lim_root) & (np.abs(peaks['y0']) < shift_lim_root) # Within acceptable offset?

        # Ratio of COG at maximum and middle value.
        fig = plt.figure()
        h = plt.hist(r[(r>1.2) & ok_mag], bins=np.arange(0, self.config["RANGE"][1], self.config["THRESHOLD_MODE"][1]/2.),range=self.config["RANGE"])
        fig.clear()

        # Modal bin count and value.
        ih = np.argmax(h[0])
        rmode = h[1][ih]
        ok_mode =  ((r/rmode-1) > self.config["THRESHOLD_MODE"][0]) & ((r/rmode-1) < self.config["THRESHOLD_MODE"][1]) # Close enough to the mode?

        # Full selection criteria.
        ok = ok_phot & ok_mode & ok_min & ok_shift & ok_mag
            
        # Fit linear relation with sigma clipping.
        # x = magnitude at maximum radius, y = ratio of COG at middle radius to maximum radius, 
        print(' Fitting and removing outliers')
        try:
            fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, sigma = self.config["SIGMA_FIT"], niter=2)
            lfit, outlier = fitter(Linear1D(),x=self.config["MAG_ZP"]-2.5*np.log10(peaks['r4'][ok]),y=(peaks['r4']/peaks['r2'])[ok])
            ioutlier = np.where(ok)[0][outlier]
            ok[ioutlier] = False
        except:
            print('  linear fit failed')
            ioutlier = 0
            lfit = None

        peaks['id'] = 1
        peaks['id'][ok] = np.arange(1,len(peaks[ok])+1)

        # Produce diagnostic figures.
        suffix = '.fits' + self.image_filename.split('.fits')[-1]
        if self.config['SAVE_FIGS'] == True:

            plt.figure(figsize=(14,8))
            plt.subplot(231)
            mags = peaks['mag']
            mlim_plot = np.nanpercentile(mags,[5,95]) + np.array([-2,1])
            plt.scatter(mags,r, alpha = 0.3, color = 'grey', s = 2, label = 'all')
            plt.scatter(mags[~ok_shift],r[~ok_shift],label='bad shift',c='C1', alpha = 0.8, s = 6)
            plt.scatter(mags[ok],r[ok],label='ok',c='C2', alpha = 0.8, s = 6)
            plt.scatter(mags[ioutlier],r[ioutlier],label='outlier',c='darkred', alpha = 0.8, s = 6)
            if lfit: plt.plot(np.arange(14,30), lfit(np.arange(14,30)),'--',c='k',alpha=0.3,label='slope = {:.3f}'.format(lfit.slope.value))
            plt.legend()
            plt.ylim(0,14)
            plt.xlim(mlim_plot[0],mlim_plot[1])
            plt.xlabel('$\\mathrm{m}_{A4}$')
            plt.ylabel('A2/A4')

            plt.subplot(232)
            ratio_median = np.nanmedian(r[ok])
            plt.scatter(mags,r, alpha = 0.3, color = 'grey', s = 3)
            plt.scatter(mags[~ok_shift],r[~ok_shift],s = 10, alpha = 0.8,label='bad shift',c='C1')
            plt.scatter(mags[ok],r[ok],s = 10, alpha = 0.8,label='ok',c='C2')
            plt.scatter(mags[ioutlier],r[ioutlier],s = 10, alpha = 0.8,label='outlier',c='darkred')
            if lfit: plt.plot(np.arange(15,30), lfit(np.arange(15,30)),'--',c='k',alpha=0.3,label='slope = {:.3f}'.format(lfit.slope.value))
            plt.ylim(ratio_median-1,ratio_median+1)
            plt.xlim(mlim_plot[0],mlim_plot[1])
            plt.xlabel('$\\mathrm{m}_{A4}$')
            plt.ylabel('A2/A4')

            plt.subplot(233)
            _ = plt.hist(r[(r>1.2) & ok_mag],bins=np.arange(0, self.config["RANGE"][1], self.config["THRESHOLD_MODE"][1]/2.),range=self.config["RANGE"], alpha = 0.7, color = 'grey')
            _ = plt.hist(r[ok],bins=np.arange(0, self.config["RANGE"][1], self.config["THRESHOLD_MODE"][1]/2.),range=self.config["RANGE"], color='C2', alpha = 1)
            plt.xlabel('A2/A4')
            plt.ylabel('N')

            plt.subplot(234)
            plt.scatter(self.config["MAG_ZP"]-2.5*np.log10(peaks['r3'][ok]),(peaks['peak_value']/peaks['r3'])[ok], color = 'C2', s = 10, alpha = 0.8)
            plt.scatter(self.config["MAG_ZP"]-2.5*np.log10(peaks['r3'])[ioutlier],(peaks['peak_value'] /peaks['r3'])[ioutlier],c='darkred', s = 10, alpha = 0.8)
            plt.ylim(0,1)
            plt.xlabel('$\\mathrm{m}_{A3}$')
            plt.ylabel('peak/A3')


            plt.subplot(235)
            plt.scatter(peaks['x0'][ok],peaks['y0'][ok],c='C2', alpha = 0.8, s=10)
            plt.scatter(peaks['x0'][ioutlier],peaks['y0'][ioutlier],c='darkred', alpha = 0.8, s =10)
            plt.xlim(-2,2)
            plt.ylim(-2,2)
            plt.xlabel('X-offset [pix]')
            plt.ylabel('Y-offset [pix]')


            plt.subplot(236)
            plt.scatter(peaks['x'][ok],peaks['y'][ok],c='C2', alpha = 0.8, s=10)
            plt.scatter(peaks['x'][ioutlier],peaks['y'][ioutlier],c='darkred', alpha = 0.8, s=10)
            plt.axis('scaled')
            plt.tight_layout()
            plt.savefig(self.config["FIG_DIR"]+'/'+os.path.basename(self.image_filename).replace(suffix,'_diagnostic.pdf'))
            plt.xlabel('X [pix]')
            plt.ylabel('Y [pix]')

            dd = [st.data for st in stars[ok]]
            title = ['{}: {:.1f} AB, ({:.1f}, {:.1f})'.format(ii, mm,xx,yy) for ii,mm,xx,yy in zip(peaks['id'][ok],mags[ok],peaks['x0'][ok],peaks['y0'][ok])]
            self.imshow(dd,nsig=30,title=title)
            plt.tight_layout()
            plt.savefig(self.config["FIG_DIR"]+'/'+os.path.basename(self.image_filename).replace(suffix,'_all_peaks.pdf'))
        
        # Write the acceptable peaks to a catalogue.
        peaks[ok].write(self.config["PSF_DIR"]+'/'+os.path.basename(self.image_filename).replace(suffix,'_star_cat.fits'),overwrite=True)
        return peaks[ok], stars[ok]
    
    def imshift(self, img, ddx, ddy, interpolation=cv2.INTER_CUBIC):
        """Recentre an image using an affine transformation.

        Parameters
        ----------
        img (numpy.ndarray):
            2D image array to be recentred.
        ddx (float):
            Shift in the x direction.
        ddy (numpy.ndarray/None):
            Shift in the y direction.
        interpolation (cv2 interpolator):
            Interpolation approach to use.
        
        Returns
        -------
        (numpy.ndarray):
            The recentred image.
        """

        # Create the transformation matrix.
        M = np.float32([[1,0,ddx],[0,1,ddy]])

        # Output cutout size.
        wxh = img.shape[::-1]
        return cv2.warpAffine(img, M, wxh, flags=interpolation)
    
    def centre(self,window,interpolation=cv2.INTER_CUBIC):
        """Centre the stars measured from the image onto the PSF and measure contamination.

        Parameters
        ----------
        window (numpy.ndarray):
            Length of square window side to use for cutout.
        interpolation (cv2 interpolator):
            The type of interpolation to use.
        ddy (numpy.ndarray/None):
            Shift in the y direction.
        interpolation (cv2 interpolator):
            Interpolation approach to use.
        """

        # Get the window width and cutout centre.
        cw = window//2
        c0 = self.cutout_centre

        pos = []
        # Iterate over the different point sources.
        for i in np.arange(len(self.psf_data)):
            p = self.psf_data[i,:,:]

            # Measure the COM of the source.
            st = Cutout2D(p,(c0,c0),window,mode='partial',fill_value=0).data
            st[~np.isfinite(st)] = 0
            x0, y0 = centroid_com(st)

            # Recentre the cutout.
            p = self.imshift(p, (cw-x0), (cw-y0),interpolation=interpolation)

            # Now measure COM on recentered cutout.
            # First in a small window
            st = Cutout2D(p,(c0,c0),window,mode='partial',fill_value=0).data
            x1,y1 = centroid_com(st)

            # Measure moment shift in positive definite in case there are strong ying yang residuals
            x2,y2 = centroid_com(np.maximum(p,0))

            # Mask infinite or zero values.
            p = np.ma.array(p, mask = ~np.isfinite(p) | (p==0))

            # Save this recentred cutout.
            self.psf_data[i,:,:] = p

            # Difference in shift between central window and half of stamp is measure of contamination
            # from bright off axis sources.
            dsh = np.sqrt(((c0-x2)-(cw-x1))**2 + ((c0-y2)-(cw-y1))**2)
            pos.append([cw-x0,cw-y0,cw-x1,cw-y1,dsh])

        # Add these measurements to the star catalogue.
        self.star_cat = hstack([self.star_cat,Table(np.array(pos),names=['x0','y0','x1','y1','dshift'])])
    
        return

    def phot(self, radius=8):
        """Measure flux within a circular aperture for all objects in self.psf_data.

        Parameters
        ----------
        radius (int):
            Radius of the circular aperture used for photometry.

        Returns
        --------
        phot (list):
            List containing the measured photometry for each source.
        """

        caper = CircularAperture((self.cutout_centre,self.cutout_centre), r=radius)
        phot = [aperture_photometry(st, caper)['aperture_sum'][0] for st in self.psf_data]
        return phot
    
    def stamp_rms_snr(self, img, block_size=3, rotate=True):
        """Measure the MAD and corresponding SNR within an image.

        Parameters
        ----------
        img (numpy.ndarray):
            The image for which to measure the SNR.
        block_size (int):


        Returns
        --------
        rms (float):
            The measured MAD.
        snr (float):
            The calculated SNR.
        """

        # Flip the image if required.
        if rotate:
            p180 = np.flip(img,axis=(0,1))
            dp = img-p180
        else:
            dp = img.copy()

        # Create a buffer around the edge of the image.
        s = dp.shape[1]
        buf = 6
        dp[s//buf:(buf-1)*s//buf,s//buf:(buf-1)*s//buf] = np.nan

        # Calculate the MAD.
        rms = mad_std(dp,ignore_nan=True)/block_size * np.sqrt(img.size)
        if rotate: rms /= np.sqrt(2)

        # Calculate the corresponding SNR.
        snr = img.sum()/rms
        return rms, snr

    def grow(self, mask, structure=disk(2), **kwargs):
        """Grow a mask.

        Parameters
        ----------
        mask (numpy.ndarray):
            The mask to grow.
        structure (array_like):
            Structuring element used for dilation.
        **kwargs:
            Keyword arguments to pass to scipy.ndimage.binary_dilation.

        Returns
        --------
        (numpy.ndarray):
            The dilated mask.
        """

        return binary_dilation(mask,structure=structure,**kwargs)
    
    def sigma_clip_3d(self, data, sigma=3, maxiters=2, axis=0, **kwargs):
        """Measure the pixelwise sigma clipped mean from multiple images.

        Parameters
        ----------
        data (array_like):
            The data to be sigma clipped
        maxiters (int):
            The number of sigma clipping iterations.
        axis (int):
            The axis within data along which to perform the sigma clipping.
        **kwargs (array_like):
            Keyword arguments to pass to astropy.stats.sigma_clip.

        Returns
        --------
        np.mean(clipped_data,axis=axis) (array):
            The mean values of the sigma clipped pixel data.
        lo (float):
            Minimum clipping bound in final iteration.
        hi (float):
            Maximum clipping bound in the final iteration.
        clipped_data (array_like):
            The data after sigma clipping.
        """

        # Make a copy of the data.
        clipped_data = data.copy()

        # Perform required number of sigma clipping iterations.
        for i in range(maxiters):
            clipped_data, lo, hi = sigma_clip(clipped_data, sigma=sigma, maxiters=0, axis=0, masked=True, grow=False, return_bounds=True, **kwargs)
            
            # Grow the mask
            for i in range(len(clipped_data.mask)): clipped_data.mask[i,:,:] = self.grow(clipped_data.mask[i,:,:],iterations=1)

        return np.mean(clipped_data,axis=axis), lo, hi, clipped_data

    def measure(self,norm_radius=8):
        """Measure the photometric properties of a source."""

        norm_radius = self.config["NORM_RADIUS"]

        # Find the peak value in each cutout.
        peaks = np.array([st.max()for st in self.psf_data])
        peaks[~np.isfinite(peaks) | (peaks==0)] = 0 # Must be finite and non-zero.

        # Create a mask around the centre of the PSF.
        caper = CircularAperture((self.cutout_centre,self.cutout_centre), r=norm_radius)
        cmask = Cutout2D(caper.to_mask(),(norm_radius,norm_radius), self.config["PSF_SIZE"],mode='partial').data

        # Measure the flux of the PSF.
        phot = self.phot(radius=norm_radius)
        sat =  [aperture_photometry(st, caper)['aperture_sum'][0] for st in np.array(self.psf_data)] # casting to array removes mask
        # Minimum unmasked value.
        cmin = [ np.nanmin(st*cmask) for st in self.psf_data]

        # Masked fraction.
        self.star_cat['frac_mask'] = 0.0

        # Combine with mask from recentering.
        for i in np.arange(len(self.psf_data)):
            self.psf_data[i].mask |= (self.psf_data[i]*cmask) < 0.0

        # Save some information to the catalogue.
        self.star_cat['peak'] =  peaks
        self.star_cat['cmin'] =  np.array(cmin)
        self.star_cat['phot'] =  np.array(phot)
        self.star_cat['saturated'] =  np.int32(~np.isfinite(np.array(sat))) # Is the image saturated?

        # Measure SNR.
        rms_array = []
        for st in self.psf_data:
            rms, snr = self.stamp_rms_snr(st)
            rms_array.append(rms)

        self.star_cat['snr'] = 2*np.array(phot)/np.array(rms_array)
        self.star_cat['phot_frac_mask'] = 1.0

        return

    def select(self, snr_lim = 800, dshift_lim=3, mask_lim=0.40, phot_frac_mask_lim = 0.85):
        """Select objects satisfying given conditions from the catalogue.

        Parameters
        ----------
        snr (float):
            Minimum accepted SNR.
        dshift (float):
            Maximum accepted difference in shift measured when recentering.
        mask_lim (float):
            Maximum accepted fraction of masked pixels.
        phot_frac_mask_lim (float):
            Minimum accepted flux within "NORM_RADIUS" of the centre.
        """

        # Check which objects in the catalogue satisfy all conditions.
        self.ok = (self.star_cat['dshift'] < dshift_lim) & (self.star_cat['snr'] > snr_lim) & (self.star_cat['frac_mask'] < mask_lim) & (self.star_cat['phot_frac_mask'] > phot_frac_mask_lim)

        self.star_cat['ok'] = np.int32(self.ok)
        # All include individual conditions.
        self.star_cat['ok_shift'] = (self.star_cat['dshift'] < dshift_lim)
        self.star_cat['ok_snr'] = (self.star_cat['snr'] > snr_lim)
        self.star_cat['ok_frac_mask'] = (self.star_cat['frac_mask'] < mask_lim)
        self.star_cat['ok_phot_frac_mask'] = (self.star_cat['phot_frac_mask'] > phot_frac_mask_lim)

        # Format the columns to 3 D.P
        for c in self.star_cat.colnames:
            if 'id' not in c: self.star_cat[c].format='.3g'

        return

    def stack(self,sigma=3,maxiters=2):
        """Stack individual PSFs based on a pixelwise sigma clipped mean.

        Parameters
        ----------
        sigma (float):
            Sigma to pass to the sigma clipping function.
        maxiters (float):
            Maximum number of sigma clipping iterations.
        """

        # Get indexes of acceptable objects.
        iok = np.where(self.ok)[0]

        print(f'Measuring average PSF based on {len(iok)} cutouts...')

        # Get flux within normalisation radius.
        norm = self.star_cat['phot'][iok]

        # Find fraction of flux within the normalisation radius.
        data = self.psf_data_orig[iok].copy()
        for i in np.arange(len(data)): data[i] = data[i]/norm[i]

        # Stack the images based on the pixelwise sigma clipped mean
        stack, lo, hi, clipped = self.sigma_clip_3d(data,sigma=sigma,axis=0,maxiters=maxiters)

        # The remaining PSFs after clipping.
        self.clipped = clipped

        for i in np.arange(len(data)):
            # Does object satisfy criteria and also have its central pixel unmasked?
            self.ok[iok[i]] = self.ok[iok[i]] and ~self.clipped[i].mask[self.cutout_centre,self.cutout_centre]
            # Use the clipped mask.
            self.psf_data[iok[i]].mask = self.clipped[i].mask
            # The fraction of pixels that are masked.
            mask = self.psf_data[iok[i]].mask
            self.star_cat['frac_mask'][iok[i]] = np.size(mask[mask]) / np.size(mask)
    
        if self.config["SAVE_FIGS"] == True:
            # Save the masked cutouts of all the stacked sources.
            title = ['{}: Mask - {:.1f}%'.format(ii, 100*frac) for ii,frac in zip(self.star_cat['id'][iok],self.star_cat['frac_mask'][iok])]
            fig, ax = self.imshow(self.psf_data[iok], title=title, nsig=30)
            fig.savefig('_'.join([self.outname, 'stacked_stamps.pdf']).replace(self.config["PSF_DIR"],self.config["FIG_DIR"]),dpi=300)

        # Save the stacked PSF.
        self.psf_average = stack

        # Calculate the fraction of the flux within the normalisation radius.
        self.star_cat['phot_frac_mask'] = self.phot(radius=self.config["NORM_RADIUS"])/self.star_cat['phot']

        return

    def save(self, outname=''):
        """Save stacked PSF to a fits file and the masked PSF stamps figure.

        Parameters
        ----------
        outname (str):
            Base name of the output files.
        """

        # Save the stacked empirical PSF.
        fits.writeto('_'.join([outname, 'EPSF.fits']), np.array(self.psf_average), header = self.hdr, overwrite=True)

        # Save the final star catalogue.
        self.star_cat[self.ok].write('_'.join([outname, 'psf_cat.fits']),overwrite=True)


        if self.config["SAVE_FIGS"] == True:

            # Original unmasked data.
            data = self.psf_data_orig

            # Save the cutouts of all of the identified peaks.
            title = ['ID: {}'.format(ii) for ii in self.star_cat['id']]
            fig, ax = self.imshow(data, title=title, nsig=30)
            fig.savefig('_'.join([outname, 'all_stamps.pdf']).replace(self.config["PSF_DIR"],self.config["FIG_DIR"]),dpi=300)

            # Save Curve Of Growth.
            self.show_cogs([self.psf_average], title = self.band, linear = False, pixscale=self.config["PIXEL_SCALE"], label=[f'Average {self.band} PSF'], outname=self.outname.replace(self.config["PSF_DIR"],self.config["FIG_DIR"]))

    def renorm_psf(self, psfmodel, filt, fov, pixscl):
        """Renormalise the PSF based on the expected encircled energy.

        Parameters
        ----------
        psfmodel (numpy.ndarray):
            The 2D PSF to renormalise.
        filt (str):
            The filter code correspnding to the band used to measure the PSF
        fov (float):
            The FOV of the PSF. 
        pixscl (float):
            The pixel scale of the PSF.
        """
        
        filt = filt.upper()

        # Encircled energy expectation from telescope documentation.
        encircled = {}
        encircled['F225W'] = 0.993
        encircled['F275W'] = 0.984
        encircled['F336W'] = 0.9905
        encircled['F435W'] = 0.979
        encircled['F606W'] = 0.975
        encircled['F775W'] = 0.972
        encircled['F814W'] = 0.972
        encircled['F850LP'] = 0.970
        encircled['F098M'] = 0.974
        encircled['F105W'] = 0.973
        encircled['F125W'] = 0.969
        encircled['F140W'] = 0.967
        encircled['F160W'] = 0.966
        encircled['F090W'] = 0.9837
        encircled['F115W'] = 0.9822
        encircled['F150W'] = 0.9804
        encircled['F200W'] = 0.9767
        encircled['F277W'] = 0.9691
        encircled['F356W'] = 0.9618
        encircled['F410M'] = 0.9568
        encircled['F444W'] = 0.9546

        # Normalize to correct for missing flux
        w, h = np.shape(psfmodel) # PSF shape.
        Y, X = np.ogrid[:h, :w]

        r = fov / 2. / pixscl # Half side length of PSF in pixels.
        centre = [w/2., h/2.] # Centre of PSF

        dist_from_centre = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2) # Distance of each pixel from centre.

        psfmodel /= np.sum(psfmodel[dist_from_centre < r]) # Normalise by total flux in sphere radius r from centre.
        psfmodel *= encircled[filt] # Multiply by encircled fraction.

        # Save to seperate file.
        fits.writeto('_'.join([self.outname, 'EPSF_renorm.fits']), np.array(psfmodel),overwrite=True)

        return

    def show_cogs(self,*args, title, linear, pixscale, label, outname):
        """Display COG and profile of measured PSF.

        Parameters
        ----------
        *args (list):
            The list of PSFs for which to measure the COG.
        title (str):
            The title of the figures.
        linear (bool):
            Plot in linear scale on the x-axis. If False, use square root scale. 
        pixscale (float):
            The pixel scale of the PSFs.
        label (list):
            List of labels for the PSFs
        outname (str):
            Base output name of the figures.
        """

        # The square root scale to use if not linear.
        mscale.register_scale(SquareRootScale)
        
        npsfs = len(args)
        nfilts = len(args[0])
    
        # Set the figure size/
        plt.figure(figsize=(20,4.5))

        # x axis ticks in arcseconds.
        xtick = [0.1,0.2,0.3,0.5,0.7,1.0,1.5,2.0]

        # Set empty label if none given.
        if not label:
            label = ['' for p in range(npsfs)]

        # In theory supports multiple PSFs.
        for filti in range(nfilts):

            # Measure the curve of growth.
            psf_ref = args[0][filti]
            r, cog_ref, prof_ref = self.measure_curve_of_growth(psf_ref,nradii=50)
            r = r * pixscale # Convert radius to arcsec

            # Plot the profile.
            plt.subplot(141)
            plt.plot(r,prof_ref,label=label[0])
            plt.title(title+' profile')
            if not linear:
                plt.xscale('squareroot')
                plt.xticks(xtick)
            plt.yscale('log')
            plt.xlim(0,1)
            plt.ylim(1e-5,1)
            plt.xlabel('arcsec')
            plt.axhline(y=0,alpha=0.5,c='k')
            plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
            ax=plt.gca()
            rms, snr = self.stamp_rms_snr(psf_ref)
            dx, dy = centroid_com(psf_ref)
            plt.text(0.6,0.8,'snr = {:.2g} \nx0,y0 = {:.2f},{:.2f} '.format(snr,dx,dy),transform=ax.transAxes, c='C0')

            # Plot the curve of growth.
            plt.subplot(142)
            plt.plot(r,cog_ref,label=label[0])
            plt.xlabel('arcsec')
            plt.title('cog')
            if not linear:
                plt.xscale('squareroot')
                plt.xticks(xtick)
            plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
            plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
            plt.axvline(x=0.08,alpha=0.5,c='k',ls='--')
            plt.axvline(x=0.04,alpha=0.5,c='k',ls='--')
            plt.xlim(0.02,1)

            cogs = []
            profs = []
            psfs = [psf_ref]
            for psfi in np.arange(1, npsfs):
                psf = args[psfi][filti]
                _, cog, prof = self.measure_curve_of_growth(psf,nradii=50)
                cogs.append(cog)
                profs.append(prof)
                dx, dy = centroid_com(psf)
                rms, snr = self.stamp_rms_snr(psf)

                plt.subplot(141)
                plt.plot(r,prof)

                plt.text(0.5,0.8-psfi*0.1,'snr = {:.2g} \nx0,y0 = {:.2f},{:.2f} '.format(snr,dx,dy),transform=ax.transAxes, c='C'+str(psfi))
                plt.xlim(0.02,1)

                plt.subplot(142)
                plt.plot(r,cog,label=label[psfi],c='C'+str(psfi))
                plt.legend()

                psfs.append(psf)

            plt.savefig('_'.join([outname,'cog.pdf']),dpi=300)

            _ = self.imshow(psfs,cross_hairs=True,nsig=50,title=label)

            plt.savefig('_'.join([outname,'average.pdf']),dpi=300)
        
        return
    

    def measure_psf(self):
        """Employ the functions defined above to measure an empirical PSF from an image."""

        # Find the stars.
        peaks, stars = self.find_stars()

        # Only use stars of a particular magnitude.
        ok = (peaks['mag'] > self.config['MAG_MAX']) & ( peaks['mag'] < self.config['MAG_MIN'])
        print(f' Selected {sum(ok)} objects')
        self.get_acceptable_cutouts(peaks[ok])

        # Centre the images.
        self.centre(window=self.config["WINDOW"])
        # Measure objects.
        self.measure()
        # Select objects with acceptable SNR and shift.
        self.select(self.config["SNR_LIM"], self.config["DSHIFT_LIM"], 0.99, 0.99)
        # Stack objects.
        self.stack(sigma=self.config["STACK_SIGMA"])
        # Save the PSF.
        self.save(self.outname)
        # Renormalise to account for missing flux.
        self.renorm_psf(self.psf_average, self.band, self.config["PIXEL_SCALE"]*self.config["PSF_SIZE"], self.config["PIXEL_SCALE"])
        
        if self.config["SAVE_FIGS"] == True:
            # Select dominant objects for plotting. <40% total masked and >85% of flux within the aperture.
            self.select(self.config["SNR_LIM"], self.config["DSHIFT_LIM"], 0.4, 0.85)

            # Save cutouts of the dominant stamps used for PSF generation.
            title = ['{}: {:.1f} AB, ({:.1f}, {:.1f})'.format(ii, mm,xx,yy) for ii,mm,xx,yy in zip(peaks['id'][self.ok],peaks['mag'][self.ok],peaks['x0'][self.ok],peaks['y0'][self.ok])]
            self.imshow(self.psf_data[self.ok],nsig=50,title=title)
            plt.savefig('_'.join([self.outname, 'dominant_stamps.pdf']).replace(self.config["PSF_DIR"],self.config["FIG_DIR"]),dpi=300)
        
        return
    
    def create_matching_kernel(self, target_PSF):
        """Create a kernel to match the measured PSF to a target PSF.

        Parameters
        ----------
        target (str/pyex.PSF):
            The target PSF to match to. If string should be a filepath to the target PSF.
            If pyex.PSF should be a PSF object that has already called the 'measure_psf' method.
        """

        print('Generating a matching kernel:')

        self.matching_band = ''
        # Read in the target PSF
        if type(target_PSF) == str:
            target_name = os.path.basename(target_PSF)
            target = fits.getdata(target_PSF)
            hdr = fits.getheader(target_PSF)
        if type(target_PSF) == PSF:
            target_name = os.path.basename(target_PSF.outname)
            target = fits.getdata('_'.join([target_PSF.outname, 'EPSF.fits']))
            hdr = fits.getheader('_'.join([target_PSF.outname, 'EPSF.fits']))
            self.matching_band = target_PSF.band
        else:
            raise TypeError(f'target_PSF must be type "str" or "PSF" but is type {type(target_PSF)}')

        # Oversample if required.
        if self.config['OVERSAMPLE'] > 1:
            print(f' Oversampling target PSF by {self.config["OVERSAMPLE"]}x')
            target = zoom(target, self.config['OVERSAMPLE'])

        # Renormalise
        target /= target.sum()

        # Save to temporary file.
        fits.writeto(f'{os.path.dirname(self.outname)}/target.temp.fits',target,header=hdr,overwrite=True)

        # Now do the same with the source PSF.
        source = fits.getdata('_'.join([self.outname, 'EPSF.fits']))
        if self.config['OVERSAMPLE'] > 1:
            print(f' Oversampling source PSF by {self.config["OVERSAMPLE"]}x')
            source = zoom(source, self.config['OVERSAMPLE'])
    
        source /= source.sum()

        fits.writeto(f'{os.path.dirname(self.outname)}/source.temp.fits',source,header=hdr,overwrite=True)

        # Filename of matching kernel.
        match_name = f'{self.outname}_to_{target_name}_kernel.fits'

        # Run pypher
        print(' Running pypher')
        pypherCMD = ['pypher', f'{os.path.dirname(self.outname)}/source.temp.fits', f'{os.path.dirname(self.outname)}/target.temp.fits', match_name, '-r', str(self.config["R_PARAMETER"]), '-s', str(self.config["ANGLE_S"]), '-t', str(self.config["ANGLE_T"])]
        p = subprocess.Popen(pypherCMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        # Remove the temporary files.
        os.remove(f'{os.path.dirname(self.outname)}/target.temp.fits')
        os.remove(f'{os.path.dirname(self.outname)}/source.temp.fits')

        # Store the matching kernel for later.
        self.matching_kernel = fits.getdata(match_name)

        # If oversampled, renormalise and overwrite saved and stored kernels.
        if self.config['OVERSAMPLE'] > 1:
            print(' Renormalising the oversampled kernel')
            matching_hdr = fits.getheader(match_name)

            # Renormalise
            self.matching_kernel = block_reduce(self.matching_kernel,self.config['OVERSAMPLE'], func=np.sum)
            self.matching_kernel /= self.matching_kernel.sum()

            # Overwrite original kernel.
            fits.writeto(match_name, self.matching_kernel, header=matching_hdr, overwrite=True)

        return
    
    def convolve_image(self, wht_filename, sci_filename = None):
        """Convolve an image with the measured matching kernel.

        Parameters
        ----------
        wht_filename (str):
            Path to the weight image associated with the image that will be convolved.
        sci_filename (str):
            Path to the science image to convolve. If None, use the image the PSF was measured from.
        """

        # If no science image given, use that used to generate PSF.
        if sci_filename == None:
            sci_filename = self.image_filename
        
        print(f'Convolving {sci_filename}:')

        # Load the two images.
        sci_image, sci_hdr = fits.getdata(sci_filename, header=True)
        wht_image, wht_hdr = fits.getdata(wht_filename, header=True)

        # Convert weight to error image for convolution.
        err_image = np.where(wht_image==0, 0, 1/np.sqrt(wht_image))

        # Name of matched image.
        sci_matched = f'{sci_filename.removesuffix(".fits")}_match{self.matching_band.lower()}.fits'
        wht_matched = f'{wht_filename.removesuffix(".fits")}_match{self.matching_band.lower()}.fits'


        # Convolve the images.
        if self.config["USE_FFT"] == True:
            print(' Using fft')
            print(' Convolving science image')
            convolved_sci = convolve_fft(sci_image, self.matching_kernel, allow_huge=True)
            print(' Convolving weight image')
            convolved_err = convolve_fft(err_image, self.matching_kernel, allow_huge=True)
            sci_hdr['MMETH'] = ('convolve_fft', 'Function used for convolution')
            wht_hdr['MMETH'] = ('convolve_fft', 'Function used for convolution')

        else:
            print(' Convolving science image')
            convolved_sci = convolve(sci_image, self.matching_kernel)
            print(' Convolving weight image')
            convolved_err = convolve(err_image, self.matching_kernel)
            sci_hdr['MMETH'] = ('convolve', 'Function used for convolution')
            wht_hdr['MMETH'] = ('convolve', 'Function used for convolution')

        sci_hdr['MBAND'] = (self.matching_band, 'Image has been matched to this PSF')
        wht_hdr['MBAND'] = (self.matching_band, 'Image has been matched to this PSF')

        # If weight pixel is zero, set convolved science pixel to zero.
        convolved_sci[wht_image==0] = 0.0

        # Convert back to weight.
        convolved_wht = np.where(convolved_err==0, 0, 1./(convolved_err**2))

        # Save the images.
        fits.writeto(sci_matched,convolved_sci,header=sci_hdr,overwrite=True)
        fits.writeto(wht_matched,convolved_wht,header=wht_hdr,overwrite=True)

        return