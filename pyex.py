import os
import yaml
import subprocess
import re
import copy
import h5py

import numpy as np
import matplotlib.pyplot as plt


from astropy.table import Table
from astropy.io import ascii, fits
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord

from scipy.stats import median_abs_deviation

from photutils.utils import ImageDepth

import emcee

class pyex():

    def __init__(self, config_file, sexpath ='sex', sexfile=None, outdir=None, verbose = True):

        """__init__ method for pyex

        Args:
            config (str):
                Path to ".yml" configuration file.
            sexpath (str):
                Path to the SExtractor executable.
            sexfile (str):
                Path to the base SExtractor configuration file.
            outdir (str):
                Path to directory in which to store outputs.
            verbose (bool):
                Whether to print init summary information.
        """

        # The path to the SExtractor executable.
        self.sexpath = sexpath

        # The directory in which to store any output files.
        if outdir is not None:
            self.outdir = outdir
            if os.path.isdir(outdir) == False:
                os.makedirs(outdir)
        # If none given, create and store in 'output' directory
        else:
            if os.path.isdir('output') == False:
                os.makedirs('output')
            self.outdir = 'output'

        # Path to a default or custom SExtractor configuration (.sex) file.
        if sexfile is not None:
            self.sexfile = sexfile
        else:
            self.sexfile = self.generate_default()        

        # Read the pyex configuration file and split into configuration, measurement, etc components.
        self.configfile = config_file
        with open(self.configfile, 'r') as file:
            yml = yaml.safe_load_all(file)
            content = []
            for entry in yml:
                content.append(entry)
            self.config, self.measurement, self.uncertainty, self.other = content

        # Print the parameters used to initalise the class for checking.
        if verbose == True:
            print(f'Initalised a pyex class with: \n Config parameters: {self.config} \n Outputs: {self.measurement} \n Uncertainty parameters: {self.uncertainty} \n Other: {self.other} \n')

    # Generate the default SExtractor configuration file if none given.
    def generate_default(self):
        """Function to generate the default SExtractor configuration file.

        Generates the default SExtractor configuration file and
        returns its path.

        Returns
        -------
        str
            Path to the default SExtractor configuration file.
        """

        # Pipe the SExtractor output.
        p = subprocess.Popen([self.sexpath, "-d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        # Save the file.
        f = open(f'{self.outdir}/default.sex', 'w')
        f.write(out.decode(encoding='UTF-8'))
        f.close()
        return f'{self.outdir}/default.sex'
    
    # Get the SExtractor version associated with the instance.
    def get_version(self):
        """Function to retrieve the SExtractor version.

        Runs SExtractor with no inputs and returns its version number.

        Returns
        -------
        str
            SExtractor version number used to initalise class.
        """
        
        # Run SExtractor with no inputs.
        p = subprocess.Popen([self.sexpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        # Search the outputs for the version number.
        version_match = re.search("[Vv]ersion ([0-9\.])+", err.decode(encoding='UTF-8'))
        
        # Raise error if no version found.
        if version_match is False:
            raise RuntimeError("Could not determine SExctractor version, check the output of running '%s'" % (self.sexpath))
        
        version = str(version_match.group()[8:])
        assert len(version) != 0

        return version	
    
    # Write the output parameters given in the yaml file to a text file which can be given to SExtractor.
    # This will be deleted once SExtractor has finished running. 
    def write_params(self):
        """Write the output parameters to a file.

        Write the output parameters given in the yaml file to a text file which can be given to SExtractor.
        This will be deleted once SExtractor has finished running.

        Returns
        -------
        str
            Path to the parameter file.
        """

        parameter_filename = f'{self.outdir}/{os.path.splitext(os.path.basename(self.configfile))[0]}.params'
        f = open(parameter_filename, 'w')
        f.write("\n".join(self.measurement))
        f.write("\n")
        f.close()

        return parameter_filename
    
    # Write the output parameters required for uncertainty estimation.
    def write_uncertainty_params(self, n):
        """Write the uncertainty estimation parameters to a file.

        Parameters
        ----------
        n (int):
            The number of aperture sizes being used in the uncertainty estimation iteration.

        Returns
        -------
        str
            Path to the parameter file.
        """

        parameter_filename = f'{self.outdir}/uncertainty.params'
        f = open(parameter_filename, 'w')
        f.write(f'FLUX_APER({n})')    # Just need the different aperture fluxes.
        f.write("\n")
        f.close()

        return parameter_filename
    

    def convert_to_flux(self, cat):
        """Convert counts in a catalogue to fluxes.

        Uses multiplicative conversion factor to convert from pure counts
        to a flux.

        Parameters
        ----------
        cat (str):
            Path to catalogue file to be converted.
        """

        if self.other['TO_FLUX'] == False:
            return
        
        print('Converting to flux...')

        catalogue = ascii.read(cat)
        # Search for any flux columns and apply conversion.
        for column in catalogue.colnames:
            if 'FLUX' in column:
                catalogue[column] = catalogue[column] * self.other['TO_FLUX']
        catalogue.write(cat, format='ascii', overwrite = True)

        return

    def convert_to_hdf5(self, catalogue, function):
        """Converts an ascii catalogue to HDF5.

        Parameters
        ----------
        catalogue (str):
            Path to catalogue file to be converted.
        run (str):
            The type of SExtractor run used to generate the catalogue.
        """

        print('Saving to HDF5...')

        # Read the ascii catalogue.
        cat = Table.read(catalogue, format='ascii')

        # Create HDF5 file with the same name.
        with h5py.File(f'{catalogue[:-3]}hdf5', 'w') as f:

            # Add contents to a "photometry" group.
            f.create_group('photometry')
            for column in cat.colnames:
                f[f'photometry/{column}'] = cat[column]

            # Add config information from the ".yml" file and class instance to a "config" group.
            f.create_group('config')
            for key in self.config:
                f[f'config/config/{key}'] = self.config[key]
            f['config/config/FILE'] = self.sexfile
            f['config/config/FUNCTION'] = function    # Indicate the type of SExtractor run. 

            f['config/measurement'] = self.measurement

            for key in self.uncertainty:
                f[f'config/uncertainty/{key}'] = self.uncertainty[key]

            for key in self.other:
                f[f'config/other/{key}'] = self.other[key]

        # Delete the original ascii catalogue.
        os.remove(catalogue)
        return
    
    def run_SExtractor(self, basecmd, imgconfig):
        """Passes a command to SExtractor.

        Parameters
        ----------
        basecmd (str):
            String containing the base (file, image, weight) command line arguments.
        imgconfig (dict):
            Additional arguments from the config file to be added.
        """

        SEcmd = copy.deepcopy(basecmd)

        # Add parameters given in the config file to the base command.
        for (key, value) in imgconfig.items():
            SEcmd.append("-"+str(key))
            SEcmd.append(str(value).replace(' ',''))

        # Run SExtractor and print the outputs.
        p = subprocess.Popen(SEcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for line in p.stderr:
            print(line.decode(encoding="UTF-8"))
        out, err = p.communicate()

        return
    
    def get_WCS_coordinates(self, catalogue, detection_filename):

        if self.other['GET_WCS'] == False:
            return
        
        print('Getting WCS coordinates...')

        detection = fits.open(detection_filename)
        header = detection[0].header
        detection.close()

        wcs = WCS(header)

        cat = Table.read(catalogue, format='ascii')

        coordinates = pixel_to_skycoord(cat['X_IMAGE'], cat['Y_IMAGE'], wcs)

        cat['RA'] = coordinates.ra.degree
        cat['DEC'] = coordinates.dec.degree

        cat.write(catalogue, format='ascii', overwrite = True)

        return

    def measure_uncertainty(self, sci_filename, err_filename, seg_filename, imgconfig):
        """Perform empirical uncertainty estimation based on Finkelstein+23.

        Parameters
        ----------
        sci_filename (str):
            Path to science image to measure uncertainty for.
        err_filename (str):
            Path to an associated error/weight image.
        seg_filename (str):
            Path to the segmentation file produced by SExtractor.
        imgconfig (object):
            The copy of the configuration parameters specific to this image.
        """

        print('\nBeginning uncertainty estimation:')

        # Open the image files.
        sci_image = fits.open(sci_filename)
        sci = sci_image[0].data
        sci_image.close()

        err_image = fits.open(err_filename)
        err = err_image[0].data
        err_image.close()

        seg_image = fits.open(seg_filename)
        seg = seg_image[0].data
        seg_image.close()

        # Create a copy of the configuration parameters specific to the image.
        errconfig = copy.deepcopy(self.config)

        # Set some values to those appropriate for uncertainty estimation.
        errconfig['DETECT_MINAREA'] = 1
        errconfig['DETECT_THRESH'] = 1E-12
        errconfig['FILTER'] = 'N'
        errconfig['CLEAN'] = 'N'
        errconfig['MASK_TYPE'] = 'NONE'
        errconfig['BACK_TYPE'] = 'MANUAL'
        errconfig['BACK_VALUE'] = 0.0
        errconfig['CHECKIMAGE_TYPE'] = 'NONE'

        # Mask positive areas of the segmentation map and invalid areas in the error map.
        mask = (seg > 0)+np.isnan(err)+(err<0)+(err>1000)

        # Seperate the radii into small and large components based on the median value.
        radii = np.array(self.uncertainty['RADII'])
        smaller = radii < np.median(radii)
        larger = radii >= np.median(radii)

        print('Getting aperture locations...')

        # Get locations of first iteration of smaller apertures.
        if self.uncertainty['D_SMALL'] == False:
            depth = ImageDepth(max(radii[smaller]), nsigma=1.0, napers=self.uncertainty['N_SMALL'], 
                            niters=1, overlap=False, overlap_maxiters=100000)
        else:
            depth = ImageDepth(self.uncertainty['D_SMALL'], nsigma=1.0, napers=self.uncertainty['N_SMALL'], 
                            niters=1, overlap=False, overlap_maxiters=100000)           
        limits = depth(sci, mask)
        small = depth.apertures[0].positions
        print(f'Placed {int(depth.napers_used)} small apertures.')

        # Save image showing aperture locations.
        if self.uncertainty['SAVE_FIGS'] == True:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            norm = simple_norm(sci, 'sqrt', percent=99.)
            ax.imshow(sci, norm=norm)
            color = 'red'
            depth.apertures[0].plot(ax, color=color)
            plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95,wspace=0.15)
            plt.savefig(f'{imgconfig["CATALOG_NAME"][:-4]}_small.png')
            fig.clear()

        # Get locations of second iteration larger apertures.
        if self.uncertainty['D_LARGE'] == False:
            depth = ImageDepth(max(radii), nsigma=1.0, napers=self.uncertainty['N_LARGE'], 
                            niters=1, overlap=False,overlap_maxiters=100000)
        else:
            depth = ImageDepth(self.uncertainty['D_LARGE'], nsigma=1.0, napers=self.uncertainty['N_LARGE'], 
                            niters=1, overlap=False,overlap_maxiters=100000)        
        limits = depth(sci, mask)
        large = depth.apertures[0].positions
        print(f'Placed {int(depth.napers_used)} large apertures.')

        # Save image showing aperture locations.
        if self.uncertainty['SAVE_FIGS'] == True:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            norm = simple_norm(sci, 'sqrt', percent=99.)
            ax.imshow(sci, norm=norm)
            color = 'red'
            depth.apertures[0].plot(ax, color=color)
            plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95,wspace=0.15)
            plt.savefig(f'{imgconfig["CATALOG_NAME"][:-4]}_large.png')
            fig.clear()

        # Create a new detection image that has value 1 at the aperture centres, zero everywhere else.
        sci_s = np.zeros(sci.shape)
        sci_l = np.zeros(sci.shape)
        for i in np.round(small).astype(int):
            sci_s[i[1],i[0]] = 1
        for i in np.round(large).astype(int):
            sci_l[i[1],i[0]] = 1

        small_filename = f'{self.outdir}/small_apertures.temp.fits'
        large_filename = f'{self.outdir}/large_apertures.temp.fits'

        # Copy the segementation image to retain header information.
        os.system(f'cp {seg_filename} {small_filename}')
        os.system(f'cp {seg_filename} {large_filename}')

        # Overwrite image with new image.
        small_locations = fits.open(small_filename)
        large_locations = fits.open(large_filename)
        small_locations[0].data = sci_s
        large_locations[0].data = sci_l

        # Save new image to file.
        small_locations.writeto(small_filename, overwrite= True)
        large_locations.writeto(large_filename, overwrite= True)
        small_locations.close()
        large_locations.close()

		# Build the command line arguments for small and large aperture runs.
        smallcmd = [self.sexpath, "-c", self.sexfile, small_filename, sci_filename,'-WEIGHT_IMAGE', err_filename]
        largecmd = [self.sexpath, "-c", self.sexfile, large_filename, sci_filename, '-WEIGHT_IMAGE', err_filename]

        # Add the correct name and aperture diameters to the command line arguments for the small run.
        errconfig['CATALOG_NAME'] = f'{self.outdir}/small_apertures.temp.cat'
        apertures = ''
        for radius in radii[smaller]:
            apertures += str(round(radius*2,2))+','
        apertures = apertures[:-1]
        errconfig['PHOT_APERTURES'] = apertures

        # Write the output parameters to a text file which can be given to SExtractor.
        parameter_filename = self.write_uncertainty_params(sum(smaller))
        errconfig['PARAMETERS_NAME'] = parameter_filename

        print('Running SExtractor on the small apertures...')
        self.run_SExtractor(smallcmd, errconfig)

        # Remove temporary parameter file and detection image.
        os.remove(parameter_filename)
        os.remove(small_filename)

        # Add the correct name and aperture diameters to the command line arguments for the large run.
        errconfig['CATALOG_NAME'] = f'{self.outdir}/large_apertures.temp.cat'
        apertures = ''
        for radius in radii[larger]:
            apertures += str(round(radius*2,2))+','
        apertures = apertures[:-1]
        errconfig['PHOT_APERTURES'] = apertures

        # Write the output parameters to a text file which can be given to SExtractor.
        parameter_filename = self.write_uncertainty_params(sum(larger))
        errconfig['PARAMETERS_NAME'] = parameter_filename

        print('Running SExtractor on the large apertures...')
        self.run_SExtractor(largecmd, errconfig)

        # Remove temporary parameter file and detection image.
        os.remove(parameter_filename)
        os.remove(large_filename)

        # Read in the catalogues with the different aperture measurements.
        small = ascii.read(f'{self.outdir}/small_apertures.temp.cat')
        large = ascii.read(f'{self.outdir}/large_apertures.temp.cat')

        # Calaculate the median-absolute-deviation noise for each aperture size.
        # Factor 1.48 converts to Gaussian-like standard deviation.
        medians = []
        s = (small['FLUX_APER'] != 0)
        for column in small.colnames:
            medians.append(median_abs_deviation(small[column][s], nan_policy='omit')*1.48)
        s = (large['FLUX_APER'] != 0)
        for column in large.colnames:
            medians.append(median_abs_deviation(large[column][s], nan_policy='omit')*1.48)

        # Defining the model to fit. 
        # a,b,c,d are the free parameters.
        sig1 = sigma_clipped_stats(sci, mask)[2]    # The sigma-clipped sigma-clipped  
                                                    # standard deviation of all non-object pixels
        Npix = np.pi*(radii**2)    # The number of pixels in each aperture.

        def model(theta, Npix=Npix):
            a,b,c,d = theta
            return sig1*(((a/1E10)*(Npix**b))+(c*(Npix**(d/1E1))))
        
        # Using a chi2 log-likelihood function.
        def lnlike(theta, x, y, yerr):
            return -0.5 * np.sum(((y - model(theta, x))/yerr) ** 2)
        
        # Requiring all free parameters be > 0.
        def lnprior(theta):
            a, b, c, d = theta
            if -1e6<a<1e6 and -1e6<b<1e6 and -1e6<c<1e6 and -1e6<d>1e6:
                return 0.0
            return -np.inf
        
        # Set up the MCMC.
        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)
        
        Merr = 0.05*np.array(medians) # Adding a 5% percentage error to the median noise values.
                                         # This will weight the fit towards the more common smaller radii.

        data = (Npix, medians,Merr)
        nwalkers = self.uncertainty['WALKERS'] # The number of walkers to use.
        niter = self.uncertainty['ITER'] # The number of iterations.
        initial = np.array(self.uncertainty['INITIAL']) # The inital assumptions for the parameter values.

        ndim = len(initial)
        p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)] # Step methodology.
        
        # Begin the MCMC
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
        print('Running MCMC burn-in...')
        p0, _, _ = sampler.run_mcmc(p0, 100)
        sampler.reset()
        print('Running MCMC production...')
        pos, prob, state = sampler.run_mcmc(p0, niter)

        # Get most likely parameter values.
        samples = sampler.flatchain
        theta_max  = samples[np.argmax(sampler.flatlnprobability)]

        # Median error value of the whole map.
        median_err = np.median(err[~np.isnan(err)])

        # Read original catalogue produced by SExtractor.
        cat = ascii.read(imgconfig['CATALOG_NAME'])

        # Calculate the area of each KRON and circular aperture and 
        # compare to fit to get noise value. Scale by the ratio of error map value
        # at the centre of the source to the median to get a more local value.

        print('Calculating noise for catalogue sources...')

        for column in cat.colnames:
            if column == 'FLUX_AUTO':
                cat['FLUX_AUTO_AREA'] = np.pi * cat['A_IMAGE'] * cat['B_IMAGE'] * np.power(cat['KRON_RADIUS'],2)
                cat['FLUXERR_AUTO'] = model(theta_max,cat['FLUX_AUTO_AREA'])*(err[cat['Y_IMAGE'].astype(int), cat['X_IMAGE'].astype(int)]/median_err)
                cat.remove_column('FLUX_AUTO_AREA')
            if column == 'FLUX_APER':
                cat[f'FLUXAPER_AREA'] = np.pi * np.power(radii[0],2)
                cat['FLUXERR_APER'] = model(theta_max, cat[f'FLUXAPER_AREA'])*(err[cat['Y_IMAGE'].astype(int), cat['X_IMAGE'].astype(int)]/median_err)
                cat.remove_column('FLUXAPER_AREA')
            if 'FLUX_APER_' in column:
                aper = int(column.split('FLUX_APER_')[1])
                cat[f'FLUXAPER_{aper}_AREA'] = np.pi * np.power(radii[aper],2)
                cat[f'FLUXERR_APER_{aper}'] = model(theta_max, cat[f'FLUXAPER_{aper}_AREA'])*(err[cat['Y_IMAGE'].astype(int), cat['X_IMAGE'].astype(int)]/median_err)
                cat.remove_column(f'FLUXAPER_{aper}_AREA')
        cat.write(imgconfig['CATALOG_NAME'], format='ascii', overwrite = True)

        # Save a plot of noise vs aperture size.
        if self.uncertainty['SAVE_FIGS'] == True:

            x = np.linspace(0, max(Npix),10000)
            fig = plt.figure()
            ax = plt.gca()
            plt.scatter(np.sqrt(Npix),medians,s=15, color = 'white', edgecolors='blue', alpha = 0.8)
            plt.plot(np.sqrt(x), model(theta_max, x), color = 'grey', linestyle = '--', linewidth = 1)  
            plt.xlabel('sqrt(Number of pixels in aperture)')
            plt.ylabel('Noise in aperture [counts]')
            plt.text(min(x)*0.001, max(medians)*0.98, f'{os.path.splitext(os.path.basename(imgconfig["CATALOG_NAME"]))[0].split(".")[0]}')
            plt.minorticks_on()
            ax.tick_params(axis='both', direction='in', which = 'both')
            plt.savefig(f'{imgconfig["CATALOG_NAME"].split(".")[0]}_noise.png')
            fig.clear()

        # Remove remaining aperture files and the segmentation map.
        os.remove(f'{self.outdir}/small_apertures.temp.cat')
        os.remove(f'{self.outdir}/large_apertures.temp.cat')
        os.remove(seg_filename)

        print('Done. \n')

        return

    def SExtract(self, image, weight = None):
        """Runs SExtractor in any of its standard modes.

        Performs SExtraction using parameters defined in the config file 
        in either single or image mode, with or without weights. 
        Will generate a HDF5 file named based on the measurement image with the photometry,
        and any CHECK_IMAGES requested in the config.

        Parameters
        ----------
        image (str, list):
            If a string the path of the image to perfrom extraction on.
            If a list the path to the detection image as the zeroth entry and measurement as the second.
        weight (None, str, list):
            If None, perform extraction without weights.
            If string, weight image for single image mode.
            If list, detection image weight as the zeroth entry, measurement image weight as the second.
        """

        print('Standard SExtraction')
        print('-'*len('Standard SExtraction'))
		
        # Create a copy of the configuration parameters specific to the image.
        imgconfig = copy.deepcopy(self.config)

		# Set the catalogue name based on the measurement image.
        if type(image) == list:
            imgconfig['CATALOG_NAME'] = f'{self.outdir}/{os.path.splitext(os.path.basename(image[1]))[0]}.cat'
            print(f'SExtracting {os.path.splitext(os.path.basename(image[1]))[0]}...')
        else:
            imgconfig['CATALOG_NAME'] = f'{self.outdir}/{os.path.splitext(os.path.basename(image))[0]}.cat'
            print(f'SExtracting {os.path.splitext(os.path.basename(image))[0]}...')

        # Add quantities needed for uncertainty estimation to ".sex" if required.
        if self.uncertainty['EMPIRICAL'] == True:
            for i in ['A_IMAGE','B_IMAGE','KRON_RADIUS', 'X_IMAGE', 'Y_IMAGE']:
                if i not in self.measurement:
                    self.measurement.append(i)
            imgconfig['CHECKIMAGE_TYPE'] = 'SEGMENTATION'
            imgconfig['CHECKIMAGE_NAME'] = f'{imgconfig["CATALOG_NAME"][:-4]}.temp_seg.fits'

        if self.other['GET_WCS'] == True:
            for i in ['X_IMAGE', 'Y_IMAGE']:
                if i not in self.measurement:
                    self.measurement.append(i)
        
        # Write the output parameters to a text file which can be given to SExtractor.
        parameter_filename = self.write_params()
        imgconfig['PARAMETERS_NAME'] = parameter_filename

		# Build the command line arguments depending on the mode.
        if (type(image) == list) and (type(weight) == list):
            basecmd = [self.sexpath, "-c", self.sexfile, image[0], image[1], '-WEIGHT_IMAGE', f'{weight[0]},{weight[1]}']
        if (type(image) == list) and (type(weight) == type(None)):
            basecmd = [self.sexpath, "-c", self.sexfile, image[0], image[1]]
        if (type(image) == str) and (type(weight) == type(None)):
            basecmd = [self.sexpath, "-c", self.sexfile, image]
        if (type(image) == str) and (type(weight) == str):
            basecmd = [self.sexpath, "-c", self.sexfile, image, '-WEIGHT_IMAGE', weight]

        self.run_SExtractor(basecmd, imgconfig)

        # Remove temporary parameter file.
        os.remove(parameter_filename)

        # Begin uncertainty estimation if needed.
        if self.uncertainty['EMPIRICAL'] == True:
            if (type(image) == str):
                self.measure_uncertainty(sci_filename=image, err_filename=weight, seg_filename=imgconfig['CHECKIMAGE_NAME'], imgconfig=imgconfig)
            else:
                self.measure_uncertainty(sci_filename=image[1], err_filename=weight[1], seg_filename=imgconfig['CHECKIMAGE_NAME'], imgconfig=imgconfig)

        # Convert to flux if required.
        self.convert_to_flux(imgconfig['CATALOG_NAME'])

        # Get WCS coordinates if required.
        if (type(image) == str):
            self.get_WCS_coordinates(catalogue=imgconfig['CATALOG_NAME'], detection_filename=image)
        else:
            self.get_WCS_coordinates(imgconfig['CATALOG_NAME'], detection_filename=image[0])

        # Convert catalogue to HDF5.
        self.convert_to_hdf5(imgconfig['CATALOG_NAME'], function = 'SExtract')

        print(f'Completed SExtraction and saved to {imgconfig["CATALOG_NAME"][:-4]}.hdf5 \n')

        return
    
    def psf_corrected_SExtract(self, detection, images, weights = None):
        """Runs SExtractor in two image mode and perform a PSF correction.

        Performs SExtraction using parameters defined in the config file on:
            - A uncorrected image in a given band
            - An image in a second band.
            - The image in the second band matched to the PSF of the first.
        A PSF correction is applied to the first image based on the 
        ratio in flux between the two additional images.

        Parameters
        ----------
        detection (str):
            Path to the detection image.
        images (list):
            List containing the uncorrected image and then two images in a different band
            the first with its default PSF the second matched to that of the first. In that order.
        """

        print('PSF Corrected SExtraction')
        print('-'*len('PSF Corrected SExtraction'))

        # Create a copy of the configuration parameters specific to the image.
        imgconfig = copy.deepcopy(self.config)

        # Add quantities needed for uncertainty estimation to ".sex" if required.
        if self.uncertainty['EMPIRICAL'] == True:
            for i in ['A_IMAGE','B_IMAGE','KRON_RADIUS', 'X_IMAGE', 'Y_IMAGE']:
                if i not in self.measurement:
                    self.measurement.append(i)
            imgconfig['CHECKIMAGE_TYPE'] = 'SEGMENTATION'
            imgconfig['CHECKIMAGE_NAME'] = f'{self.outdir}/{os.path.splitext(os.path.basename(images[0]))[0]}.temp_seg.fits'

        if self.other['GET_WCS'] == True:
            for i in ['X_IMAGE', 'Y_IMAGE']:
                if i not in self.measurement:
                    self.measurement.append(i)

        # Write the output parameters to a text file which can be given to SExtractor.
        parameter_filename = self.write_params()
        imgconfig['PARAMETERS_NAME'] = parameter_filename
		
		# Build the command line arguments depending of whether weights were given.
        if (type(weights) == None):
            uncorrected = [self.sexpath, "-c", self.sexfile, detection, images[0]]
            unmatched = [self.sexpath, "-c", self.sexfile, detection, images[1]]
            matched = [self.sexpath, "-c", self.sexfile, detection, images[2]]
        if (type(weights) == list):
            uncorrected = [self.sexpath, "-c", self.sexfile, detection, images[0], '-WEIGHT_IMAGE', f'{weights[0]},{weights[1]}']
            unmatched = [self.sexpath, "-c", self.sexfile, detection, images[1], '-WEIGHT_IMAGE', f'{weights[0]},{weights[2]}']
            matched = [self.sexpath, "-c", self.sexfile, detection, images[2], '-WEIGHT_IMAGE', f'{weights[0]},{weights[2]}']

        cat_filenames = []
		# Loop over each image and 
        for i, basecmd in enumerate([uncorrected, unmatched, matched]):
            # change name,
            imgconfig['CATALOG_NAME'] = f'{self.outdir}/{os.path.splitext(os.path.basename(images[i]))[0]}.temp.cat'
            cat_filenames.append(imgconfig['CATALOG_NAME'])

            print(f'SExtracting {os.path.splitext(os.path.basename(images[i]))[0]}...')

            self.run_SExtractor(basecmd, imgconfig)

        # Remove temporary parameter file.
        os.remove(parameter_filename)

        # Begin uncertainty estimation if needed.
        imgconfig['CATALOG_NAME'] = cat_filenames[0]
        if self.uncertainty['EMPIRICAL'] == True:
            self.measure_uncertainty(sci_filename=images[0], err_filename=weights[1], seg_filename=imgconfig['CHECKIMAGE_NAME'], imgconfig=imgconfig)

        print('Applying correction...')
        # Read in the created catalogues.
        uncorrected = ascii.read(cat_filenames[0])
        unmatched = ascii.read(cat_filenames[1])
        matched = ascii.read(cat_filenames[2])

        # Calculate and apply the psf correction to each flux bands count and noise.
        for column in uncorrected.colnames:
            if ('FLUX' in column) and ('ERR' not in column):
                uncorrected[column] = uncorrected[column] * (unmatched[column]/matched[column])
                uncorrected[f'FLUXERR{column.split("FLUX")[1]}'] = uncorrected[f'FLUXERR{column.split("FLUX")[1]}'] * (unmatched[column]/matched[column])

        # Write to a new corrected catalogue and remove .temp catalogues.
        uncorrected.write(f'{imgconfig["CATALOG_NAME"][:-9]}_psfcorrected.cat', format='ascii', overwrite = True)
        for filename in cat_filenames:
            os.remove(filename)

        # Convert counts to flux if needed.
        self.convert_to_flux(f'{imgconfig["CATALOG_NAME"][:-9]}_psfcorrected.cat')

        # Get WCS coordinates if required.
        self.get_WCS_coordinates(catalogue=f'{imgconfig["CATALOG_NAME"][:-9]}_psfcorrected.cat', detection_filename=detection)

        # Convert catalogue to HDF5.
        self.convert_to_hdf5(f'{imgconfig["CATALOG_NAME"][:-9]}_psfcorrected.cat', function = 'psf_corrected_SExtract')

        print(f'Completed SExtraction and saved to {imgconfig["CATALOG_NAME"][:-9]}_psfcorrected.hdf5 \n')

        return