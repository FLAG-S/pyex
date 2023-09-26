import os
import yaml

import subprocess
import re
import copy

from astropy.io import ascii, fits
from astropy.visualization import simple_norm

from photutils.utils import ImageDepth

import numpy as np

from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation

class pyex():

    def __init__(self, config, sexpath ='sex', sexfile=None, outdir=None, verbose = True):

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

        # Read the pyex configuration file and split into configuration and parameter components.
        self.configfile = config
        with open(config, 'r') as file:
            yml = yaml.safe_load_all(file)
            content = []
            for entry in yml:
                content.append(entry)
            self.config, self.measurement, self.uncertainty = content

        # Print the parameters used to initalise the class for checking.
        if verbose == True:
            print(f'Initalised a pyex class with: \n Config parameters: {self.config} \n Outputs: {self.measurement} \n')

    # Generate the default SExtractor configuration file if none given.
    def generate_default(self):
        # Pipe the SExtractor output.
        p = subprocess.Popen([self.sexpath, "-d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        # Save the file.
        f = open('default.sex', 'w')
        f.write(out.decode(encoding='UTF-8'))
        f.close()
        return 'default.sex'
    
    # Get the SExtractor version associated with the instance.
    def get_version(self):
        p = subprocess.Popen([self.sexpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        version_match = re.search("[Vv]ersion ([0-9\.])+", err.decode(encoding='UTF-8'))
        if version_match is False:
            raise RuntimeError("Could not determine SExctractor version, check the output of running '%s'" % (self.sexpath))
        version = str(version_match.group()[8:])
        assert len(version) != 0
        return version	
    
    # Write the output parameters given in the yaml file to a text file which can be given to SExtractor.
    # This will be deleted once SExtractor has finished running. 
    def write_params(self):		

        parameter_filename = f'{self.outdir}/{os.path.splitext(os.path.basename(self.configfile))[0]}.params'
        f = open(parameter_filename, 'w')
        f.write("\n".join(self.measurement))
        f.write("\n")
        f.close()
        return parameter_filename
    
    def write_uncertainty_params(self, n):		

        parameter_filename = f'{self.outdir}/uncertainty.params'
        f = open(parameter_filename, 'w')
        f.write(f'FLUX_APER({n})')
        f.write("\n")
        f.close()
        return parameter_filename
    
    def convert_to_flux(self, cat, conversion):
        catalogue = ascii.read(cat)
        for column in catalogue.colnames:
            if 'FLUX' in column:
                catalogue[column] = catalogue[column] * conversion
        catalogue.write(cat, format='ascii', overwrite = True)

    def measure_uncertainty(self, sci_filename, err_filename, seg_filename, imgconfig):

        sci = fits.open(sci_filename)[0].data
        err = fits.open(err_filename)[0].data
        seg = fits.open(seg_filename)[0].data

        # Create a copy of the configuration parameters specific to the image.
        appconfig = copy.deepcopy(self.config)

        # Set some values to those appropriate for uncertainty estimation.
        appconfig['DETECT_MINAREA'] = 1
        appconfig['DETECT_THRESH'] = 1E-6
        appconfig['FILTER'] = 'N'
        appconfig['CLEAN'] = 'N'
        appconfig['MASK_TYPE'] = 'NONE'
        appconfig['BACK_TYPE'] = 'MANUAL'
        appconfig['BACK_VALUE'] = 0.0
        appconfig['WEIGHT_TYPE'] = 'NONE'
        
        # Mask positive areas of the segmentation map and NaNs in the error map.
        mask = (seg > 0)+np.isnan(err)

        radii = np.array(self.uncertainty['RADII'])

        smaller = radii < np.median(radii)
        larger = radii >= np.median(radii)

        # First iteration of smaller apertures.
        depth = ImageDepth(max(radii[smaller]), nsigma=1.0, napers=self.uncertainty['N_SMALL'], niters=1, overlap=False)
        limits = depth(sci, mask)
        small = depth.apertures[0].positions

        if self.uncertainty['SAVE_FIGS'] == True:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            norm = simple_norm(sci, 'sqrt', percent=99.)
            ax.imshow(sci, norm=norm)
            color = 'red'
            depth.apertures[0].plot(ax, color=color)
            plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95,wspace=0.15)
            plt.savefig(f'{self.outdir}/{os.path.splitext(os.path.basename(sci_filename))[0]}_small.png')
            plt.close(fig)

        # Larger aperture locations.
        depth = ImageDepth(max(radii), nsigma=1.0, napers=self.uncertainty['N_LARGE'], niters=1, overlap=False)
        limits = depth(sci, mask)
        large = depth.apertures[0].positions

        if self.uncertainty['SAVE_FIGS'] == True:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            norm = simple_norm(sci, 'sqrt', percent=99.)
            ax.imshow(sci, norm=norm)
            color = 'red'
            depth.apertures[0].plot(ax, color=color)
            plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95,wspace=0.15)
            plt.savefig(f'{self.outdir}/{os.path.splitext(os.path.basename(sci_filename))[0]}_large.png')
            plt.close(fig)


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

        small_locations = fits.open(small_filename)
        large_locations = fits.open(large_filename)

        # Overwrite image with new image.
        small_locations[0].data = sci_s
        large_locations[0].data = sci_l

        small_locations.writeto(small_filename, overwrite= True)
        large_locations.writeto(large_filename, overwrite= True)

		# Build the command line arguments.
        small_run = [self.sexpath, "-c", self.sexfile, small_filename, sci_filename]
        large_run = [self.sexpath, "-c", self.sexfile, large_filename, sci_filename]

		# add parameters given in the config file.
        appconfig['CATALOG_NAME'] = f'{self.outdir}/small_apertures.temp.cat'
        apertures = ''
        for radius in radii[smaller]:
            apertures += str(round(radius*2,2))+','
        apertures = apertures[:-1]
        appconfig['PHOT_APERTURES'] = apertures

        # Write the output parameters to a text file which can be given to SExtractor.
        parameter_filename = self.write_uncertainty_params(sum(smaller))

        # Set the path to the parameter file.
        appconfig['PARAMETERS_NAME'] = parameter_filename

        for (key, value) in appconfig.items():
            small_run.append("-"+str(key))
            small_run.append(str(value).replace(' ',''))

        # Run SExtractor and print the outputs.
        p = subprocess.Popen(small_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        for line in p.stderr:
            print(line.decode(encoding="UTF-8"))
        out, err_ = p.communicate()

        # Remove temporary parameter file.
        os.remove(parameter_filename)
        os.remove(small_filename)

		# add parameters given in the config file.
        appconfig['CATALOG_NAME'] = f'{self.outdir}/large_apertures.temp.cat'
        apertures = ''
        for radius in radii[larger]:
            apertures += str(round(radius*2,2))+','
        apertures = apertures[:-1]
        appconfig['PHOT_APERTURES'] = apertures

        # Write the output parameters to a text file which can be given to SExtractor.
        parameter_filename = self.write_uncertainty_params(sum(larger))

        # Set the path to the parameter file.
        appconfig['PARAMETERS_NAME'] = parameter_filename

        for (key, value) in appconfig.items():
            large_run.append("-"+str(key))
            large_run.append(str(value).replace(' ',''))

        # Run SExtractor and print the outputs.
        p = subprocess.Popen(large_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        for line in p.stderr:
            print(line.decode(encoding="UTF-8"))
        out, err_ = p.communicate()

        # Remove temporary parameter file.
        os.remove(parameter_filename)
        os.remove(large_filename)

        small = ascii.read(f'{self.outdir}/small_apertures.temp.cat')
        large = ascii.read(f'{self.outdir}/large_apertures.temp.cat')

        medians = []
        s = (small['FLUX_APER'] != 0)
        for column in small.colnames:
            medians.append(median_abs_deviation(small[column][s], nan_policy='omit')*1.48)

        s = (large['FLUX_APER'] != 0)
        for column in large.colnames:
            medians.append(median_abs_deviation(large[column][s], nan_policy='omit')*1.48)

        mask = (seg > 0)+np.isnan(err) 

        sig1 = sigma_clipped_stats(sci, mask)[2]

        Npix = np.pi*(radii**2)

        # Fit function to the data.
        def func(N, a, b, c, d):
            return sig1*((a*np.power(N,b))+(c*np.power(N,d)))

        popt, pcov = curve_fit(func, Npix, medians, maxfev = 5000)

        x = np.linspace(0, max(Npix),10000)

        median_err = np.median(err[np.invert(np.isnan(err))])

        cat = ascii.read(imgconfig['CATALOG_NAME'])
        for column in cat.colnames:
            if column == 'FLUX_AUTO':
                cat['FLUX_AUTO_AREA'] = np.pi * cat['A_IMAGE'] * cat['B_IMAGE'] * np.power(cat['KRON_RADIUS'],2)
                cat['FLUXERR_AUTO'] = (sig1 * ((popt[0]*np.power(cat['FLUX_AUTO_AREA'],popt[1])) + (popt[2]*np.power(cat['FLUX_AUTO_AREA'],popt[3]))))*(err[cat['Y_IMAGE'].astype(int), cat['X_IMAGE'].astype(int)]/median_err)
                cat.remove_column('FLUX_AUTO_AREA')
            if column == 'FLUX_APER':
                cat[f'FLUXAPER_AREA'] = np.pi * np.power(radii[0],2)
                cat['FLUXERR_APER'] = (sig1 * ((popt[0]*np.power(cat['FLUXAPER_AREA'],popt[1])) + (popt[2]*np.power(cat['FLUXAPER_AREA'],popt[3]))))*(err[cat['Y_IMAGE'].astype(int), cat['X_IMAGE'].astype(int)]/median_err)
                cat.remove_column('FLUXAPER_AREA')
            if 'FLUX_APER_' in column:
                aper = int(column.split('FLUX_APER')[1])
                cat[f'FLUXAPER_{aper}_AREA'] = np.pi * np.power(radii[aper],2)
                cat[f'FLUXERR_APER_{aper}'] = (sig1 * ((popt[0]*np.power(cat[f'FLUXAPER_{aper}_AREA'],popt[1])) + (popt[2]*np.power(cat[f'FLUXAPER_{aper}_AREA'],popt[3]))))*(err[cat['Y_IMAGE'].astype(int), cat['X_IMAGE'].astype(int)]/median_err)
                cat.remove_column(f'FLUXAPER_{aper}_AREA')
        cat.write(imgconfig['CATALOG_NAME'], format='ascii', overwrite = True)

        if self.uncertainty['SAVE_FIGS'] == True:
            fig = plt.figure()
            plt.scatter(np.sqrt(Npix),medians, label = 'Median values')
            plt.plot(np.sqrt(x), func(x, popt[0], popt[1], popt[2], popt[3]), label = '4-parameter fit')
            plt.xlabel('sqrt(Number of pixels)')
            plt.ylabel('Noise in aperture')
            plt.legend()
            plt.savefig(f'{self.outdir}/{os.path.splitext(os.path.basename(sci_filename))[0]}_noise.png')
            plt.close(fig)

        os.remove(f'{self.outdir}/small_apertures.temp.cat')
        os.remove(f'{self.outdir}/large_apertures.temp.cat')
        os.remove(seg_filename)

        return

    # Run SExtractor.
    def run_SExtractor(self, image, weight = None, to_flux = None):
		
        # Create a copy of the configuration parameters specific to the image.
        imgconfig = copy.deepcopy(self.config)

        if self.uncertainty['EMPIRICAL'] == True:
            for i in ['A_IMAGE','B_IMAGE','KRON_RADIUS', 'X_IMAGE', 'Y_IMAGE']:
                if i not in self.measurement:
                    self.measurement.append(i)

        # Write the output parameters to a text file which can be given to SExtractor.
        parameter_filename = self.write_params()

        # Set the path to the parameter file.
        imgconfig['PARAMETERS_NAME'] = parameter_filename

		# Set the catalog name based on the measurement image.
        if type(image) == list:
            imgconfig['CATALOG_NAME'] = f'{self.outdir}/{os.path.splitext(os.path.basename(image[1]))[0]}.cat'
        else:
            imgconfig['CATALOG_NAME'] = f'{self.outdir}/{os.path.splitext(os.path.basename(image))[0]}.cat'
		
		# Build the command line arguments.
        if (type(image) == list) and (type(weight) == list):
            popencmd = [self.sexpath, "-c", self.sexfile, image[0], image[1], '-WEIGHT_IMAGE', f'{weight[0]},{weight[1]}']
        if (type(image) == list) and (type(weight) == type(None)):
            popencmd = [self.sexpath, "-c", self.sexfile, image[0], image[1]]
        if (type(image) == str) and (type(weight) == type(None)):
            popencmd = [self.sexpath, "-c", self.sexfile, image]
        if (type(image) == str) and (type(weight) == str):
            popencmd = [self.sexpath, "-c", self.sexfile, image, '-WEIGHT_IMAGE', weight]

        if self.uncertainty['EMPIRICAL'] == True:
            imgconfig['CHECKIMAGE_TYPE'] = 'SEGMENTATION'
            imgconfig['CHECKIMAGE_NAME'] = f'{self.outdir}/{os.path.splitext(os.path.basename(image[0]))[0]}.temp_seg.fits'

		# add parameters given in the config file.
        for (key, value) in imgconfig.items():
            popencmd.append("-"+str(key))
            popencmd.append(str(value).replace(' ',''))

        # Run SExtractor and print the outputs.
        p = subprocess.Popen(popencmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        for line in p.stderr:
            print(line.decode(encoding="UTF-8"))
        out, err = p.communicate()

        # Remove temporary parameter file.
        os.remove(parameter_filename)

        # Apply conversion to flux if required.
        #if to_flux != None:
            #self.convert_to_flux(imgconfig['CATALOG_NAME'], to_flux)

        if self.uncertainty['EMPIRICAL'] == True:
            if (type(image) == str):
                self.measure_uncertainty(sci_filename=image, err_filename=weight, seg_filename=imgconfig['CHECKIMAGE_NAME'], imgconfig=imgconfig)
            else:
                self.measure_uncertainty(sci_filename=image[1], err_filename=weight[1], seg_filename=imgconfig['CHECKIMAGE_NAME'], imgconfig=imgconfig)

        return
    
    def psf_corrected_SExtractor(self, detection, images, weights = None):

        '''Only works in two-image mode'''

        # Create a copy of the configuration parameters specific to the image.
        imgconfig = copy.deepcopy(self.config)

        if self.uncertainty['EMPIRICAL'] == True:
            for i in ['A_IMAGE','B_IMAGE','KRON_RADIUS', 'X_IMAGE', 'Y_IMAGE']:
                if i not in self.measurement:
                    self.measurement.append(i)

        # Write the output parameters to a text file which can be given to SExtractor.
        parameter_filename = self.write_params()

        # Set the path to the parameter file.
        imgconfig['PARAMETERS_NAME'] = parameter_filename
		
		# Build the command line arguments.
        if (type(weights) == None):
            uncorrected = [self.sexpath, "-c", self.sexfile, detection, images[0]]
            unmatched = [self.sexpath, "-c", self.sexfile, detection, images[1]]
            matched = [self.sexpath, "-c", self.sexfile, detection, images[2]]
        if (type(weights) == list):
            uncorrected = [self.sexpath, "-c", self.sexfile, detection, images[0], '-WEIGHT_IMAGE', f'{weights[0]},{weights[1]}']
            unmatched = [self.sexpath, "-c", self.sexfile, detection, images[1], '-WEIGHT_IMAGE', f'{weights[0]},{weights[2]}']
            matched = [self.sexpath, "-c", self.sexfile, detection, images[2], '-WEIGHT_IMAGE', f'{weights[0]},{weights[2]}']

        cat_filenames = []

        if self.uncertainty['EMPIRICAL'] == True:
            imgconfig['CHECKIMAGE_TYPE'] = 'SEGMENTATION'
            imgconfig['CHECKIMAGE_NAME'] = f'{self.outdir}/{os.path.splitext(os.path.basename(images[0]))[0]}.temp_seg.fits'

		# Add parameters given in the config file.
        for i, popencmd in enumerate([unmatched, matched, uncorrected]):
            imgconfig['CATALOG_NAME'] = f'{self.outdir}/{os.path.splitext(os.path.basename(images[i]))[0]}.temp.cat'
            cat_filenames.append(imgconfig['CATALOG_NAME'])
            for (key, value) in imgconfig.items():
                popencmd.append("-"+str(key))
                popencmd.append(str(value).replace(' ',''))
            
            # Run SExtractor on each image.
            p = subprocess.Popen(popencmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for line in p.stderr:
                print(line.decode(encoding="UTF-8"))
            out, err = p.communicate()

        os.remove(parameter_filename)

        if self.uncertainty['EMPIRICAL'] == True:
            self.measure_uncertainty(sci_filename=images[0], err_filename=weights[1], seg_filename=imgconfig['CHECKIMAGE_NAME'], imgconfig=imgconfig)

        # Read in the created catalogues.
        uncorrected = ascii.read(cat_filenames[2])
        unmatched = ascii.read(cat_filenames[0])
        matched = ascii.read(cat_filenames[1])

        # Calculate and apply the psf correction to each flux band.
        for column in uncorrected.colnames:
            if ('FLUX' in column) and ('ERR' not in column):
                uncorrected[column] = uncorrected[column] * (unmatched[column]/matched[column])
                uncorrected[f'FLUXERR{column.split("FLUX")[1]}'] = uncorrected[f'FLUXERR{column.split("FLUX")[1]}'] * (unmatched[column]/matched[column])

        # Write to a new corrected catalogue and remove .temp catalogues.
        uncorrected.write(f'{self.outdir}/{os.path.splitext(os.path.basename(images[0]))[0]}_psfcorrected.cat', format='ascii')
        for filename in cat_filenames:
            os.remove(filename)

        return