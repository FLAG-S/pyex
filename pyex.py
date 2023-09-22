import os
import yaml

import subprocess
import re
import copy

from astropy.io import ascii


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
            self.config, self.measurement = content

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
    
    def convert_to_flux(self, cat, conversion):
        catalogue = ascii.read(cat)
        for column in catalogue.colnames:
            if 'FLUX' in column:
                catalogue[column] = catalogue[column] * conversion
        catalogue.write(cat, format='ascii', overwrite = True)

    # Run SExtractor.
    def run_SExtractor(self, image, weight = None, to_flux = None):
		
        # Create a copy of the configuration parameters specific to the image.
        imgconfig = copy.deepcopy(self.config)

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
        if to_flux != None:
            self.convert_to_flux(imgconfig['CATALOG_NAME'], to_flux)

        return
    
    def psf_corrected_SExtractor(self, detection, images, weights = None, to_flux = None):

        # Create a copy of the configuration parameters specific to the image.
        imgconfig = copy.deepcopy(self.config)

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

		# Add parameters given in the config file.
        for i, popencmd in enumerate([uncorrected, unmatched, matched]):
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

        # Read in the created catalogues.
        uncorrected = ascii.read(cat_filenames[0])
        unmatched = ascii.read(cat_filenames[1])
        matched = ascii.read(cat_filenames[2])

        # Apply flux conversion if required.
        if to_flux != None:
            self.convert_to_flux(cat_filenames[0], to_flux)

        # Calculate and apply the psf correction to each flux band.
        for column in uncorrected.colnames:
            if 'FLUX' in column:
                uncorrected[column] = uncorrected[column] * (unmatched[column]/matched[column])

        # Write to a new corrected catalogue and remove .temp catalogues.
        uncorrected.write(f'{self.outdir}/{os.path.splitext(os.path.basename(images[0]))[0]}_psfcorrected.cat', format='ascii')
        for filename in cat_filenames:
            os.remove(filename)

        return