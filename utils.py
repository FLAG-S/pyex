import os
import h5py
from astropy.table import Table
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

def multiband_catalogue(catalogues, bands, new_catalogue = 'pyex_multiband_catalogue.hdf5', base_band = None):
    """Combine multiple hdf5 catalogues produced by pyex into a multiband catalogue.

    Parameters
    ----------
    catalogues (list):
        A list of paths to the catalogues to be combined.
    bands (list):
        A list of strings corresponding to the bands used to create each catalogue in 'catalogues'.
    new_catalogue (str):
        Path to the new catalogue.
    base_band (None, str):
        The band(s) to use for any non flux measurements. 
        'None' for the first in 'bands', 'all' for measurements from all bands
        or a string corresponding to one of the bands in 'bands'.
    """

    if (base_band not in bands) and (base_band != None) and (base_band != 'all'):
        raise ValueError(f"{base_band} is not in list of bands or 'None' or 'all'.")

    # Create the new catalogue and its photometry group.
    with h5py.File(new_catalogue, 'w') as newcat:
        newcat.create_group('photometry')

        # Iterate over the different catalogues.
        for band, catalogue in zip(bands, catalogues):

            with h5py.File(catalogue, 'r') as cat:

                if 'photometry' in cat.keys():

                    # Collect aperture fluxes into a single array for each source.
                    if 'FLUX_APER' in cat['photometry'].keys():
                        has_error = False
                        all_f = []
                        all_e = []
                        for source in np.arange(len(cat[f'photometry/{list(cat["photometry"].keys())[0]}'])):

                            app_f = []
                            app_f.append(cat['photometry/FLUX_APER'][source])

                            # If errors are included add this to their own single array per source.
                            if 'FLUXERR_APER' in cat['photometry'].keys():
                                app_e = []
                                has_error = True
                                app_e.append(cat['photometry/FLUXERR_APER'][source])

                            # Loop over additional apertures.
                            i = 1
                            while f'FLUX_APER_{i}' in cat['photometry'].keys():
                                app_f.append(cat[f'photometry/FLUX_APER_{i}'][source])
                                if has_error == True:
                                    app_e.append(cat[f'photometry/FLUXERR_APER_{i}'][source])
                                i += 1
                                
                            all_f.append(np.array(app_f))
                            if has_error == True:
                                all_e.append(np.array(app_e))

                        # Add arrays to new file.
                        newcat[f'photometry/FLUX_APER_{band}'] = np.array(all_f)
                        if has_error == True:
                            newcat[f'photometry/FLUXERR_APER_{band}'] = np.array(all_e)
                            
                    # Add any other flux and error measurements to the new catalogue.
                    for key in cat['photometry'].keys():
                        if ('FLUX' in key) and ('APER' not in key):
                            newcat[f'photometry/{key}_{band}'] = cat[f'photometry/{key}'][:]

                    # Add any additional quantities to the catalogue in the specified way.
                    # If no base_band given use values from the first catalogue in the list.
                    if base_band == None:
                        if bands.index(band) == 0:
                            for key in cat['photometry'].keys():
                                if ('FLUX' not in key):
                                    newcat[f'photometry/{key}'] = cat[f'photometry/{key}'][:]
                    # If all is given include values from all the bands (same approach as fluxes).
                    if base_band == 'all':
                        for key in cat['photometry'].keys():
                            if ('FLUX' not in key):
                                newcat[f'photometry/{key}_{band}'] = cat[f'photometry/{key}'][:]
                    # If base_band is a band name, use values from corresponding catalogue.
                    if base_band == band:
                        for key in cat['photometry'].keys():
                            if ('FLUX' not in key):
                                newcat[f'photometry/{key}'] = cat[f'photometry/{key}'][:]
                else:
                    raise Warning('No photometry group found!')

                # Add the configuration information for each of the bands.
                if 'config' in cat.keys():
                    cat.copy('config', newcat, name=f'config_{band}')
                else:
                    raise Warning('No config group found!')
                
    print(f'Created a multiband catalogue and saved to {new_catalogue}')

    return

def multifield_catalogue(catalogues, new_catalogue, numbering = None, replace = False):
    """Combine multiple hdf5 multiband catalogues produced by pyex into a multifield catalogue.

    Parameters
    ----------
    catalogues (list):
        A list of paths to the catalogues to be combined, in pointing number order.
    new_catalogue (str):
        Path to the new catalogue.
    numbering (list):
        Numbering to use to name pointings. If None, use the position of the catalogue in catalogues.
    replace (bool):
        Whether to remove the individual field catalogues.
    """

    # Create the new catalogue.
    with h5py.File(new_catalogue, 'a') as new_cat: 

        # Iterate over the multiband catalogues.
        for catalogue in catalogues:
            with h5py.File(catalogue, 'r+') as cat:

                cat_group = cat['photometry']

                # Create a dataset indicating the source pointing.
                if numbering == None:
                    cat_group['FIELD'] = [catalogues.index(catalogue)+1]*len(cat[f'photometry/{list(cat["photometry"].keys())[0]}'])
                else:
                    cat_group['FIELD'] = [numbering[catalogues.index(catalogue)]]*len(cat[f'photometry/{list(cat["photometry"].keys())[0]}'])
                
                # Create the photometry group in the new catalogue.
                if 'photometry' in new_cat:
                    new_cat_group = new_cat['photometry']
                else:
                    new_cat_group = new_cat.create_group('photometry')

                # Iterate over datasets within the group
                for dataset_name, dataset in cat_group.items():
                    cat_data = dataset[()]

                    # If the dataset already exists in the destination group, concatenate it
                    if dataset_name in new_cat_group:
                        new_cat_dataset = new_cat_group[dataset_name]
                        new_cat_data = new_cat_dataset[()]  # Read the data from the destination dataset
                        combined_data = np.concatenate((new_cat_data, cat_data))
                        
                        # Delete the existing dataset and create a new one with the combined data
                        del new_cat_group[dataset_name]
                        new_cat_group.create_dataset(dataset_name, data=combined_data)

                    # If the dataset doesn't exist in the destination group, create it
                    else:
                        new_cat_group.create_dataset(dataset_name, data=cat_data)

                # Remove the field group from the orginal catalogue.
                del cat_group['FIELD']

                # Remove the orginal catalogue if required.
                if replace == True:
                    os.remove(catalogue)

    return

def weight_to_error(weight_image, error_filename = None):
    """Convert a weight image to an error map.

    Parameters
    ----------
    weight_image (str):
        Path to the weight image to convert.
    error_filename (str):
        Name of the error image to output. If None, append "to_error" to weight filename.
    """

    # Load the weight image and header.
    wht, hdr = fits.open(weight_image, header = True)

    # Convert the weight image to error map.
    err = np.where(wht==0, 0, 1/np.sqrt(wht))

    # If no name for new file given, use weight filename as base.
    if error_filename == None:
        error_filename = f'{weight_image.remove(".fits")}_to_error.fits'

    # Add header keyword to indicate how the error map was generated.
    hdr['CONVERTED'] = ('T', 'Converted to error image from weight image.')

    # Save the error image to a new file.
    fits.writeto(err,error_filename,header=hdr,overwrite=True)

    return

def create_stack(sci_images, wht_images, stack_name = 'pyex_stacked'):
    """Create a stacked image for detection.

    Parameters
    ----------
    sci_images (list):
        A list of science image file paths to stack.
    wht_images (list):
        A list of corresponding weight image file paths.
    stack_name (str):
        Filename of the output stacked image.

    The header used in the final images will be taken from the first image in the corresponding list.
    """

    if len(sci_images) != len(wht_images):
        raise ValueError('The number of science and weight images must be equall.')

    # Get the image size and headers from the first image.
    first_image, sci_hdr = fits.getdata(sci_images[0], header=True)
    wht_hdr = fits.getheader(wht_images[0])

    shape = first_image.data.shape
    stack_sci = np.zeros(shape)
    stack_wht = np.zeros(shape)

    # Stack the images. 
    for sci, wht in zip(sci_images, wht_images):
        wht_ = fits.getdata(wht)
        stack_sci += fits.getdata(sci) * wht_
        stack_wht += wht_

    stack_sci /= stack_wht

    # Save the images.
    fits.writeto(f'{stack_name}_sci.fits', stack_sci, header = sci_hdr, overwrite=True)
    fits.writeto(f'{stack_name}_wht.fits', stack_wht, header = wht_hdr, overwrite=True)

    return



                







