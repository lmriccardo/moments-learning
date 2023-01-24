from multiprocessing import Pool
import avis.utils as utils
from typing import List
import os.path as opath
import libsbml
import os


def sbml_to_raterules(document: libsbml.SBMLDocument) -> libsbml.SBMLDocument:
    """
    Convert the SBML model by replacing all the reactions with
    the corresponding Rate Rules, i.e. ODE. In this way we also
    have that all the parameters local to reactions, now they
    become global to all the model. 

    :param document: a handler to the SBML document
    :return:
    """
    conv_properties = libsbml.ConversionProperties()
    conv_properties.addOption("replaceReactions")
    conversion_result = document.convert(conv_properties)
    if conversion_result != libsbml.LIBSBML_OPERATION_SUCCESS:
        utils.print_errors(document, document.getNumErrors())
        exit(utils.Errors.CONVERSION_ERROR)
    
    return document


def transform(sbmlfile: str, outputfile: str) -> None:
    """
    Load the model, transform the model and finally save the modified model.

    :param sbmlfile:   the absolute path to the SBML model to load
    :param outputfile: the absolute path to the file where to store the new SBML
    :return:
    """
    # Obtain the corresponding document
    document = utils.load_sbml(sbmlfile)

    # Convert all reaction to rate rules
    document = sbml_to_raterules(document)

    # Get the handler to the SBML Model
    model = document.getModel()

    # Save the new model
    utils.save_model(model, outputfile)


def transform_model(sbml_path: str) -> str:
    """
    Transform a SBML model using the transform function

    :param sbml_path: the path to the SBML model
    :return: the path where the transformed model has been saved
    """
    # Obtain the output path
    path, extension = sbml_path.split(".")
    output_path = f"{path}_output.{extension}"

    # Call the transform function to perform the transformation
    transform(sbml_path, output_path)

    return output_path


def convert_one(args: List[any]) -> str:
    """
    Convert one model speciefied by the model_id input parameter

    :param args: a List of arguments for the multiprocessing map function
        This parameter must contains only three elements:
            - prefix_path: the path where to save the model
            - model_id:    the ID of the model in the BioModels database
            - bvalue:      the normalization parameter

    :return: the path where the transformed model has been saved
    """
    # Extrapolate arguments
    prefix_path, model_id = args

    # Download the model
    model_path = utils.download_model(prefix_path, model_id + 1)
    print(f"({model_id}) [*] Dowloaded file {model_path}")

    # Transform the model
    trans_model_path = transform_model(model_path)
    print(f"({model_id}) [*] Transformed model file {trans_model_path}")

    return trans_model_path


def convert_models(prefix_path: str, nmodels: int) -> List[str]:
    """
    Download and converts a bunch of models

    :param prefix_path: the path where to save the downloaded models
    :param nmodels:     the number of models to download and convert
    :return: a list containing all the paths to the transformed models
    """
    output_paths = []
    cpu_count = os.cpu_count()
    for (minc, maxc) in utils.get_ranges(nmodels, cpu_count):
        with Pool(cpu_count) as pool:
            args = list(map(lambda x: (prefix_path, x), range(minc, maxc)))
            trans_model_paths = pool.map(convert_one, args)
            output_paths += trans_model_paths
    
    return output_paths


def main() -> None:
    prefix_path = opath.join(os.getcwd(), "tests")
    nmodels = 1

    paths = convert_models(prefix_path, nmodels)
    utils.write_paths(paths, opath.join(os.getcwd(), "data/paths.txt"))
    utils.remove_original(paths)


if __name__ == "__main__":
    main()